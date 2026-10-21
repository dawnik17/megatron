#!/usr/bin/env python3
"""Time and check the attention kernels train.py uses against Megatron's own, forward and backward, on one GPU.

    python bridge/profile/attn_bench.py [--mbs 4]

global : 7 layers,  head_dim 512, causal                          Megatron's unfused attention vs te
sliding: 35 layers, head_dim 256, causal with a 512-token window  Megatron's unfused attention vs fa2
Both use 8 query heads, 2 KV heads, sequence 4096, BF16 and softmax scale 1.0 (Gemma 4 normalizes Q and K), in
Megatron's [seq, batch, heads, dim] layout. "err" is the largest absolute error of the output and of the input
gradients against an FP32 reference, divided by the largest reference value.
"""

import argparse

import torch

SEQ, HEADS, KV_HEADS = 4096, 8, 2
LAYERS = {
    "global": (512, None, "te"),
    "sliding": (256, (511, 0), "fa2"),
}  # head_dim, window, train.py's kernel


def fa2_attention(query, key, value, window, scale):
    """Causal sliding-window attention with FlashAttention 2: [s, b, h, d] query, [s, b, g, d] key/value ->
    [s, b, h * d]. It reads Megatron's layout through strides, without copies."""
    from flash_attn import flash_attn_func

    sq, b, np_, hn = query.shape
    out = flash_attn_func(
        query.transpose(0, 1),
        key.transpose(0, 1),
        value.transpose(0, 1),
        softmax_scale=scale,
        causal=True,
        window_size=(int(window[0]), int(window[1])),
    )
    return out.transpose(0, 1).reshape(sq, b, np_ * hn)


def te_causal_attention(num_heads, head_dim, num_gqa_groups, scale):
    """Transformer Engine attention for [s, b, h, d] inputs. For head_dim 512 TE has no fused backend and runs
    its unfused one: batched matrix multiplies and a fused softmax with a causal mask.
    """
    import transformer_engine.pytorch as te

    return te.DotProductAttention(
        num_heads,
        head_dim,
        num_gqa_groups=num_gqa_groups,
        attention_dropout=0.0,
        qkv_format="sbhd",
        attn_mask_type="causal",
        softmax_scale=scale,
    )


def megatron_attention(head_dim, window):
    """Megatron's unfused DotProductAttention, which the model uses without train.py's attention patch."""
    from megatron.core import parallel_state, tensor_parallel
    from megatron.core.transformer.dot_product_attention import DotProductAttention
    from megatron.core.transformer.enums import AttnMaskType
    from megatron.core.transformer.transformer_config import TransformerConfig

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            "nccl", init_method="tcp://127.0.0.1:29790", world_size=1, rank=0
        )
        parallel_state.initialize_model_parallel(1, 1)
        tensor_parallel.model_parallel_cuda_manual_seed(1234)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=2560,
        ffn_hidden_size=10240,
        num_attention_heads=HEADS,
        num_query_groups=KV_HEADS,
        kv_channels=head_dim,
        bf16=True,
        params_dtype=torch.bfloat16,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        attention_softmax_in_fp32=False,
        apply_query_key_layer_scaling=False,
        masked_softmax_fusion=False,
        window_size=window,
        window_attn_skip_freq=None,
    )
    module = DotProductAttention(
        config=config,
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        softmax_scale=1.0,
    )
    return lambda q, k, v: module(q, k, v, attention_mask=None)


def build(kernel, head_dim, window):
    if kernel == "local":
        return megatron_attention(head_dim, window)
    if kernel == "fa2":
        return lambda q, k, v: fa2_attention(q, k, v, window, 1.0)
    module = te_causal_attention(HEADS, head_dim, KV_HEADS, 1.0)
    return lambda q, k, v: module(q, k, v)


def reference(q, k, v, window):
    """FP32 attention in [seq, batch, heads, dim], with the causal (and window) mask."""
    q, k, v = (t.float().permute(1, 2, 0, 3) for t in (q, k, v))
    k, v = (t.repeat_interleave(HEADS // KV_HEADS, dim=1) for t in (k, v))
    i = torch.arange(SEQ, device=q.device)
    allowed = i[None, :] <= i[:, None]
    if window:
        allowed &= i[None, :] >= i[:, None] - window[0]
    out = (q @ k.transpose(-1, -2)).masked_fill(~allowed, float("-inf")).softmax(-1) @ v
    b, h, s, d = out.shape
    return out.permute(2, 0, 1, 3).reshape(s, b, h * d)


def measure(kernel, head_dim, window, mbs):
    """Median forward and backward time over 10 calls, and the errors against the FP32 reference."""
    torch.manual_seed(0)
    q = torch.randn(
        SEQ, mbs, HEADS, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    k = torch.randn(
        SEQ, mbs, KV_HEADS, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    v = torch.randn(
        SEQ, mbs, KV_HEADS, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    grad = torch.randn(SEQ, mbs, HEADS * head_dim, device="cuda", dtype=torch.bfloat16)

    fn = build(kernel, head_dim, window)
    for _ in range(3):
        fn(q, k, v).backward(grad)
    fwd, bwd = [], []
    for _ in range(10):
        q.grad = k.grad = v.grad = None
        events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        events[0].record()
        out = fn(q, k, v)
        events[1].record()
        out.backward(grad)
        events[2].record()
        torch.cuda.synchronize()
        fwd.append(events[0].elapsed_time(events[1]))
        bwd.append(events[1].elapsed_time(events[2]))
    got = [out.detach(), q.grad, k.grad, v.grad]

    qr, kr, vr = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    ref_out = reference(qr, kr, vr, window)
    ref_out.backward(grad.float())
    refs = [ref_out.detach(), qr.grad, kr.grad, vr.grad]
    err = [((g.float() - r).abs().max() / r.abs().max()).item() for g, r in zip(got, refs)]
    return sorted(fwd)[5], sorted(bwd)[5], err


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mbs", type=int, default=4)
    args = p.parse_args()

    print(f"micro batch {args.mbs}, sequence {SEQ}; time per call, median of 10")
    print(
        f"{'layer':8s} {'kernel':8s} {'fwd ms':>8s} {'bwd ms':>8s} {'total':>8s}   err out / dq / dk / dv"
    )
    for layer, (head_dim, window, kernel) in LAYERS.items():
        for name in ("local", kernel):
            fwd, bwd, err = measure(name, head_dim, window, args.mbs)
            errs = " / ".join(f"{e:.1e}" for e in err)
            print(
                f"{layer:8s} {name:8s} {fwd:8.2f} {bwd:8.2f} {fwd + bwd:8.2f}   {errs}", flush=True
            )


if __name__ == "__main__":
    main()
