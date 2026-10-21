#!/usr/bin/env python3
"""Launch Megatron-Bridge's run_recipe.py with runtime speed patches for Gemma 4 E4B.

Nothing in Megatron-Bridge is edited: the patches below replace functions in memory,
in every training process, before the model is built. Then run_recipe.py runs unchanged.

Patches (the first three are on by default; set the variable to 0 to turn one off):
  GEMMA4_FAST_NORM=1     torch.compile the Gemma 4 RMSNorm and the value norm (same math,
                         one fused kernel instead of ~6 FP32 elementwise kernels per norm)
  GEMMA4_FAST_SOFTCAP=1  torch.compile the final-logit softcapping (30 * tanh(x / 30))
  GEMMA4_FAST_ATTN=1     sliding-window layers use flex_attention with a block-sparse
                         512-token window instead of the unfused 4096x4096 attention
                         (global layers, head_dim 512, keep the original implementation)
  GEMMA4_FREEZE_UNUSED=0 (off by default) set to 1 to freeze k_layernorm in the 18 KV-shared
                         layers (their output is discarded, so they never get a gradient);
                         needed only if you turn on ddp.overlap_grad_reduce=true

Usage: same arguments as run_recipe.py, e.g.
  python -m torch.distributed.run --nproc_per_node 4 train/gemma/megatron_mfu_optim/train.py \
      --recipe gemma4_e4b_pretrain_config --mode pretrain --dataset mock ...
"""

import os
import runpy
import sys

import torch


def _enabled(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default) != "0"


def _log(msg: str) -> None:
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"[gemma4-fast] {msg}", flush=True)


def patch_norms(gm) -> None:
    """Compile Gemma 4's RMSNorm expression and the scaleless value norm (same math)."""
    gm._gemma4_rms_norm = torch.compile(gm._gemma4_rms_norm, dynamic=False)

    @torch.compile(dynamic=False)
    def _v_norm(value):
        vf = value.float()
        return (vf * torch.pow(vf.pow(2).mean(-1, keepdim=True) + 1e-6, -0.5)).to(
            value.dtype
        )

    gm.Gemma4DenseSelfAttention._v_norm = lambda self, value: _v_norm(value)
    _log("RMSNorm + value norm: torch.compile")


def patch_softcap(gm) -> None:
    """Compile the final-logit softcapping applied to the [tokens, 262144] logits."""
    gm._logit_softcapping = torch.compile(gm._logit_softcapping, dynamic=False)
    _log("final logit softcapping: torch.compile")


def patch_sliding_attention() -> None:
    """Route sliding-window layers of the local DotProductAttention to flex_attention."""
    from megatron.core.transformer.dot_product_attention import DotProductAttention
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    flex = torch.compile(flex_attention, dynamic=False)
    block_masks = {}
    used = []
    original_forward = DotProductAttention.forward

    def _block_mask(sq: int, sk: int, left: int, right: int, device):
        key = (sq, sk, left, right, device)
        if key not in block_masks:
            offset = sk - sq

            # Same rule as megatron.core.transformer.utils.get_sliding_window_causal_mask:
            # query i attends to keys j with  i + offset - left <= j <= i + offset + right.
            def window(b, h, q_idx, kv_idx):
                return (kv_idx >= q_idx + offset - left) & (
                    kv_idx <= q_idx + offset + right
                )

            block_masks[key] = create_block_mask(
                window, None, None, sq, sk, device=device
            )
        return block_masks[key]

    def forward(
        self,
        query,
        key,
        value,
        attention_mask,
        attn_mask_type=None,
        attention_bias=None,
        packed_seq_params=None,
    ):
        window = self.scale_mask_softmax.window_size
        usable = (
            window is not None
            and attention_mask is None
            and attention_bias is None
            and packed_seq_params is None
            and getattr(self, "softcap", None) is None
            and self.softmax_offset is None
            and self.scale_mask_softmax.scale is None
            and (not self.training or self.attention_dropout.p == 0.0)
        )
        if not usable:
            return original_forward(
                self,
                query,
                key,
                value,
                attention_mask,
                attn_mask_type,
                attention_bias,
                packed_seq_params,
            )

        sq, b, np_, hn = query.shape
        sk = key.size(0)
        if not used:
            used.append(True)
            _log(
                f"sliding-window attention: flex path in use (seq {sq}, window {window})"
            )
        # [s, b, heads, d] -> [b, heads, s, d]
        q = query.permute(1, 2, 0, 3).contiguous()
        k = key.permute(1, 2, 0, 3).contiguous()
        v = value.permute(1, 2, 0, 3).contiguous()
        out = flex(
            q,
            k,
            v,
            block_mask=_block_mask(sq, sk, window[0], window[1], query.device),
            scale=self.softmax_scale,
            enable_gqa=np_ != k.size(1),
        )
        # [b, heads, s, d] -> [s, b, heads * d]
        return out.permute(2, 0, 1, 3).reshape(sq, b, np_ * hn)

    DotProductAttention.forward = forward
    _log("sliding-window attention: flex_attention (global layers unchanged)")


def patch_freeze_unused_kv_norm() -> None:
    """Freeze k_layernorm in KV-shared layers: their K is computed and discarded, so the
    weight never gets a gradient. With it frozen, ddp.overlap_grad_reduce=true works
    (otherwise the first step fails on `per_param_grad_ready_counts == len(params)`)."""
    from megatron.bridge.models.gemma import gemma4_provider
    from megatron.bridge.models.gemma.modeling_gemma4 import Gemma4DenseSelfAttention

    original_build = gemma4_provider.Gemma4DenseProvider.build

    def build(self, *args, **kwargs):
        model = original_build(self, *args, **kwargs)
        frozen = 0
        for module in model.modules():
            if (
                isinstance(module, Gemma4DenseSelfAttention)
                and module.is_kv_shared_layer
            ):
                if module.k_layernorm is not None:
                    for p in module.k_layernorm.parameters():
                        p.requires_grad_(False)
                        frozen += 1
        _log(f"froze {frozen} unused k_layernorm weights in KV-shared layers")
        return model

    gemma4_provider.Gemma4DenseProvider.build = build


def main() -> None:
    torch._dynamo.config.cache_size_limit = max(
        torch._dynamo.config.cache_size_limit, 64
    )

    from megatron.bridge.models.gemma import modeling_gemma4 as gm

    if _enabled("GEMMA4_FAST_NORM"):
        patch_norms(gm)
    if _enabled("GEMMA4_FAST_SOFTCAP"):
        patch_softcap(gm)
    if _enabled("GEMMA4_FAST_ATTN"):
        patch_sliding_attention()
    if _enabled("GEMMA4_FREEZE_UNUSED", default="0"):
        patch_freeze_unused_kv_norm()

    bridge_root = os.environ.get("BRIDGE_ROOT", "/opt/Megatron-Bridge")
    run_recipe = os.path.join(bridge_root, "scripts", "training", "run_recipe.py")
    sys.argv = [run_recipe] + sys.argv[1:]
    runpy.run_path(run_recipe, run_name="__main__")


if __name__ == "__main__":
    main()
