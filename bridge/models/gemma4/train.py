#!/usr/bin/env python3
"""Launch Megatron-Bridge's run_recipe.py with runtime speed patches for Gemma 4 E4B.

Nothing in Megatron-Bridge is edited: the patches below replace functions in memory,
in every training process, before the model is built. Then run_recipe.py runs unchanged.
Each patch computes the same function as the code it replaces. Gradients are accumulated and
reduced in FP32 (the bf16_mixed preset's grad_reduce_in_fp32=True), as in Megatron's BF16 training.

Patches (all on by default; set the variable to 0, or the attention ones to local, to turn one off):
  GEMMA4_FAST_NORM=1     torch.compile the Gemma 4 RMSNorm and the value norm (same math,
                         one fused kernel instead of ~6 FP32 elementwise kernels per norm)
  GEMMA4_FAST_SOFTCAP=1  torch.compile the final-logit softcapping (30 * tanh(x / 30))
  GEMMA4_SLIDING_ATTN=fa2   kernel for the 35 sliding-window layers (head_dim 256, 512-token window):
                         fa2 (FlashAttention 2) or local (Megatron's unfused 4096x4096 attention)
  GEMMA4_GLOBAL_ATTN=te     kernel for the 7 global layers (head_dim 512): te (Transformer Engine's
                         unfused attention with a fused masked softmax) or local (Megatron's)
  GEMMA4_FREEZE_UNUSED=1 freeze k_layernorm in the 18 KV-shared layers (their output is discarded,
                         so they never get a gradient); ddp.overlap_grad_reduce=true needs it

Recipes in bridge/models/gemma4/<size>/recipe.py (e.g. 12B/recipe.py) are added to Megatron-Bridge's recipes, so
--recipe finds them like a shipped one.

Usage: same arguments as run_recipe.py (bridge/models/launch.py builds them from a config), e.g.
  python -m torch.distributed.run --nproc_per_node 4 bridge/models/gemma4/train.py \
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
        return (vf * torch.pow(vf.pow(2).mean(-1, keepdim=True) + 1e-6, -0.5)).to(value.dtype)

    gm.Gemma4DenseSelfAttention._v_norm = lambda self, value: _v_norm(value)
    _log("RMSNorm + value norm: torch.compile")


def patch_softcap(gm) -> None:
    """Compile the final-logit softcapping applied to the [tokens, 262144] logits."""
    gm._logit_softcapping = torch.compile(gm._logit_softcapping, dynamic=False)
    _log("final logit softcapping: torch.compile")


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


def patch_attention(sliding: str, global_: str) -> None:
    """Route the local DotProductAttention to faster kernels, per layer type: the 35 sliding-window layers
    (head_dim 256) to FlashAttention 2 ("fa2") and the 7 global layers (head_dim 512) to Transformer Engine
    ("te"); "local" keeps Megatron's unfused attention for that layer type."""
    from megatron.core.transformer.dot_product_attention import DotProductAttention

    te_modules = (
        {}
    )  # id(module) -> TE module; kept out of the model so its state_dict doesn't change
    used = set()
    original_forward = DotProductAttention.forward

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
        kind = sliding if window is not None else global_
        plain = (
            attention_mask is None
            and attention_bias is None
            and packed_seq_params is None
            and getattr(self, "softcap", None) is None
            and self.softmax_offset is None
            and self.scale_mask_softmax.scale is None
            and (not self.training or self.attention_dropout.p == 0.0)
            # TE's causal mask is aligned top-left and fa2's causal=True drops a right-hand window, so both
            # match Megatron only for self-attention without look-ahead.
            and query.size(0) == key.size(0)
            and (window is None or window[1] == 0)
        )
        if kind == "local" or not plain:
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

        if kind not in used:
            used.add(kind)
            layer_type = "sliding-window" if window is not None else "global"
            _log(
                f"{layer_type} attention: {kind} path in use (seq {query.size(0)}, head_dim {query.size(3)}, "
                f"window {window})"
            )

        if kind == "fa2":
            return fa2_attention(query, key, value, window, self.softmax_scale)
        module = te_modules.get(id(self))
        if module is None:
            module = te_modules[id(self)] = te_causal_attention(
                query.size(2), query.size(3), key.size(2), self.softmax_scale
            )
        return module(query, key, value)

    DotProductAttention.forward = forward
    _log(f"attention: sliding layers {sliding}, global layers {global_}")


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
            if isinstance(module, Gemma4DenseSelfAttention) and module.is_kv_shared_layer:
                if module.k_layernorm is not None:
                    for p in module.k_layernorm.parameters():
                        p.requires_grad_(False)
                        frozen += 1
        _log(f"froze {frozen} unused k_layernorm weights in KV-shared layers")
        return model

    gemma4_provider.Gemma4DenseProvider.build = build


def register_recipes() -> None:
    """Add the recipes in bridge/models/gemma4/<size>/recipe.py (each module's __all__) to megatron.bridge.recipes,
    where run_recipe.py's --recipe looks them up. Importing a recipe module also registers its bridge."""
    import importlib
    from pathlib import Path

    import megatron.bridge.recipes as recipes

    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parents[2]))  # the repo root: bridge.models.gemma4.<size>.recipe
    for path in sorted(here.glob("*/recipe.py")):
        module = importlib.import_module(f"bridge.models.gemma4.{path.parent.name}.recipe")
        for name in module.__all__:
            setattr(recipes, name, getattr(module, name))
            _log(f"recipe {name} from {path.relative_to(here)}")


def main() -> None:
    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)

    from megatron.bridge.models.gemma import modeling_gemma4 as gm

    if _enabled("GEMMA4_FAST_NORM"):
        patch_norms(gm)
    if _enabled("GEMMA4_FAST_SOFTCAP"):
        patch_softcap(gm)

    sliding = os.environ.get("GEMMA4_SLIDING_ATTN", "fa2")
    global_ = os.environ.get("GEMMA4_GLOBAL_ATTN", "te")
    if sliding not in ("fa2", "local") or global_ not in ("te", "local"):
        sys.exit(
            f"GEMMA4_SLIDING_ATTN must be fa2 or local and GEMMA4_GLOBAL_ATTN te or local; "
            f"got {sliding!r} and {global_!r}"
        )
    if (sliding, global_) != ("local", "local"):
        patch_attention(sliding, global_)

    if _enabled("GEMMA4_FREEZE_UNUSED"):
        patch_freeze_unused_kv_norm()

    register_recipes()

    bridge_root = os.environ.get("BRIDGE_ROOT", "/opt/Megatron-Bridge")
    run_recipe = os.path.join(bridge_root, "scripts", "training", "run_recipe.py")
    sys.argv = [run_recipe] + sys.argv[1:]
    runpy.run_path(run_recipe, run_name="__main__")


if __name__ == "__main__":
    main()
