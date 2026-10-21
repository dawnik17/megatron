"""Gemma 4 E4B: what bridge/models/gemma4/train.py and its training config change, and how the model is loaded."""

import copy
import os

from megatron.bridge.models.gemma import gemma4_provider, modeling_gemma4 as gm
from megatron.bridge.recipes.gemma import gemma4_e4b_pretrain_config
from megatron.core.transformer import attention
from megatron.core.transformer.dot_product_attention import DotProductAttention

from .base import BRIDGE, Config

ROPE = (attention, "apply_rotary_pos_emb")

CHANGED = [
    (gm, "_gemma4_rms_norm"),
    (gm, "_logit_softcapping"),
    (gm.Gemma4DenseSelfAttention, "_v_norm"),
    (DotProductAttention, "forward"),
    # runs once, when setup() builds the model: the gradient check covers it
    (gemma4_provider.Gemma4DenseProvider, "build"),
    ROPE,
]


class Gemma4E4BConfig(Config):
    def __init__(self, changes=CHANGED):
        super().__init__(
            gemma4_e4b_pretrain_config,
            changes,
            BRIDGE / "models" / "gemma4",
            "google/gemma-4-E4B-it",
            seq_length=1024,  # 4096 needs ~110 GB; an A100 has 80
        )

    def load_model(self, model):
        # the -it checkpoint is multimodal; convert its text model only
        os.environ["GEMMA4_CONVERSION_MODE"] = "text"
        super().load_model(model)

    def config_flag_changes(self, provider, ootb):
        # Fused RoPE: the training config's model.apply_rope_fusion=true
        fused = copy.copy(provider)
        fused.apply_rope_fusion = True
        return {ROPE: lambda **kwargs: ootb[ROPE](**{**kwargs, "config": fused})}
