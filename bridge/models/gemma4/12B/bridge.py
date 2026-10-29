"""Text-tower bridge for Gemma 4 12B (`google/gemma-4-12B`), registered on import.

The 12B checkpoint is `Gemma4UnifiedForConditionalGeneration` (model_type `gemma4_unified`), an encoder-free
text + vision + audio model. Megatron-Bridge 0.6.1 registers no bridge for it, so AutoBridge rejects it. Its
`text_config` has every field Gemma4Bridge's dense path reads, so this bridge reuses the dense provider, weight map,
K=V handling and logit-softcapping output layer; only two things differ:
  - the text fields sit under `text_config`, and
  - the HF weights sit under `model.language_model.*` instead of `model.*`.
The vision (`model.embed_vision.*`) and audio (`model.embed_audio.*`) towers are not mapped.
"""

from megatron.core.models.gpt import GPTModel

from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.gemma.gemma4_bridge import Gemma4Bridge
from megatron.bridge.models.gemma.gemma4_provider import Gemma4DenseProvider


@MegatronModelBridge.register_bridge(
    source="Gemma4UnifiedForConditionalGeneration",
    target=GPTModel,
    provider=Gemma4DenseProvider,
    model_type="gemma4_unified",
)
class Gemma4UnifiedBridge(Gemma4Bridge):
    """Gemma 4 12B's language model as a dense Gemma 4 GPTModel (PP=1 only, like Gemma4DenseProvider)."""

    def provider_bridge(self, hf_pretrained) -> Gemma4DenseProvider:
        text_config = getattr(hf_pretrained.config, "text_config", hf_pretrained.config)
        self._is_dense = True
        self._unified_text_config = text_config
        return self._build_dense_provider(text_config)

    def _text_config(self):
        return getattr(self, "_unified_text_config", None) or getattr(self, "hf_config", None)

    def _hf_layer_prefix(self) -> str:
        return "model.language_model."
