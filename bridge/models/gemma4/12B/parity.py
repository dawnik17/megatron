"""Layer-by-layer parity of Gemma 4 12B: the Megatron model built by this folder's bridge against Hugging Face's.

Both models load google/gemma-4-12B-it and run the same 4096 random text-token ids. A forward hook on each of the
48 decoder layers records its output hidden state. The script prints, for every layer, the per-token cosine
similarity between the two models' outputs, then the per-token cosine of their logits (after the final norm and the
30 * tanh(x / 30) softcap). A layer whose cosine falls away from 1 is where the implementations differ.

Two settings keep the comparison like for like:
  - masked_softmax_fusion=False: for 16 < seq <= 4096 MCore's fused causal softmax ignores the 1024-token window,
    which would turn the 40 sliding-window layers into full-attention layers;
  - HF eager attention: the same unfused softmax(QK^T)V as Megatron's local attention.

Runs in the NeMo container on one free 80 GB GPU (about 2 minutes on an A100), from the repo root (the folder that
holds bridge/, /repo in the container), as a module so that `from . import bridge` resolves. See README.md here.
  python -m torch.distributed.run --standalone --nproc-per-node 1 -m bridge.models.gemma4.12B.parity
"""

import torch
from torch.nn.functional import normalize
from transformers import AutoModelForCausalLM, AutoTokenizer

from megatron.bridge import AutoBridge

from . import bridge  # noqa: F401  (registers Gemma4UnifiedForConditionalGeneration with AutoBridge)


MODEL_12B = "google/gemma-4-12B-it"
tokenizer = AutoTokenizer.from_pretrained(MODEL_12B)


def spy(calls, layer_id):
    detach = lambda x: x.detach().clone().cpu()
    def recorded(module, args, output):
        calls[layer_id] = detach(output) if torch.is_tensor(output) else detach(output[0])

    return recorded

# load bridge model
bridge = AutoBridge.from_hf_pretrained(MODEL_12B)
provider = bridge.to_megatron_provider(load_weights=False)
provider.masked_softmax_fusion = False
provider.initialize_model_parallel(seed=0)
model = provider.provide().cuda().eval()
bridge.load_hf_weights([model])
print(model.embedding.word_embeddings.weight)

# load hf model
hf_model = AutoModelForCausalLM.from_pretrained(MODEL_12B, attn_implementation="eager").cuda()
print(hf_model.model.language_model.embed_tokens.weight)

# record outputs of all layers
outputs = {}
n_layers, mlayers, hflayers = len(model.decoder.layers), model.decoder.layers, hf_model.model.language_model.layers
for i in range(n_layers):
    mlayers[i].register_forward_hook(spy(outputs, f"megatron_{i}"))
    hflayers[i].register_forward_hook(spy(outputs, f"huggingface_{i}"))

# query bridge model
batch_size, seq_len = 1, 4096
torch.manual_seed(0)
pos = torch.arange(seq_len).repeat(batch_size, 1).cuda()
ids = torch.randint(low=0, high=len(tokenizer), size=(batch_size, seq_len))
# HF embeds image/video/audio placeholder ids as pad; Megatron's text tower embeds them as themselves.
cfg = hf_model.config
placeholders = torch.tensor([cfg.image_token_id, cfg.video_token_id, cfg.audio_token_id])
ids = torch.where(torch.isin(ids, placeholders), cfg.text_config.pad_token_id, ids).cuda()

with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    megatron_logits = model(input_ids=ids, position_ids=pos, attention_mask=None)

# query hf model
with torch.no_grad():
    hf_logits = hf_model(input_ids=ids, position_ids=pos, attention_mask=None).logits

# compare outputs of all layers
for i in range(n_layers):
    mkey = f"megatron_{i}"
    hfkey = f"huggingface_{i}"

    print(
        f"Layer {i} Cosine Similarity:", 
        (normalize(outputs[mkey].transpose(1, 0).float(), dim=-1) * normalize(outputs[hfkey].float(), dim=-1)).sum(-1))

# compare logits
print(
    "Logits Cosine Similarity:", 
    (normalize(megatron_logits.cpu().float(), dim=-1) * normalize(hf_logits.cpu().float(), dim=-1)).sum(-1))

torch.distributed.destroy_process_group()
