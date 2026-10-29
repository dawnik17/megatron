# gemma4/12B

Gemma 4 12B's text tower on Megatron-Bridge.

```
12B/
├── bridge.py     registers Gemma4UnifiedForConditionalGeneration (google/gemma-4-12B-it) with AutoBridge
├── recipe.py     our pre-training recipe, gemma4_12b_pretrain_h200_bf16_config (train.py adds it to the recipes)
└── parity.py     layer-by-layer parity of the Megatron model against Hugging Face's
```

## Parity

`parity.py` loads `google/gemma-4-12B-it` twice, as the Megatron model this folder's bridge builds and as
Hugging Face's model. It runs both on the same 4096 random text-token ids and prints the per-token cosine similarity
of every decoder layer's output (48 lines), then of the logits. It needs the checkpoint in the repo's HF cache
(`.cache/hf`, which the container sees as `/repo/.cache/hf`) and one free 80 GB GPU; on an A100 it takes about
2 minutes.

It runs as a module, `bridge.models.gemma4.12B.parity`, so it must be started **from the repo root**: the folder
that holds `bridge/`, which is `/repo` in the container and `/sadata/nikkothari/flipkart/megatron` on the host.

**Inside the container**, from `/repo`:

```bash
cd /repo
export HF_HOME=/repo/.cache/hf HF_HUB_OFFLINE=1 \
  PYTHONPATH=/opt/Megatron-Bridge/src:/opt/Megatron-Bridge/3rdparty/Megatron-LM
python -m torch.distributed.run --standalone --nproc-per-node 1 -m bridge.models.gemma4.12B.parity
```

**From the host**, from the repo root (it starts the container on GPU 7 and runs the same command in `/repo`):

```bash
cd /sadata/nikkothari/flipkart/megatron
enroot start -e NVIDIA_VISIBLE_DEVICES=7 -e HOME=/tmp -e HF_HOME=/repo/.cache/hf -e HF_HUB_OFFLINE=1 \
  -e PYTHONPATH=/opt/Megatron-Bridge/src:/opt/Megatron-Bridge/3rdparty/Megatron-LM \
  -m $PWD:/repo pyxis_nemo-26-08 \
  bash -c 'cd /repo && python -m torch.distributed.run --standalone --nproc-per-node 1 -m bridge.models.gemma4.12B.parity'
```

`--standalone` picks a free rendezvous port, so the run doesn't collide with other jobs on the node.

```
Script Output:
Per Layer Per Token Cosine Similarity

Layer 0 Cosine Similarity: tensor([[1.0000, 1.0000, 1.0000,  ..., 1.0000, 1.0000, 1.0000]])
Layer 1 Cosine Similarity: tensor([[1.0000, 1.0000, 1.0000,  ..., 1.0000, 1.0000, 1.0000]])
Layer 2 Cosine Similarity: tensor([[1.0000, 1.0000, 1.0000,  ..., 1.0000, 1.0000, 1.0000]])
Layer 3 Cosine Similarity: tensor([[1.0000, 0.9999, 1.0000,  ..., 1.0000, 1.0000, 1.0000]])
Layer 4 Cosine Similarity: tensor([[0.9999, 0.9999, 0.9999,  ..., 0.9999, 1.0000, 1.0000]])
Layer 5 Cosine Similarity: tensor([[0.9998, 0.9997, 0.9999,  ..., 0.9999, 1.0000, 0.9999]])
Layer 6 Cosine Similarity: tensor([[0.9998, 0.9997, 0.9998,  ..., 0.9999, 0.9999, 0.9999]])
Layer 7 Cosine Similarity: tensor([[0.9998, 0.9998, 0.9999,  ..., 0.9999, 0.9999, 0.9999]])
Layer 8 Cosine Similarity: tensor([[0.9998, 0.9998, 0.9999,  ..., 0.9999, 0.9999, 0.9999]])
Layer 9 Cosine Similarity: tensor([[0.9998, 0.9998, 0.9999,  ..., 0.9999, 0.9999, 0.9999]])
Layer 10 Cosine Similarity: tensor([[0.9998, 0.9999, 0.9999,  ..., 0.9999, 0.9999, 1.0000]])
Layer 11 Cosine Similarity: tensor([[0.9997, 0.9997, 0.9996,  ..., 0.9999, 0.9999, 1.0000]])
Layer 12 Cosine Similarity: tensor([[0.9996, 0.9997, 0.9997,  ..., 1.0000, 1.0000, 1.0000]])
Layer 13 Cosine Similarity: tensor([[0.9999, 0.9999, 1.0000,  ..., 1.0000, 1.0000, 1.0000]])
Layer 14 Cosine Similarity: tensor([[0.9999, 0.9999, 0.9999,  ..., 1.0000, 1.0000, 1.0000]])
Layer 15 Cosine Similarity: tensor([[0.9999, 0.9999, 0.9999,  ..., 1.0000, 1.0000, 1.0000]])
Layer 16 Cosine Similarity: tensor([[0.9999, 1.0000, 1.0000,  ..., 1.0000, 1.0000, 1.0000]])
Layer 17 Cosine Similarity: tensor([[0.9997, 0.9998, 0.9999,  ..., 1.0000, 1.0000, 1.0000]])
Layer 18 Cosine Similarity: tensor([[0.9997, 0.9997, 0.9998,  ..., 1.0000, 1.0000, 1.0000]])
Layer 19 Cosine Similarity: tensor([[0.9996, 0.9998, 0.9998,  ..., 1.0000, 1.0000, 1.0000]])
Layer 20 Cosine Similarity: tensor([[0.9998, 0.9998, 0.9999,  ..., 1.0000, 1.0000, 1.0000]])
Layer 21 Cosine Similarity: tensor([[0.9999, 0.9999, 0.9999,  ..., 1.0000, 1.0000, 1.0000]])
Layer 22 Cosine Similarity: tensor([[0.9999, 0.9999, 0.9999,  ..., 1.0000, 1.0000, 1.0000]])
Layer 23 Cosine Similarity: tensor([[0.9998, 0.9998, 0.9998,  ..., 1.0000, 1.0000, 1.0000]])
Layer 24 Cosine Similarity: tensor([[0.9999, 0.9998, 0.9998,  ..., 1.0000, 1.0000, 1.0000]])
Layer 25 Cosine Similarity: tensor([[0.9999, 0.9998, 0.9998,  ..., 1.0000, 0.9999, 1.0000]])
Layer 26 Cosine Similarity: tensor([[0.9998, 0.9997, 0.9998,  ..., 1.0000, 0.9999, 0.9999]])
Layer 27 Cosine Similarity: tensor([[0.9997, 0.9995, 0.9998,  ..., 1.0000, 0.9999, 0.9999]])
Layer 28 Cosine Similarity: tensor([[0.9998, 0.9996, 0.9998,  ..., 0.9999, 0.9999, 0.9999]])
Layer 29 Cosine Similarity: tensor([[0.9996, 0.9992, 0.9994,  ..., 0.9997, 0.9994, 0.9994]])
Layer 30 Cosine Similarity: tensor([[0.9995, 0.9990, 0.9990,  ..., 0.9996, 0.9991, 0.9991]])
Layer 31 Cosine Similarity: tensor([[0.9993, 0.9987, 0.9986,  ..., 0.9997, 0.9992, 0.9993]])
Layer 32 Cosine Similarity: tensor([[0.9997, 0.9995, 0.9994,  ..., 0.9998, 0.9996, 0.9996]])
Layer 33 Cosine Similarity: tensor([[0.9998, 0.9997, 0.9997,  ..., 0.9999, 0.9998, 0.9998]])
Layer 34 Cosine Similarity: tensor([[0.9998, 0.9997, 0.9997,  ..., 0.9999, 0.9998, 0.9998]])
Layer 35 Cosine Similarity: tensor([[0.9997, 0.9996, 0.9995,  ..., 0.9993, 0.9989, 0.9989]])
Layer 36 Cosine Similarity: tensor([[0.9997, 0.9996, 0.9996,  ..., 0.9993, 0.9989, 0.9990]])
Layer 37 Cosine Similarity: tensor([[0.9998, 0.9997, 0.9997,  ..., 0.9994, 0.9990, 0.9992]])
Layer 38 Cosine Similarity: tensor([[0.9999, 0.9998, 0.9998,  ..., 0.9997, 0.9993, 0.9995]])
Layer 39 Cosine Similarity: tensor([[0.9999, 0.9998, 0.9998,  ..., 0.9996, 0.9993, 0.9995]])
Layer 40 Cosine Similarity: tensor([[0.9998, 0.9997, 0.9996,  ..., 0.9993, 0.9987, 0.9990]])
Layer 41 Cosine Similarity: tensor([[0.9994, 0.9993, 0.9991,  ..., 0.9959, 0.9915, 0.9963]])
Layer 42 Cosine Similarity: tensor([[0.9995, 0.9994, 0.9993,  ..., 0.9966, 0.9928, 0.9970]])
Layer 43 Cosine Similarity: tensor([[0.9996, 0.9995, 0.9994,  ..., 0.9972, 0.9944, 0.9977]])
Layer 44 Cosine Similarity: tensor([[0.9997, 0.9996, 0.9996,  ..., 0.9979, 0.9957, 0.9982]])
Layer 45 Cosine Similarity: tensor([[0.9997, 0.9996, 0.9995,  ..., 0.9969, 0.9942, 0.9978]])
Layer 46 Cosine Similarity: tensor([[0.9996, 0.9996, 0.9996,  ..., 0.9966, 0.9936, 0.9976]])
Layer 47 Cosine Similarity: tensor([[0.9998, 0.9998, 0.9998,  ..., 0.9960, 0.9926, 0.9977]])
Logits Cosine Similarity: tensor([[0.9998, 0.9999, 0.9998,  ..., 0.9947, 0.9840, 0.9960]])
```
