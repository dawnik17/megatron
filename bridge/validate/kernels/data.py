import sys
from pathlib import Path

from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "data"))
from prepare import SOURCE  # noqa: E402


def text_tokens(hf_tokenizer_name, seq_length):
    """The first seq_length tokens of FineWeb-Edu text, from the shard prepare.py pins."""
    parquet = hf_hub_download(
        SOURCE["repo"], SOURCE["files"][0], repo_type="dataset", revision=SOURCE["revision"]
    )
    texts = (
        next(pq.ParquetFile(parquet).iter_batches(batch_size=64, columns=[SOURCE["text"]]))
        .column(0)
        .to_pylist()
    )
    tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_name)
    ids = tokenizer("\n\n".join(texts))["input_ids"][:seq_length]
    if len(ids) < seq_length:
        raise RuntimeError(f"only {len(ids)} tokens of text; need {seq_length}")
    return torch.tensor([ids], device="cuda")
