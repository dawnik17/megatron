#!/usr/bin/env python3
"""Tokenize the FineWeb-Edu shard that training and validation use, once, into Megatron's indexed format.

Runs in the NeMo container, from the repo root (it needs the network once, for the 2.15 GB shard). The container sees
the repo at /repo, wherever it is on the host.
  enroot start -e HOME=/tmp -e HF_HOME=/repo/.cache/hf -e NVIDIA_VISIBLE_DEVICES=none \
    -e PYTHONPATH=/opt/Megatron-Bridge/src:/opt/Megatron-Bridge/3rdparty/Megatron-LM \
    -m "$PWD:/repo" pyxis_nemo-26-08 python /repo/bridge/data/prepare.py

The shard is pinned to a Hub revision and every document becomes <bos> text <eos>, so the output is the same on every
machine: 726,000 documents, 745,470,669 tokens with Gemma 4's tokenizer, in
.cache/data/fineweb-edu/google--gemma-4-E4B-it/fineweb-edu_text_document.{bin,idx}.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SOURCE = dict(
    repo="HuggingFaceFW/fineweb-edu",
    revision="87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
    files=["sample/10BT/000_00000.parquet"],
    text="text",
)
TOKENIZER = "google/gemma-4-E4B-it"
PREPROCESS = "/opt/Megatron-Bridge/3rdparty/Megatron-LM/tools/preprocess_data.py"
REPO = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--out", type=Path, default=REPO / ".cache/data/fineweb-edu" / TOKENIZER.replace("/", "--")
    )
    out = p.parse_args().out
    out.mkdir(parents=True, exist_ok=True)

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from megatron.core.datasets.indexed_dataset import IndexedDataset
    from transformers import AutoTokenizer

    # preprocess_data.py appends EOD but adds no BOS; the tokenizer turns the literal BOS token at the start into its id.
    bos = AutoTokenizer.from_pretrained(TOKENIZER).bos_token
    jsonl = out / "documents.jsonl"

    with open(jsonl, "w") as f:
        for name in SOURCE["files"]:
            path = hf_hub_download(
                SOURCE["repo"], name, repo_type="dataset", revision=SOURCE["revision"]
            )
            for batch in pq.ParquetFile(path).iter_batches(
                columns=[SOURCE["text"]], batch_size=10_000
            ):
                f.writelines(
                    json.dumps({"text": bos + text}) + "\n" for text in batch.column(0).to_pylist()
                )
    subprocess.run(
        [
            sys.executable,
            PREPROCESS,
            "--input",
            jsonl,
            "--output-prefix",
            out / "fineweb-edu",
            "--tokenizer-type",
            "HuggingFaceTokenizer",
            "--tokenizer-model",
            TOKENIZER,
            "--append-eod",
            "--workers",
            str(min(64, os.cpu_count())),
        ],
        check=True,
    )
    jsonl.unlink()

    tokenized = IndexedDataset(str(out / "fineweb-edu_text_document"))
    manifest = dict(
        SOURCE,
        name="fineweb-edu",
        tokenizer=TOKENIZER,
        bos=True,
        eod=True,
        documents=len(tokenized),
        tokens=int(tokenized.sequence_lengths.sum()),
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{manifest['documents']:,} documents, {manifest['tokens']:,} tokens -> {out}")


if __name__ == "__main__":
    main()
