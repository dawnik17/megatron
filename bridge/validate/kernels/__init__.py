"""Functional check, kernel level: every kernel our changes swap in computes the same function as the one Megatron-Bridge
runs out of the box (OOTB), forward and backward, and is at least as accurate.

Each kernel runs on the same BF16 inputs as the OOTB one it replaces, and both are compared with the exact result: the
OOTB code itself run in float64, so every detail of Megatron's semantics (mask, window, softmax scale, GQA layout) is in
the reference. The inputs are the ones each patched function receives in one forward pass of the pretrained model on
FineWeb-Edu text, one call per distinct shape. The check also builds the model with train.py's patches and runs one
forward and backward: the parameters left without a gradient must be exactly the ones train.py freezes, with and
without the patches.

Runs in the container on one GPU and writes the JSON bridge/validate/run.py reads:
  python bridge/validate/kernels --out kernels.json [--model NAME] [--only FUNCTION ...]
Everything model-specific is a Config in config/.
"""

import argparse
import json
from pathlib import Path

import torch

from .data import text_tokens
from .eval import MARGIN, compare
from .grad import check_gradients
from .inputs import capture
from .config import MODELS


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", required=True, help="JSON file for the results")
    p.add_argument("--model", default="gemma4-e4b", choices=MODELS)
    p.add_argument(
        "--only",
        nargs="+",
        default=[],
        metavar="FUNCTION",
        help="compare only these changed functions, by attribute name",
    )
    args = p.parse_args()
    config = MODELS[args.model]()
    unknown = set(args.only) - {name for _, name in config.changes}
    if unknown:
        p.error(f"--only: {sorted(unknown)} not among {args.model}'s changes")

    provider, model, wrapped, ootb, ours = config.setup()
    gradients = check_gradients(model, wrapped, ootb, ours, config.seq_length)

    config.load_model(model)
    tokens = text_tokens(config.hf, config.seq_length)
    torch.manual_seed(0)

    ours.update(config.config_flag_changes(provider, ootb))
    calls = capture(wrapped, tokens, ootb)

    torch.cuda.empty_cache()  # the softcap check's 8.6 GB float64 tensors don't fit the blocks cached so far
    results = []
    for key, bound_dict in calls.items():
        if args.only and key[1] not in args.only:
            continue
        for shape, bound in bound_dict.items():
            name = f"{key[1]} {[s for s in shape if isinstance(s, tuple)]}"
            results.append(compare(name, ootb[key], ours[key], bound))

    Path(args.out).write_text(
        json.dumps(dict(margin=MARGIN, gradients=gradients, kernels=results), indent=1) + "\n"
    )
