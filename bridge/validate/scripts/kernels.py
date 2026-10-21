#!/usr/bin/env python3
"""Functional check, kernel level: every kernel our changes swap in computes the same function as the one Megatron-Bridge
runs out of the box (OOTB), forward and backward, and is at least as accurate.

Each kernel runs on the same BF16 inputs as the OOTB one it replaces, and both are compared with the exact result: the
OOTB code itself run in float64, so every detail of Megatron's semantics (mask, window, softmax scale, GQA layout) is in
the reference. The inputs are the ones each patched function receives in one forward pass of the pretrained model on
FineWeb-Edu text, one call per distinct shape. The check also builds the model with train.py's patches and runs one
forward and backward: the parameters left without a gradient must be exactly the ones train.py freezes, with and
without the patches.

Runs in the container on one GPU; bridge/validate/run.py starts it and reads the JSON it writes.
"""

import argparse
import copy
import json
import runpy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models" / "gemma4"))
import train  # noqa: E402  the code under test
import inspect
import os

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from megatron.bridge import AutoBridge
from transformers import AutoTokenizer

SEQ = 4096
MARGIN = 1.25  # ours may be up to 25% less accurate than OOTB: a different summation order rounds differently


HF = "google/gemma-4-E4B-it"


def text_tokens():
    """The first SEQ tokens of FineWeb-Edu text, from the shard prepare.py pins."""
    parquet = hf_hub_download(
        "HuggingFaceFW/fineweb-edu",
        "sample/10BT/000_00000.parquet",
        repo_type="dataset",
        revision="87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
    )
    texts = (
        next(pq.ParquetFile(parquet).iter_batches(batch_size=64, columns=["text"]))
        .column("text")
        .to_pylist()
    )
    ids = AutoTokenizer.from_pretrained(HF)("\n\n".join(texts))["input_ids"][:SEQ]
    if len(ids) < SEQ:
        raise RuntimeError(f"only {len(ids)} tokens of text; need {SEQ}")
    return torch.tensor([ids], device="cuda")


# Get "ootb" and "ours" functions
def setup():
    """One-rank Megatron; the OOTB functions and train.py's replacements; the model as training builds it (random
    weights, TP 1) with train.py's patches applied. main() loads the pretrained weights after the gradient check.
    """
    from megatron.bridge.models.gemma import gemma4_provider, modeling_gemma4 as gm
    from megatron.bridge.recipes.gemma import gemma4_e4b_pretrain_config
    from megatron.bridge.training.mixed_precision import get_mixed_precision_config
    from megatron.core import parallel_state, tensor_parallel
    from megatron.core.transformer.dot_product_attention import DotProductAttention
    from megatron.core.transformer.module import Float16Module

    torch.distributed.init_process_group(
        "nccl", init_method="tcp://127.0.0.1:29791", world_size=1, rank=0
    )
    parallel_state.initialize_model_parallel(1, 1)
    tensor_parallel.model_parallel_cuda_manual_seed(1234)

    patched = [
        (gm, "_gemma4_rms_norm"),
        (gm, "_logit_softcapping"),
        (gm.Gemma4DenseSelfAttention, "_v_norm"),
        (DotProductAttention, "forward"),
        (gemma4_provider.Gemma4DenseProvider, "build"),
    ]
    ootb = {key: getattr(*key) for key in patched}
    runpy.run_path = (
        lambda *args, **kwargs: None
    )  # train.main() ends by starting training; stop it there
    train.main()
    ours = {key: getattr(*key) for key in patched}

    cfg = gemma4_e4b_pretrain_config()
    get_mixed_precision_config(cfg.mixed_precision).setup(cfg.model)
    provider = cfg.model
    provider.tensor_model_parallel_size, provider.seq_length = 1, SEQ
    provider.finalize()
    model = (
        provider.provide().cuda()
    )  # the per-layer-embedding modules are built on the CPU, as in training
    return provider, model, Float16Module(provider, model), ootb, ours


def params_without_grad(wrapped, functions):
    """Names of the parameters that get no gradient from one forward and backward on a random sequence."""
    for (owner, name), function in functions.items():
        setattr(owner, name, function)
    tokens = torch.randint(
        0,
        262144,
        (1, SEQ + 1),
        device="cuda",
        generator=torch.Generator("cuda").manual_seed(0),
    )
    logits = wrapped(tokens[:, :-1], torch.arange(SEQ, device="cuda")[None], None)
    F.cross_entropy(logits.flatten(0, 1).float(), tokens[:, 1:].flatten()).backward()
    names = {
        name.removeprefix("module.")
        for name, p in wrapped.named_parameters()
        if p.grad is None
    }
    wrapped.zero_grad(set_to_none=True)
    return names


def run(function, inputs, grad):
    """function's output and the gradients of the inputs that need one."""
    leaves = {
        k: v.detach().requires_grad_(v.requires_grad) if torch.is_tensor(v) else v
        for k, v in inputs.items()
    }
    out = function(**leaves)
    out.backward(grad.to(out.dtype))
    grads = {
        k: x.grad for k, x in leaves.items() if torch.is_tensor(x) and x.requires_grad
    }
    missing = [k for k, g in grads.items() if g is None]
    if missing:
        raise RuntimeError(f"no gradient for {missing}")
    return [out.detach()] + list(grads.values())


def compare(name, ootb, ours, bound):
    """Relative L2 error of each output, OOTB's and ours, against OOTB's code in float64."""
    arguments = bound.arguments
    labels = ["out"] + [
        f"d{n}" for n, v in arguments.items() if torch.is_tensor(v) and v.requires_grad
    ]

    with torch.no_grad():
        probe = ootb(**arguments)
    grad = torch.randn(probe.shape, device="cuda", dtype=probe.dtype)
    del probe

    exact_arguments = {
        k: v.double() if torch.is_tensor(v) and v.is_floating_point() else v
        for k, v in arguments.items()
    }

    exact, base, mine = (
        run(f, args, grad)
        for f, args in ((ootb, exact_arguments), (ootb, arguments), (ours, arguments))
    )

    rel = lambda got, ref: (
        (got.double() - ref).norm() / ref.norm()
    ).item()  # noqa: E731
    rows = [
        dict(output=label, ootb=rel(b, e), ours=rel(m, e))
        for label, e, b, m in zip(labels, exact, base, mine)
    ]

    result = dict(
        kernel=name,
        identical=torch.equal(base[0], mine[0]),
        outputs=rows,
        passed=all(r["ours"] <= MARGIN * r["ootb"] + 1e-6 for r in rows),
    )
    print(
        f"{name:40s} {'PASS' if result['passed'] else 'FAIL'}  "
        + "  ".join(f"{r['output']} {r['ootb']:.2e} -> {r['ours']:.2e}" for r in rows),
        flush=True,
    )
    return result


# Helper function to get the shape of the input tensor
def signature(v):
    return tuple(v.shape) if torch.is_tensor(v) else type(v).__name__


# Tap the model at the changed functions and get the inputs.
# We will use these inputs to check "ours" vs "ootb"!
def spy(calls, original):
    sig = inspect.signature(original)

    def recorded(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        shape = tuple(signature(v) for v in bound.arguments.values())

        if shape not in calls:
            bound.arguments = {
                n: (
                    v.detach().clone().requires_grad_(v.requires_grad)
                    if torch.is_tensor(v)
                    else v
                )
                for n, v in bound.arguments.items()
            }
            calls[shape] = bound

        return original(*args, **kwargs)

    return recorded


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", required=True, help="JSON file for the results")
    out = p.parse_args().out
    from megatron.bridge.models.gemma import modeling_gemma4 as gm
    from megatron.core.transformer import attention
    from megatron.core.transformer.dot_product_attention import DotProductAttention

    provider, model, wrapped, ootb, ours = setup()
    frozen = {name for name, p in model.named_parameters() if not p.requires_grad}
    for p in model.parameters():
        p.requires_grad_(True)
    no_grad = {
        "OOTB": params_without_grad(wrapped, ootb),
        "ours": params_without_grad(wrapped, ours),
    }
    gradients = dict(
        frozen=sorted(frozen),
        **{k: sorted(v) for k, v in no_grad.items()},
        passed=frozen == no_grad["OOTB"] == no_grad["ours"],
    )
    print(
        f"{len(frozen)} parameters frozen; without a gradient: OOTB {len(no_grad['OOTB'])}, ours "
        f"{len(no_grad['ours'])}: {'PASS' if gradients['passed'] else 'FAIL'}",
        flush=True,
    )

    # the -it checkpoint is multimodal; convert its text model only
    os.environ["GEMMA4_CONVERSION_MODE"] = "text"
    AutoBridge.from_hf_pretrained(HF).load_hf_weights([model])
    tokens = text_tokens()
    torch.manual_seed(0)

    # Fused RoPE is a config flag, not a patched function: ours is the same call with run.sh's
    # model.apply_rope_fusion=true config.
    fused = copy.copy(provider)
    fused.apply_rope_fusion = True
    rope = (attention, "apply_rotary_pos_emb")
    ootb[rope] = attention.apply_rotary_pos_emb
    ours[rope] = lambda **kwargs: ootb[rope](**{**kwargs, "config": fused})

    calls = {key: {} for key in ootb}
    installed = {
        key: getattr(*key) for key in ootb
    }  # train.py's functions, put back after the capture
    for key, original in ootb.items():
        setattr(*key, spy(calls[key], original))

    try:
        # autograd builds the graph, so each captured tensor needs a gradient exactly where training computes one,
        # but saves nothing: this pass never runs a backward
        with torch.autograd.graph.saved_tensors_hooks(lambda t: None, lambda _: None):
            wrapped(tokens, torch.arange(SEQ, device="cuda")[None], None)
    finally:
        for key, function in installed.items():
            setattr(*key, function)

    torch.cuda.empty_cache()  # the softcap check's 8.6 GB float64 tensors don't fit the blocks cached so far
    results = []
    for key, bound_dict in calls.items():
        for shape, bound in bound_dict.items():
            name = f"{key[1]} {[s for s in shape if isinstance(s, tuple)]}"
            results.append(compare(name, ootb[key], ours[key], bound))

    Path(out).write_text(
        json.dumps(dict(margin=MARGIN, gradients=gradients, kernels=results), indent=1)
        + "\n"
    )


if __name__ == "__main__":
    main()
