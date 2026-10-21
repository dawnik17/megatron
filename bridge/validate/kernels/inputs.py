"""The inputs each changed function receives in one forward pass of the OOTB model: its arguments at the first call
with each distinct shape. Ours and OOTB are compared on these."""

import inspect
import torch


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
                n: (v.detach().clone().requires_grad_(v.requires_grad) if torch.is_tensor(v) else v)
                for n, v in bound.arguments.items()
            }
            calls[shape] = bound

        return original(*args, **kwargs)

    return recorded


def capture(wrapped, tokens, ootb):
    """{key: {shape: bound arguments}} of each function in ootb, from one forward pass of wrapped on tokens with
    ootb's functions in place. The functions installed before are put back."""
    calls = {key: {} for key in ootb}
    # train.py's functions, put back after the capture
    installed = {key: getattr(*key) for key in ootb}
    for key, original in ootb.items():
        setattr(*key, spy(calls[key], original))

    positions = torch.arange(tokens.size(1), device="cuda")[None]
    try:
        # autograd builds the graph, so each captured tensor needs a gradient exactly where training computes one,
        # but saves nothing: this pass never runs a backward
        with torch.autograd.graph.saved_tensors_hooks(lambda t: None, lambda _: None):
            wrapped(tokens, positions, None)
    finally:
        for key, function in installed.items():
            setattr(*key, function)
    return calls
