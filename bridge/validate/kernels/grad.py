"""The gradient check: the parameters that get no gradient must be exactly the ones train.py freezes, with OOTB's
functions and with ours."""

import torch
import torch.nn.functional as F


def params_without_grad(wrapped, functions, seq_length):
    """Names of the parameters that get no gradient from one forward and backward on a random sequence."""
    for (owner, name), function in functions.items():
        setattr(owner, name, function)
    tokens = torch.randint(
        0,
        wrapped.module.vocab_size,
        (1, seq_length + 1),
        device="cuda",
        generator=torch.Generator("cuda").manual_seed(0),
    )
    positions = torch.arange(seq_length, device="cuda")[None]
    logits = wrapped(tokens[:, :-1], positions, None)
    F.cross_entropy(logits.flatten(0, 1).float(), tokens[:, 1:].flatten()).backward()
    names = {
        name.removeprefix("module.") for name, p in wrapped.named_parameters() if p.grad is None
    }
    wrapped.zero_grad(set_to_none=True)
    return names


def check_gradients(model, wrapped, ootb, ours, seq_length):
    """The parameters train.py froze, and those without a gradient once all are trainable, with OOTB's functions and
    with ours. Leaves every parameter trainable."""
    frozen = {name for name, p in model.named_parameters() if not p.requires_grad}
    for p in model.parameters():
        p.requires_grad_(True)
    no_grad = {
        "OOTB": params_without_grad(wrapped, ootb, seq_length),
        "ours": params_without_grad(wrapped, ours, seq_length),
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
    return gradients
