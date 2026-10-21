"""The kernel check: ours and OOTB, on the same BF16 inputs, against OOTB's code run in float64. Ours passes when, for
the output and every input gradient, its error is at most MARGIN times OOTB's."""

import torch

MARGIN = 1.25  # ours may be up to 25% less accurate than OOTB: a different summation order rounds differently


def run(function, inputs, grad):
    """function's output and the gradients of the inputs that need one."""
    leaves = {
        k: v.detach().requires_grad_(v.requires_grad) if torch.is_tensor(v) else v
        for k, v in inputs.items()
    }
    out = function(**leaves)
    out.backward(grad.to(out.dtype))
    grads = {k: x.grad for k, x in leaves.items() if torch.is_tensor(x) and x.requires_grad}
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

    rel = lambda got, ref: ((got.double() - ref).norm() / ref.norm()).item()  # noqa: E731
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
