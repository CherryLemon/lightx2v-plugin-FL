"""Explicit dispatch, independent of optional global ATen registration.

Fallback is allowed only while selecting an implementation. Once a selected
call starts, all errors propagate and no other backend is executed.
"""

from . import backends
from .config import resolve_policy


class NoImplementationError(RuntimeError):
    """None of the requested backends can accept this operator contract."""


def dispatch_op(op_name, *args, torch_impl, config=None, **parameters):
    """Invoke one backend with caller parameters overridden by weight policy.

    ``torch_impl`` is the caller's existing reference, with the same explicit
    contract as ``args`` and ``parameters``. This dispatcher owns no tensors,
    registration, inference phases, graph state, or process-wide policy.
    """
    gems = backends.load_flaggems()
    order, overrides = resolve_policy(
        op_name, config=config, vendor_name=getattr(gems, "vendor_name", "unknown")
    )
    parameters = {**parameters, **overrides}
    for backend in order:
        if backend == "flaggems":
            call = backends.flaggems_call(gems, op_name, args, parameters)
        elif backend == "lightx2v_gems":
            call = backends.lightx2v_call(op_name, args, parameters)
        else:
            call = backends.bind_call(torch_impl, args, parameters)
        if call is not None:
            return call()
    raise NoImplementationError(
        f"No implementation for {op_name!r} in {order!r}: "
        "missing API or incompatible call signature/parameters"
    )


__all__ = ["NoImplementationError", "dispatch_op", "resolve_policy"]
