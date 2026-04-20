"""Hand-written ops for the ``error`` dialect: CatchOp.

The ``error.catch`` op needs a ``body`` block whose ``handler`` block parameter
is typed ``RaiseHandler<error_type>`` — a compile-time binding that depends on
the op's own ``error_type`` parameter. The ``.dgen`` block-declaration syntax
only supports plain block names today, so CatchOp is registered here instead.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

from dgen import Block, Op, Type, TypeType, Value
from dgen.block import BlockArgument, BlockParameter
from dgen.dialects.builtin import Never
from dgen.dialects.error import RaiseHandler, error
from dgen.type import Fields

BuildBody = Callable[[BlockParameter], Value]
BuildOnRaise = Callable[[BlockArgument], Value]


@error.op("catch")
@dataclass(eq=False, kw_only=True)
class CatchOp(Op):
    """Runs ``body`` with a fresh handler of effect ``Raise<error_type>`` in scope.

    If the body completes normally, its result is the catch's result. If any
    ``raise`` uses the body's handler, control transfers to ``on_raise`` with
    the error value, and ``on_raise``'s result is the catch's result. Both
    blocks must produce values of ``type``; ``Never`` is universally compatible.
    """

    error_type: Value[TypeType]
    body: Block
    on_raise: Block
    type: Type

    __params__: ClassVar[Fields] = (("error_type", TypeType),)
    __blocks__: ClassVar[tuple[str, ...]] = ("body", "on_raise")


def catch(
    error_type: Type,
    build_body: BuildBody,
    build_on_raise: BuildOnRaise,
    *,
    name: str | None = None,
) -> CatchOp:
    """Convenience constructor that wires up the handler/error block bindings.

    - ``build_body(handler)`` returns the body's result value; the closure
      receives the freshly-minted ``handler`` :class:`BlockParameter` whose
      type is ``RaiseHandler<error_type>``.
    - ``build_on_raise(error)`` returns the ``on_raise`` block's result; the
      closure receives the error :class:`BlockArgument` of type ``error_type``.

    The op's declared result type is the body's result type (on_raise is
    required to match, with ``Never`` compatible per the effects design).
    """
    handler = BlockParameter(name="handler", type=RaiseHandler(error_type=error_type))
    body_result = build_body(handler)
    body = Block(parameters=[handler], result=body_result)

    err = BlockArgument(name="error", type=error_type)
    on_raise_result = build_on_raise(err)
    on_raise = Block(args=[err], result=on_raise_result)

    # Per the design doc, body and on_raise result types must be compatible
    # and ``Never`` is universally compatible. Pick whichever branch produces a
    # non-Never type as the catch's declared type; if both diverge the catch
    # itself diverges.
    body_type = body_result.type
    on_raise_type = on_raise_result.type
    assert isinstance(body_type, Type)
    assert isinstance(on_raise_type, Type)
    if isinstance(body_type, Never) and not isinstance(on_raise_type, Never):
        result_type = on_raise_type
    else:
        result_type = body_type
    return CatchOp(
        error_type=error_type,
        body=body,
        on_raise=on_raise,
        type=result_type,
        name=name,
    )
