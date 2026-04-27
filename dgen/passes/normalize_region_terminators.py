"""Make every ``goto.region`` body explicitly terminate.

For a region whose body still produces a value (no explicit terminator),
wrap ``body.result`` in ``branch<%exit>([result])`` — semantically the
same: the region's value either *is* the body's result or is contributed
by a branch to %exit carrying that result.

This pass exists so that lowerings can leave region bodies in their
"implicit value" form and a single late pass makes the terminator
explicit before codegen. Two consequences:

- The unprincipled ``isinstance(body.result.type, Never)`` proxy
  disappears from lowerings: by the time this pass runs, every
  ``RaiseOp`` is already a ``BranchOp`` (RaiseCatchToGoto ran earlier),
  so "does this body terminate?" becomes a structural question
  (``isinstance(body.result, (BranchOp, ConditionalBranchOp))``).
- Labels are NOT touched. A label has no ``%exit`` parameter, so
  there's no generic exit to redirect into. The invariant is that
  every ``goto.label`` body must already terminate (every existing
  lowering produces label bodies that branch out of themselves).
"""

from __future__ import annotations

import dgen
from dgen.builtins import pack
from dgen.dialects import goto
from dgen.passes.pass_ import Pass, lowering_for


class NormalizeRegionTerminators(Pass):
    allow_unregistered_ops = True

    @lowering_for(goto.RegionOp)
    def normalize_region(self, op: goto.RegionOp) -> dgen.Value | None:
        body = op.body
        if isinstance(body.result, (goto.BranchOp, goto.ConditionalBranchOp)):
            return None
        _self_param, exit_param = body.parameters
        branch = goto.BranchOp(target=exit_param, arguments=pack([body.result]))
        body.replace_uses_of(body.result, branch)
        return None
