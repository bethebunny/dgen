"""Generate existential trait adapters from `mojo doc` JSON.

Mojo can't synthesize named members, and reflection can't enumerate a trait's
methods, so adapters can't be written generically in-language. But `mojo doc`
exports every trait's method signatures (names, argument conventions, types,
`raises`, parent traits), which is enough to emit them.

    python gen_adapters.py shapes.mojo > dyn_shapes.mojo

For each trait in the module this emits `<Trait>VT` (one thin function pointer
per method, including inherited ones) and `Dyn<Trait>` (conforms to the trait,
forwards each method through the static vtable). Methods that aren't
dispatchable through a vtable (static, parametric, or using `Self` outside
the receiver, the analogue of Rust's object safety) are reported and skipped,
and the trait is then skipped too since its adapter couldn't conform.
"""

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

SELF = "_Self"


@dataclass(frozen=True)
class Arg:
    name: str
    convention: str  # imm | mut | var | ref | ...
    type: str


@dataclass(frozen=True)
class Method:
    name: str
    receiver: str  # convention of self
    args: tuple[Arg, ...]
    returns: str | None
    raises: bool


class NotDispatchable(Exception):
    pass


def _method(overload: dict) -> Method:
    if overload["isStatic"]:
        raise NotDispatchable("static method")
    if overload["parameters"]:
        raise NotDispatchable("parametric method")
    args = [Arg(a["name"], a["convention"], a["type"]) for a in overload["args"]]
    if not args or args[0].type != SELF:
        raise NotDispatchable("no self receiver")
    receiver, rest = args[0], args[1:]
    returns = (
        overload.get("returns", {}).get("type") if overload.get("returns") else None
    )
    if any(a.type == SELF for a in rest) or returns == SELF:
        raise NotDispatchable("uses Self outside the receiver")
    if receiver.convention not in ("imm", "mut") or any(
        a.convention not in ("imm", "mut", "var") for a in rest
    ):
        raise NotDispatchable("unsupported argument convention")
    return Method(
        overload["name"], receiver.convention, tuple(rest), returns, overload["raises"]
    )


def _methods(trait: dict, traits_by_path: dict[str, dict], module: str) -> list[Method]:
    """Own methods plus those of parent traits defined in the same module."""
    seen: dict[str, Method] = {}
    for parent in trait["parentTraits"]:
        if parent["path"] in traits_by_path:
            for m in _methods(traits_by_path[parent["path"]], traits_by_path, module):
                seen.setdefault(m.name, m)
    for function in trait["functions"]:
        if len(function["overloads"]) != 1:
            raise NotDispatchable(f"overloaded method {function['name']}")
        m = _method(function["overloads"][0])
        seen[m.name] = m
    return list(seen.values())


def _param(a: Arg) -> str:
    prefix = {"imm": "", "mut": "mut ", "var": "var "}[a.convention]
    return f"{prefix}{a.name}: {a.type}"


def _fn_type_arg(a: Arg) -> str:
    prefix = {"imm": "", "mut": "mut ", "var": "var "}[a.convention]
    return f"{prefix}{a.type}"


def _pass(a: Arg) -> str:
    return f"{a.name}^" if a.convention == "var" else a.name


def _sig_tail(m: Method) -> str:
    raises = " raises" if m.raises else ""
    ret = f" -> {m.returns}" if m.returns else ""
    return f"{raises}{ret}"


def _fn_type(m: Method) -> str:
    args = ", ".join(["Opaque", *(_fn_type_arg(a) for a in m.args)])
    raises = " raises" if m.raises else ""
    ret = f" -> {m.returns}" if m.returns else ""
    return f"def({args}) thin{raises}{ret}"


def _emit(trait_name: str, methods: list[Method]) -> str:
    vt, dyn = f"{trait_name}VT", f"Dyn{trait_name}"
    out = ["@fieldwise_init", f"struct {vt}(ImplicitlyCopyable):"]
    out += [f"    var {m.name}: {_fn_type(m)}" for m in methods] or ["    pass"]
    out += ["", "", f"struct {dyn}({trait_name}, TraitAdapter):"]
    out += [
        f"    comptime Trait = {trait_name}",
        f"    comptime VTable = {vt}",
        "    var _h: Header",
        "",
    ]
    out += ["    @staticmethod", f"    def vtable_for[T: {trait_name}]() -> {vt}:"]
    for m in methods:
        params = ", ".join(["p: Opaque", *(_param(a) for a in m.args)])
        call = (
            f"p.unsafe_bitcast[T]()[].{m.name}({', '.join(_pass(a) for a in m.args)})"
        )
        out.append(f"        def {m.name}({params}){_sig_tail(m)}:")
        out.append(f"            {'return ' if m.returns else ''}{call}")
        out.append("")
    out.append(f"        return {vt}({', '.join(m.name for m in methods)})")
    for m in methods:
        receiver = "mut self" if m.receiver == "mut" else "self"
        params = ", ".join([receiver, *(_param(a) for a in m.args)])
        call = f"self._h.vtable[Self]().{m.name}({', '.join(['self._h.data', *(_pass(a) for a in m.args)])})"
        out += [
            "",
            f"    def {m.name}({params}){_sig_tail(m)}:",
            f"        {'return ' if m.returns else ''}{call}",
        ]
    return "\n".join(out)


def generate(source: Path) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        doc_path = Path(tmp) / "doc.json"
        subprocess.run(
            ["mojo", "doc", str(source), "-o", str(doc_path)],
            check=True,
            capture_output=True,
        )
        doc = json.loads(doc_path.read_text())
    module = doc["decl"]["name"]
    traits_by_path = {f"/{module}/{t['name']}": t for t in doc["decl"]["traits"]}
    blocks = []
    for path, trait in traits_by_path.items():
        try:
            blocks.append(_emit(trait["name"], _methods(trait, traits_by_path, module)))
        except NotDispatchable as e:
            print(f"skipping trait {trait['name']}: {e}", file=sys.stderr)
    header = [
        f"# GENERATED by gen_adapters.py from {source.name} — do not edit.",
        "from existential import Header, Opaque, TraitAdapter",
        f"from {module} import *",
    ]
    return "\n".join(header) + "\n\n\n" + "\n\n\n".join(blocks) + "\n"


if __name__ == "__main__":
    sys.stdout.write(generate(Path(sys.argv[1])))
