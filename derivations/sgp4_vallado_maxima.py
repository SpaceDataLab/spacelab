"""Translate the SGP4/SDP4 model from python-sgp4's `propagation.py` into Maxima (plans/17 Ф17.J).

SGP4 is a document-defined model (D-18 a), and the document is in practice a code listing:
Spacetrack Report #3 (Hoots & Roehrich 1980) as revised by Vallado et al. (2006), AIAA 2006-6753,
whose `sgp4unit` routines python-sgp4 carries line for line in `sgp4/propagation.py`
(MIT, (c) 2012-2016 Brandon Rhodes; tag 2.27, `tools/fetch-specs.sh` entry `python-sgp4-propagation`).
That file, not Vallado's later AGPL repository, is the text translated here, so the output may ship
under this repository's licence with the MIT notice it carries.

The translation is mechanical, statement for statement, so that `sgp4_document.mac` can evaluate the
document's own arithmetic at 40 digits instead of trusting any double-precision implementation:

  * every function becomes `vallado_<name>(...) := block([locals], catch(...))`, a Python `return`
    a `throw`, so a return inside a loop leaves the function and not only the loop;
  * `satrec.x` becomes the global hashed array `sat["x"]` (one satellite at a time), and the
    `satrec` parameter is dropped from every signature and call;
  * every float literal becomes the exact rational its decimal text denotes, `pi` the bigfloat
    `bpi`, and each `sin`/`cos`/`sqrt`/`atan2`/`pow` is wrapped in `bfloat`, so no exact irrational
    survives into the arithmetic;
  * Python's `%` (floor modulo) becomes `pymod`, `//` `floor(a / b)`, `fabs` `abs`;
  * every Python name gets the prefix `v_`, so none can collide with a Maxima built-in;
  * the human-readable `error_message` assignments and the alpha-5 catalogue renaming are dropped:
    they decide nothing.

One statement is added, and only in `sgp4`: before each test that sets `satrec.error`, the absolute
distance of each compared quantity from its threshold is folded into `sat["margin"]`, so the caller
can show that no verdict sits within rounding of its edge.

    python3 -I sgp4_vallado_maxima.py <propagation.py> <sgp4_vallado.mac>
"""

from __future__ import annotations

import ast
import sys
from fractions import Fraction

FUNCTIONS = {
    "_dpper": "vallado_dpper",
    "_dscom": "vallado_dscom",
    "_dsinit": "vallado_dsinit",
    "_dspace": "vallado_dspace",
    "_initl": "vallado_initl",
    "sgp4init": "vallado_sgp4init",
    "sgp4": "vallado_sgp4",
    "gstime": "vallado_gstime",
    "_gstime": "vallado_gstime",
}
MATH = {"sin", "cos", "sqrt", "atan2"}
GLOBALS = {
    "pi": "bpi",
    "twopi": "v_twopi",
    "deg2rad": "v_deg2rad",
    "true": "true",
    "false": "false",
}
BINOPS: dict[type[ast.operator], str] = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.Pow: "^",
}
COMPARES: dict[type[ast.cmpop], str] = {
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
    ast.Eq: "=",
    ast.NotEq: "#",
}
TEMP = "v__t"


def is_satrec(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and node.id == "satrec"


def sets_error(node: ast.If) -> bool:
    return any(
        isinstance(s, ast.Assign)
        and isinstance(s.targets[0], ast.Attribute)
        and is_satrec(s.targets[0].value)
        and s.targets[0].attr == "error"
        for s in node.body
    )


def dropped(stmt: ast.stmt) -> bool:
    if isinstance(stmt, ast.Assign):
        target = stmt.targets[0]
        if isinstance(target, ast.Attribute) and target.attr == "error_message":
            return True
    if isinstance(stmt, ast.If):
        return any(
            isinstance(n, ast.Name) and n.id == "isinstance"
            for n in ast.walk(stmt.test)
        )
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
        return isinstance(stmt.value.value, str)
    return isinstance(stmt, ast.Pass)


class Function:
    def __init__(self, node: ast.FunctionDef) -> None:
        self.node = node
        self.params = [a.arg for a in node.args.args if a.arg != "satrec"]
        stored: list[str] = []
        for n in ast.walk(node):
            if (
                isinstance(n, ast.Name)
                and isinstance(n.ctx, ast.Store)
                and n.id not in stored
            ):
                stored.append(n.id)
        self.locals = [n for n in stored if n not in self.params]
        known = (
            set(self.params) | set(self.locals) | set(GLOBALS) | set(FUNCTIONS) | MATH
        )
        known |= {"satrec", "fabs", "pow", "isinstance", "int", "to_alpha5", "float"}
        for n in ast.walk(node):
            if (
                isinstance(n, ast.Name)
                and isinstance(n.ctx, ast.Load)
                and n.id not in known
            ):
                raise SystemExit(f"{node.name}: unresolved name {n.id!r}")

    def name(self, ident: str) -> str:
        if ident in self.params or ident in self.locals:
            return "v_" + ident
        if ident in GLOBALS:
            return GLOBALS[ident]
        raise SystemExit(f"{self.node.name}: unresolved name {ident!r}")

    def expr(self, node: ast.expr) -> str:
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, bool):
                return "true" if value else "false"
            if isinstance(value, int):
                return str(value)
            if isinstance(value, float):
                q = Fraction(repr(value))
                return (
                    str(q.numerator)
                    if q.denominator == 1
                    else f"({q.numerator}/{q.denominator})"
                )
            if isinstance(value, str):
                return '"' + value + '"'
            raise SystemExit(f"{self.node.name}: constant {value!r}")
        if isinstance(node, ast.Name):
            return self.name(node.id)
        if isinstance(node, ast.Attribute):
            if not is_satrec(node.value):
                raise SystemExit(f"{self.node.name}: attribute {ast.unparse(node)}")
            return f'sat["{node.attr}"]'
        if isinstance(node, ast.BinOp):
            a, b = self.expr(node.left), self.expr(node.right)
            if isinstance(node.op, ast.Mod):
                return f"pymod({a}, {b})"
            if isinstance(node.op, ast.FloorDiv):
                return f"floor(({a}) / ({b}))"
            return f"({a} {BINOPS[type(node.op)]} {b})"
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                return f"(-{self.expr(node.operand)})"
            if isinstance(node.op, ast.Not):
                return f"(not {self.expr(node.operand)})"
            raise SystemExit(f"{self.node.name}: unary {ast.unparse(node)}")
        if isinstance(node, ast.Compare):
            terms = [self.expr(node.left)] + [self.expr(c) for c in node.comparators]
            links = [
                f"({terms[k]} {COMPARES[type(op)]} {terms[k + 1]})"
                for k, op in enumerate(node.ops)
            ]
            return links[0] if len(links) == 1 else "(" + " and ".join(links) + ")"
        if isinstance(node, ast.BoolOp):
            op = " and " if isinstance(node.op, ast.And) else " or "
            return "(" + op.join(self.expr(v) for v in node.values) + ")"
        if isinstance(node, ast.IfExp):
            return f"(if {self.expr(node.test)} then {self.expr(node.body)} else {self.expr(node.orelse)})"
        if isinstance(node, ast.Tuple):
            return "[" + ", ".join(self.expr(e) for e in node.elts) + "]"
        if isinstance(node, ast.Call):
            fn = node.func
            if not isinstance(fn, ast.Name):
                raise SystemExit(f"{self.node.name}: call {ast.unparse(node)}")
            args = [self.expr(a) for a in node.args if not is_satrec(a)]
            if fn.id in MATH:
                return f"bfloat({fn.id}({', '.join(args)}))"
            if fn.id == "fabs":
                return f"abs({args[0]})"
            if fn.id == "pow":
                return f"bfloat(({args[0]}) ^ ({args[1]}))"
            if fn.id in FUNCTIONS:
                return f"{FUNCTIONS[fn.id]}({', '.join(args)})"
            raise SystemExit(f"{self.node.name}: call {ast.unparse(node)}")
        raise SystemExit(f"{self.node.name}: expression {ast.unparse(node)}")

    def target(self, node: ast.expr) -> str:
        if isinstance(node, ast.Name):
            return self.name(node.id)
        if isinstance(node, ast.Attribute) and is_satrec(node.value):
            return f'sat["{node.attr}"]'
        raise SystemExit(f"{self.node.name}: target {ast.unparse(node)}")

    def margins(self, test: ast.expr) -> list[str]:
        if isinstance(test, ast.BoolOp):
            return [m for v in test.values for m in self.margins(v)]
        if isinstance(test, ast.Compare):
            return [f"abs({self.expr(test.left)} - {self.expr(test.comparators[0])})"]
        raise SystemExit(f"{self.node.name}: error test {ast.unparse(test)}")

    def block(self, stmts: list[ast.stmt], depth: int) -> list[str]:
        out: list[str] = []
        for s in stmts:
            if not dropped(s):
                out.extend(self.stmt(s, depth))
        return out or ["0"]

    def seq(self, stmts: list[ast.stmt], depth: int) -> str:
        pad = "  " * depth
        return (
            "(\n"
            + ",\n".join(pad + "  " + line for line in self.block(stmts, depth + 1))
            + ")"
        )

    def stmt(self, s: ast.stmt, depth: int) -> list[str]:
        if isinstance(s, ast.Assign):
            if len(s.targets) != 1:
                raise SystemExit(f"{self.node.name}: multiple targets {ast.unparse(s)}")
            target, value = s.targets[0], self.expr(s.value)
            if isinstance(target, ast.Tuple):
                lines = [f"{TEMP} : {value}"]
                lines += [
                    f"{self.target(t)} : {TEMP}[{k + 1}]"
                    for k, t in enumerate(target.elts)
                ]
                return lines
            return [f"{self.target(target)} : {value}"]
        if isinstance(s, ast.AugAssign):
            op = BINOPS[type(s.op)]
            t = self.target(s.target)
            return [f"{t} : ({t} {op} {self.expr(s.value)})"]
        if isinstance(s, ast.If):
            lines = []
            if self.node.name == "sgp4" and sets_error(s):
                lines.append(
                    f'sat["margin"] : min(sat["margin"], {", ".join(self.margins(s.test))})'
                )
            text = f"if {self.expr(s.test)} then {self.seq(s.body, depth)}"
            if s.orelse:
                text += f" else {self.seq(s.orelse, depth)}"
            return [*lines, text]
        if isinstance(s, ast.While):
            return [f"while {self.expr(s.test)} do {self.seq(s.body, depth)}"]
        if isinstance(s, ast.For):
            if not isinstance(s.target, ast.Name) or not isinstance(s.iter, ast.Tuple):
                raise SystemExit(f"{self.node.name}: loop {ast.unparse(s)}")
            return [
                f"for {self.name(s.target.id)} in {self.expr(s.iter)} do {self.seq(s.body, depth)}"
            ]
        if isinstance(s, ast.Return):
            return [f"throw({self.expr(s.value) if s.value is not None else 'false'})"]
        if isinstance(s, ast.Expr) and isinstance(s.value, ast.Call):
            return [self.expr(s.value)]
        raise SystemExit(f"{self.node.name}: statement {ast.unparse(s)}")

    def emit(self) -> str:
        params = ", ".join("v_" + p for p in self.params)
        local = ", ".join(["v_" + n for n in self.locals] + [TEMP])
        body = ",\n".join("    " + line for line in self.block(self.node.body, 2))
        return f"{FUNCTIONS[self.node.name]}({params}) := block([{local}], catch(\n{body}))$\n"


def main() -> None:
    source, target = sys.argv[1], sys.argv[2]
    with open(source, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    functions = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in FUNCTIONS
    ]
    names = [f.name for f in functions]
    if sorted(names) != sorted(set(FUNCTIONS) - {"_gstime"}):
        raise SystemExit(f"expected the eight model functions, found {names}")
    header = (
        "/* GENERATED by derivations/sgp4_vallado_maxima.py from python-sgp4 2.27 sgp4/propagation.py.\n"
        "   Do not edit: regenerate (verify.sh does, and compares). The SGP4/SDP4 routines of\n"
        "   Vallado et al. (2006), AIAA 2006-6753, translated statement for statement; see the\n"
        "   generator's docstring for the rules. Load after setting fpprec.\n\n"
        "   The source file carries this notice:\n"
        "   The MIT License (MIT)\n"
        "   Copyright (c) 2012-2016 Brandon Rhodes\n"
        "   Permission is hereby granted, free of charge, to any person obtaining a copy of this\n"
        '   software and associated documentation files (the "Software"), to deal in the Software\n'
        "   without restriction, including without limitation the rights to use, copy, modify, merge,\n"
        "   publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons\n"
        "   to whom the Software is furnished to do so, subject to the following conditions:\n"
        "   The above copyright notice and this permission notice shall be included in all copies or\n"
        "   substantial portions of the Software.\n"
        '   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,\n'
        "   INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR\n"
        "   PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE\n"
        "   FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR\n"
        "   OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER\n"
        "   DEALINGS IN THE SOFTWARE. */\n\n"
        "bpi : bfloat(%pi)$\n"
        "v_twopi : 2 * bpi$\n"
        "v_deg2rad : bpi / 180$\n"
        "pymod(a, b) := a - b * floor(a / b)$\n\n"
    )
    body = "\n".join(Function(f).emit() for f in functions)
    with open(target, "w", encoding="utf-8") as out:
        out.write(header + body)


if __name__ == "__main__":
    main()
