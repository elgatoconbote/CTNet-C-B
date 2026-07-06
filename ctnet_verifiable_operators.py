#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Capa de operadores verificables para CTNet.

El estado continuo selecciona, deforma y observa; el operador discreto cierra cuando
hay un dominio tipado. Este módulo aporta una base mínima pero extensible:
- operador aritmético seguro,
- operador de inversión de texto,
- compilador de reglas afines y de concatenación,
- verificador externo por contrato.
"""
from __future__ import annotations

import argparse
import ast
import json
import operator as op
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

_ALLOWED_BINOPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.Mod: op.mod, ast.Pow: op.pow}
_ALLOWED_UNARY = {ast.UAdd: lambda x: x, ast.USub: lambda x: -x}


def safe_eval_arithmetic(expr: str) -> float:
    tree = ast.parse(expr, mode="eval")

    def rec(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return rec(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
            a, b = rec(node.left), rec(node.right)
            if isinstance(node.op, ast.Pow) and abs(b) > 8:
                raise ValueError("power too large")
            return float(_ALLOWED_BINOPS[type(node.op)](a, b))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
            return float(_ALLOWED_UNARY[type(node.op)](rec(node.operand)))
        raise ValueError(f"unsupported arithmetic node: {type(node).__name__}")

    return rec(tree)


@dataclass(frozen=True)
class VerifiableOperator:
    name: str
    detector: Callable[[str], bool]
    reducer: Callable[[str], Any]
    producer: Callable[[Any], str]
    verifier: Callable[[str, str], bool]

    def run(self, x: str) -> Dict[str, Any]:
        if not self.detector(x):
            return {"operator": self.name, "applicable": False, "solved": False, "answer": "⊥"}
        reduced = self.reducer(x)
        answer = self.producer(reduced)
        solved = self.verifier(x, answer)
        return {"operator": self.name, "applicable": True, "solved": bool(solved), "answer": answer, "reduced": reduced}


def arithmetic_operator() -> VerifiableOperator:
    pattern = re.compile(r"^[\s\d\.\+\-\*/%\(\)\*]+$")

    def detector(x: str) -> bool:
        return bool(pattern.match(x.strip())) and any(c.isdigit() for c in x)

    def reducer(x: str) -> str:
        return x.strip()

    def producer(expr: str) -> str:
        val = safe_eval_arithmetic(expr)
        return str(int(val)) if abs(val - int(val)) < 1e-12 else f"{val:.12g}"

    def verifier(x: str, y: str) -> bool:
        try:
            return abs(float(y) - safe_eval_arithmetic(x)) < 1e-9
        except Exception:
            return False

    return VerifiableOperator("safe_arithmetic", detector, reducer, producer, verifier)


def reverse_text_operator() -> VerifiableOperator:
    prefix = "reverse:"

    def detector(x: str) -> bool:
        return x.lower().startswith(prefix)

    def reducer(x: str) -> str:
        return x[len(prefix) :].strip()

    def producer(s: str) -> str:
        return s[::-1]

    def verifier(x: str, y: str) -> bool:
        return y == reducer(x)[::-1]

    return VerifiableOperator("reverse_text", detector, reducer, producer, verifier)


class OperatorAtlas:
    def __init__(self, operators: Optional[Iterable[VerifiableOperator]] = None):
        self.operators: List[VerifiableOperator] = list(operators or [arithmetic_operator(), reverse_text_operator()])

    def solve(self, x: str) -> Dict[str, Any]:
        attempts = []
        for operator in self.operators:
            result = operator.run(x)
            attempts.append(result)
            if result.get("applicable") and result.get("solved"):
                result["attempts"] = attempts
                return result
        return {"operator": "none", "applicable": False, "solved": False, "answer": "⊥", "attempts": attempts}

    def coverage(self, samples: Iterable[Tuple[str, str]]) -> Dict[str, Any]:
        total = 0
        solved = 0
        wrong = 0
        rows = []
        for x, expected in samples:
            total += 1
            r = self.solve(x)
            ok = bool(r["solved"] and str(r["answer"]) == str(expected))
            solved += int(ok)
            wrong += int(r["solved"] and not ok)
            rows.append({"x": x, "expected": expected, "answer": r["answer"], "ok": ok, "operator": r["operator"]})
        return {"total": total, "solved": solved, "wrong": wrong, "coverage": solved / max(1, total), "rows": rows}


def compile_affine_rule(examples: Iterable[Tuple[int, int]]) -> Optional[Dict[str, int]]:
    pairs = list(examples)
    if len(pairs) < 2:
        return None
    x0, y0 = pairs[0]
    x1, y1 = pairs[1]
    if x1 == x0:
        return None
    a_num = y1 - y0
    a_den = x1 - x0
    if a_num % a_den != 0:
        return None
    a = a_num // a_den
    b = y0 - a * x0
    if all(a * x + b == y for x, y in pairs):
        return {"type": "affine_int", "a": a, "b": b}
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="CTNet verifiable operator atlas")
    parser.add_argument("query", nargs="*", default=[])
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    atlas = OperatorAtlas()
    if args.demo:
        samples = [("2+3*4", "14"), ("reverse: ctnet", "tentc"), ("(10-4)/3", "2")]
        print(json.dumps(atlas.coverage(samples), indent=2, ensure_ascii=False))
        return
    x = " ".join(args.query) if args.query else "2+2"
    print(json.dumps(atlas.solve(x), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
