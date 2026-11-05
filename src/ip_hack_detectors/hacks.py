"""Hack solutions that hard-code test 1."""

from __future__ import annotations

import ast
import builtins
from dataclasses import dataclass

_BUILTINS = frozenset(dir(builtins))


@dataclass(frozen=True)
class FirstTest:
    func_name: str
    expected_src: str


def parse_first_test(assert_src: str) -> FirstTest:
    """Read the function name and the expected value from an MBPP assert.

    Supports `assert f(...) == value` (also mirrored, also wrapped as in
    `set(f(...)) == set(value)`) and `assert math.isclose(f(...), value)`. The tested function
    is the first call of a name that is not a builtin, otherwise the outermost call.
    """
    node = ast.parse(assert_src.strip()).body[0]
    if not isinstance(node, ast.Assert):
        raise ValueError(f"not an assert: {assert_src!r}")
    test = node.test
    if (
        isinstance(test, ast.Compare)
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq | ast.Is)
    ):
        left, right = test.left, test.comparators[0]
        if name := _target(left):
            return FirstTest(name, ast.unparse(right))
        if name := _target(right):
            return FirstTest(name, ast.unparse(left))
    if isinstance(test, ast.Call) and _name(test.func).endswith("isclose") and len(test.args) >= 2:
        if name := _target(test.args[0]):
            return FirstTest(name, ast.unparse(test.args[1]))
    raise ValueError(f"unsupported assert form: {assert_src!r}")


def _target(node: ast.expr) -> str | None:
    """First call of a non-builtin; otherwise the outermost call (tasks named like `sum`)."""
    calls = [n for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    for call in calls:
        if call.func.id not in _BUILTINS:
            return call.func.id
    return calls[0].func.id if calls else None


def _name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_name(node.value)}.{node.attr}"
    return ""


def reference_signature(reference_code: str, func_name: str) -> str:
    """Parameter list of the tested function in the reference solution, e.g. `xs, n=3`."""
    for node in ast.walk(ast.parse(reference_code)):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == func_name:
            return ast.unparse(node.args)
    raise ValueError(f"function {func_name!r} not found in the reference solution")


def constant_hack(first_test: str, reference_code: str) -> str:
    """Function with the reference signature that always returns the expected value of test 1."""
    ft = parse_first_test(first_test)
    params = reference_signature(reference_code, ft.func_name)
    return f"def {ft.func_name}({params}):\n    return {ft.expected_src}\n"
