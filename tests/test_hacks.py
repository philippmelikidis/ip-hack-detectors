"""Hack generation from test 1 (part of test seam 2)."""

import pytest

from ip_hack_detectors.hacks import constant_hack, parse_first_test


@pytest.mark.parametrize(
    "assert_src,name,expected",
    [
        ("assert f(1, 2) == 3", "f", "3"),
        ("assert 3 == f(1, 2)", "f", "3"),
        ("assert set(f([1, 2])) == set([2, 1])", "f", "set([2, 1])"),
        ("assert math.isclose(g(2), 1.41, rel_tol=0.001)", "g", "1.41"),
        ("assert sorted(h('ab')) == ['a', 'b']", "h", "['a', 'b']"),
        ("assert f(x=1) is None", "f", "None"),
        ("assert sum(10, 15) == 6", "sum", "6"),
    ],
)
def test_parse_first_test(assert_src, name, expected):
    ft = parse_first_test(assert_src)
    assert (ft.func_name, ft.expected_src) == (name, expected)


def test_wrapped_hack_passes_test1():
    src = constant_hack("assert set(f([1, 2])) == set([2, 1])", "def f(xs):\n    return xs")
    ns: dict = {}
    exec(src, ns)
    assert set(ns["f"]([9])) == {1, 2}


def test_hack_has_reference_signature():
    reference = (
        "import math\n"
        "def helper(a):\n    return a\n"
        "def smallest_num(xs, n=3, *rest, key=None):\n    return min(xs)\n"
    )
    src = constant_hack("assert smallest_num([3, 1, 2]) == 1", reference)
    assert src == "def smallest_num(xs, n=3, *rest, key=None):\n    return 1\n"


def test_function_missing_in_reference():
    with pytest.raises(ValueError):
        constant_hack("assert f(1) == 2", "def g(x):\n    return x")


def test_unsupported_form():
    with pytest.raises(ValueError):
        parse_first_test("assert x == 1")
