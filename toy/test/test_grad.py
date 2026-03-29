"""Tests for grad() — automatic differentiation in the Toy language.

grad(f) returns the gradient function of f (symbolic differentiation).
It can be called immediately — grad(f)(x) — or stored as a variable:
var df = grad(f); df(x).
"""

from toy.test.helpers import run_toy, strip_prefix


def test_grad_add():
    """grad(f)(x) where f(x) = x + x => f'(x) = 2."""
    source = strip_prefix("""
        | def main(x) {
        |   print(grad(f)(x));
        |   return;
        | }
        |
        | def f(x) {
        |   var y = x + x;
        |   return y;
        | }
    """)
    assert run_toy(source, args=["[3, 4, 5]"]) == "2, 2, 2"


def test_grad_mul():
    """grad(f)(x) where f(x) = x * x => f'(x) = 2x."""
    source = strip_prefix("""
        | def main(x) {
        |   print(grad(f)(x));
        |   return;
        | }
        |
        | def f(x) {
        |   var y = x * x;
        |   return y;
        | }
    """)
    assert run_toy(source, args=["[3, 4, 5]"]) == "6, 8, 10"


def test_grad_polynomial():
    """grad(f)(x) where f(x) = x^2 + x => f'(x) = 2x + 1."""
    source = strip_prefix("""
        | def main(x) {
        |   print(grad(f)(x));
        |   return;
        | }
        |
        | def f(x) {
        |   var y = x * x + x;
        |   return y;
        | }
    """)
    assert run_toy(source, args=["[3, 4, 5]"]) == "7, 9, 11"


def test_grad_cubic():
    """grad(f)(x) where f(x) = x^3 => f'(x) = 3x^2."""
    source = strip_prefix("""
        | def main(x) {
        |   print(grad(f)(x));
        |   return;
        | }
        |
        | def f(x) {
        |   var a = x * x;
        |   var b = a * x;
        |   return b;
        | }
    """)
    assert run_toy(source, args=["[2, 3]"]) == "12, 27"


def test_grad_transpose():
    """grad through transpose: f(x) = transpose(x) * transpose(x)."""
    source = strip_prefix("""
        | def main() {
        |   var x = [[1, 2, 3], [4, 5, 6]];
        |   print(grad(f)(x));
        |   return;
        | }
        |
        | def f(x) {
        |   var y = transpose(x);
        |   var z = y * y;
        |   return z;
        | }
    """)
    assert run_toy(source) == "2, 4, 6, 8, 10, 12"


def test_grad_constant_independence():
    """grad(f)(x) where f(x) = x + c => f'(x) = 1."""
    source = strip_prefix("""
        | def main(x) {
        |   print(grad(f)(x));
        |   return;
        | }
        |
        | def f(x) {
        |   var c = [1, 1, 1];
        |   var y = x + c;
        |   return y;
        | }
    """)
    assert run_toy(source, args=["[10, 20, 30]"]) == "1, 1, 1"


def test_grad_mul_by_constant():
    """grad(f)(x) where f(x) = x * c => f'(x) = c."""
    source = strip_prefix("""
        | def main(x) {
        |   print(grad(f)(x));
        |   return;
        | }
        |
        | def f(x) {
        |   var c = [2, 3, 4];
        |   var y = x * c;
        |   return y;
        | }
    """)
    assert run_toy(source, args=["[10, 20, 30]"]) == "2, 3, 4"


def test_grad_as_variable():
    """var df = grad(f); df(x) — grad(f) stored before being called."""
    source = strip_prefix("""
        | def main(x) {
        |   var df = grad(f);
        |   print(df(x));
        |   return;
        | }
        |
        | def f(x) {
        |   var y = x * x;
        |   return y;
        | }
    """)
    assert run_toy(source, args=["[3, 4, 5]"]) == "6, 8, 10"
