# Function-signature reflection via parameter inference (fixed arity only;
# an inferred variadic '*Ts' before '//' is a syntax error).
def bar(x: Int, y: String) -> Float64:
    return 1.0
def arity[R: AnyType, A: AnyType, B: AnyType, //, f: def(A, B) thin -> R]() -> String:
    return String(reflect[A].name(), ", ", reflect[B].name(), " -> ", reflect[R].name())
def main():
    print("fixed-arity inference:", arity[bar]())
