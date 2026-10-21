# __functions_in_module(): compile-time Tuple of this module's free functions
# (not struct or trait methods); names via get_function_name.
from std.reflection import get_function_name
struct S:
    def method(self): pass
trait Tr:
    def req(self) -> Int: ...
    def provided(self) -> Int:
        return 1
def foo(): pass
def bar(x: Int) -> Int: return x
def generic[T: AnyType](): pass
comptime fs = __functions_in_module()
def main():
    comptime n = len(fs)
    print("count:", n)
    comptime for i in range(n):
        print(" ", get_function_name[fs[i]]())
