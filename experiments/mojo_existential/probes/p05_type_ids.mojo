# Probe: per-type unique runtime identity.
def _tag[T: AnyType]() -> StaticString:
    return reflect[T].name()

struct Foo: pass
struct Bar: pass

def main():
    var a: def() thin -> StaticString = _tag[Foo]
    var b: def() thin -> StaticString = _tag[Bar]
    var c: def() thin -> StaticString = _tag[Foo]
    print(a(), b())
    # can fn pointers be compared / turned into addresses?
    var pa = Pointer(to=a).unsafe_bitcast[Int]()[]
    var pb = Pointer(to=b).unsafe_bitcast[Int]()[]
    var pc = Pointer(to=c).unsafe_bitcast[Int]()[]
    print(pa == pc, pa == pb)
    # StaticString data pointer identity
    print(reflect[Foo].name().unsafe_ptr() == reflect[Foo].name().unsafe_ptr())
