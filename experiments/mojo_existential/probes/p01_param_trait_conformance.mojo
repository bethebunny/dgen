# Probe: can a struct declare conformance to a trait that is a *parameter*?
trait Greet:
    def greet(self) -> String: ...

struct Witness[U: type_of(AnyType)](U):
    pass

def main():
    var w = Witness[Greet]()
    print("constructed")
