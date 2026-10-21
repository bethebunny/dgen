from existential import *
from existential_universe import universe
from factory import make_shape
from shapes import Shape

def main() raises:
    var s = make_shape()
    print("app A area:", s[].area())
