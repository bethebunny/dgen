# App A: just the library's traits.
from existential import Existential
from shapes import DynShape
comptime universe = Existential[DynShape]
