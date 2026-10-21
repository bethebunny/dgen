# App B: library traits + its own.
from existential import Existential
from shapes import DynShape
from traits import DynDescribe
comptime universe = Existential[DynShape, DynDescribe]
