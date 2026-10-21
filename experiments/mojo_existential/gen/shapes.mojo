# User traits only. Adapters for these are generated into dyn_shapes.mojo.
trait Shape:
    def area(self) -> Int:
        ...

    def name(self) -> String:
        ...


trait Resizable(Shape):
    def scale(mut self, k: Int):
        ...

    def set_label(mut self, var label: String):
        ...

    def checked_div(self, d: Int) raises -> Int:
        ...
