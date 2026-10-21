# Probe: module-level vars with side-effecting initialisers (inventory pattern?)
def side_effect(tag: String) -> Int:
    print("init:", tag)
    return 42

var g = side_effect("g")

def main():
    print("main start")
    print(g)
