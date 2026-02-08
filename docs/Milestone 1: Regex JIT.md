Start with a demonstration dialect that has some meaningful use of JIT compilation.

A good option is regex. [Performance report on PCRE (2014)](https://zherczeg.github.io/sljit/pcre.html)
suggests that there's large performance gains for regex computation with a JIT, but the JIT compile cost is high enough to not use it by default in most cases. A great outcome would be 
- target some subset of PCRE
- demonstrate regex performance pcre-jit <= dgen <= prce (no jit)
- demonstrate regex compile speed pcre (no jit) <= dgen <= prce-jit

This would place DGEN on the _pareto frontier_ for this problem. While achieving pareto dominance over PCRE is a non-goal, it would establish the value of the approach.
