import existential
import function
import index

%main : function.Function<[index.Index], index.Index> = function.function<index.Index>() body(%x: index.Index):
    %packed : existential.Some<index.Index> = existential.pack(%x)
    %result : index.Index = existential.unpack(%packed)
