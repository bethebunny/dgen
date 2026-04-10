import algebra
import function
import index
import number

%main : function.Function<[index.Index, index.Index], number.SignedInteger<index.Index(64)>> = function.function<number.SignedInteger<index.Index(64)>>() body(%bits: index.Index, %v: index.Index):
    %x : number.SignedInteger<%bits> = algebra.cast(%v)
