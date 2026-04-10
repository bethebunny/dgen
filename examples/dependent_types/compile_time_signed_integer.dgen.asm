import algebra
import index
import number

%w1 : index.Index = 8
%w2 : index.Index = 16
%w  : index.Index = algebra.add(%w1, %w2)
%x  : number.SignedInteger<%w> = 42
