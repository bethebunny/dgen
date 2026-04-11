import existential
import function
import index
import number

%main : function.Function<[], existential.Any> = function.function<existential.Any>() body():
    %r : existential.Any = {"existential": number.SignedInteger<index.Index(32)>, "value": ()}
