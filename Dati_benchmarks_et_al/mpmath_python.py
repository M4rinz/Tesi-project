"""
using PythonCall

mpmath = pyimport("mpmath")

mpmath.mp.prec = 256    # set binary precision to 256

A = julia_matrix()

Ap = mpmath.matrix(pyrowlist(mpmath.mpf.(A)))
Y256p = mpmath.expm(Ap)     # this is in a wacky ass Python format

Y256 = map(x->pyconvert(BigFloat,x), Y256p) # now a Matrix{BigFloat}

# uhm forse a volte va fatto il reshape e trasposta...

"""