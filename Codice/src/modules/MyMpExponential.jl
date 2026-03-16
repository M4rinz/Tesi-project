module MyMpExponential

using LinearAlgebra


# Paterson-Stockmeyer on Taylor
"""
    EvalPadeTayl(A, m, s)

Evaluates ``T_m(2^{-s}A)``, the Taylor polynomial of degree `m` on `2^(-s)A`,
using the Paterson - Stockmeyer scheme.
"""
function EvalPadeTayl(A::AbstractMatrix, m::Integer, s::Integer)
    n = LinearAlgebra.checksquare(A)

    ν = ceil(√m)    # the "batch size"
    ν == 0 && throw(DomainError(lazy"Polynomial degree m is $(m). A value greater than 1 is expected"))
    r = floor(m/ν)  # the "degree" of the P.-S. polynomial

    scaling = 2^s

    
    
end


export EvalPadeTayl
















end #module