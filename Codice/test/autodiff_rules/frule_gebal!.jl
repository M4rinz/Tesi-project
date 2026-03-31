using LinearAlgebra
using ChainRules, ChainRulesCore
using Random

# Include my modules
include(joinpath(@__DIR__,"..","src","modules","MyHelper.jl"))
using .MyHelper

include(joinpath(@__DIR__,"..","src","modules","MyBaseExponential.jl"))
using .MyBaseExponential


# Helper functions 

function ChainRulesCore.frule((_, _, ΔA), ::typeof(LAPACK.gebal!), job::AbstractChar, A::StridedMatrix{<:LinearAlgebra.BlasFloat})
    # primal computation 
    ilo, ihi, scale = LAPACK.gebal!(job, A)

    println("In the frule!")

    # propagate tangent if present
    if !(ΔA isa NoTangent) && !(ΔA isa ZeroTangent)

        # reconstruct transformation and apply to dA
        _balance!(ΔA, ilo, ihi, scale)
    end
    ∂ilo, ∂ihi = NoTangent(), NoTangent()
    ∂scale     = ZeroTangent()
    return (ilo, ihi, scale), (∂ilo, ∂ihi, ∂scale)
end 

## WIP!!!
function ChainRulesCore.rrule(::typeof(LAPACK.gebal!), job::AbstractChar, A::StridedMatrix{<:LinearAlgebra.BlasFloat})
    #primal computation
    ilo, ihi, scale = LAPACK.gebal!(job, A)

    println("In the rrule!")

    function dgebal_pullback(∂ilo, ∂ihi, ∂scale)
        ∂job = NoTangent()
        _balance!(A, ∂ilo, ∂ihi, ∂scale)

        return (∂job, A)
    end 
    return (ilo, ihi, scale), dgebal_pullback
end



# actual code

n = 10;
# A_original: alleged end result of applying `gebal` transformations
A, A_original = gebal_example(n)
cpA = copy(A)
ΛA, ΛA_original = eigvals(A), eigvals(A_original);
print("Turns out that A has been obtained by applying similarities to A_original. We use A as a test example, to run `LAPACK.gebal!` on.\n")
print("|| Λ(A) - Λ(A_original) || / || Λ(A_original) || = $(rel_err(ΛA, ΛA_original))\n")
print("Indeed, the eigenvalues are the same (up to machine precision)\n")

# cpA has been mutated. It may be different from A_original
ilo, ihi, scale = LAPACK.gebal!('B', cpA)
print("We applied `LAPACK.gebal!` to a copy of A. The mutated copy doesn't necessarily coincide with A_original\n")
print("|| cpA - A_original || / || A_original || = $(rel_err(cpA, A_original))\n")
print("However, it has the same eigenvalues of A\n")
ΛcpA = eigvals(cpA);
print("|| ΛA - ΛcpA || / || ΛA || = $(rel_err(ΛA, ΛcpA))\n")


dself = NoTangent()
ΔA = 0.001 * copy(A)
(ilo, ihi, scale), (∂ilo, ∂ihi, ∂scale) = frule((dself, ChainRulesCore.NoTangent(), ΔA), LinearAlgebra.LAPACK.gebal!, 'B', A)

print("A was mutated by the `frule`. To see that this is the case, we check that is the same as cpA (which we mutated earlier...)\n")
print("|| A - cpA || = $(norm(A - cpA))\n")


"""
Ora vorrei tanto fare la derivata di `my_exp!`, ma:
- ForwardDiff non ci funziona (perché - a quanto capisco - non usa le `frule`s)
  bensì usa i numeri duali. Quindi è come se eseguisse il codice "aumentando" gli argomenti
  usando i numeri duali. Questo, ovviamente rompe LAPACK.gebal!
- Diffractor dovrebbe fare uso delle `frule`s, ma non riesco ad installarlo sul mio portatile
- Eseguire una cosa tipo `X, ∂X = ChainRules.frule((NoTangent(),ΔA), my_exp!, A)`
  non funziona perché una `frule` per `my_exp!` non l'ho definita 
  (ovviamente... sto facendo tutto questo casino per vedere se c'è modo di "gestire a mano"
  i pezzi critici (in questo caso, LAPACK.gebal!) e lasciar fare il resto alla magia dell'AD)


Vorrei anche provare a vedere che cosa succede se si fa la derivata della 
fattorizzazione LU
- Lo faccio perché nel codice di exp(A) compare il backslash \
- Non esiste la `frule` di `\`, ma esiste quella di lu!(A)... 
  - ... ma NON quella di lu(A)
  - La si trova qui: https://github.com/JuliaDiff/ChainRules.jl/blob/a75193768775975fac5578c89d1e5f50d7f358c2/src/rulesets/LinearAlgebra/factorization.jl#L21
- ... ma `F, ∂F = ChainRules.frule((NoTangent(), A), lu!, A, Union{RowMaximum, NoPivot})` non funziona! Grrrrrr!!




"""