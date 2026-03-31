using LinearAlgebra
using ChainRules, ChainRulesCore
using Random

# Include my modules

include(joinpath(@__DIR__,"..","src","modules","MyHelper.jl"))
using .MyHelper

include(joinpath(@__DIR__,"..","src","modules","MyBaseExponential.jl"))
using .MyBaseExponential


# Helper functions
function ChainRulesCore.frule(
    (_, ΔX), ::typeof(_balance), X::StridedMatrix{<:LinearAlgebra.BlasFloat};
    job::AbstractChar='B'
)
    Y, (ilo, ihi, scale) = _balance(X, job = job)

    ΔY = _balance!(copy(ΔX), ilo, ihi, scale)
    ∂ilo, ∂ihi = NoTangent(), NoTangent()
    ∂scale     = ZeroTangent()
    
    return (Y, (ilo, ihi, scale)), (ΔY, (∂ilo, ∂ihi, ∂scale))
end


function ChainRulesCore.rrule(
    ::typeof(_balance), A::StridedMatrix{<:LinearAlgebra.BlasFloat};
    job::AbstractChar='B'
)
    # primal computation (basically to retrieve transformations)
    y = _balance(A, job = job)
    _, (ilo, ihi, scale) = y
    function _balance_pullback(ȳ)
        ȳ_Y, _ = ȳ

        Ā = copy(ȳ_Y)   # don't wanna mutate the received output sensitivity
        n = LinearAlgebra.checksquare(Ā)
        _unbalance!(Ā, ilo, ihi, scale, n)

        return NoTangent(), Ā
    end
    return y, _balance_pullback
end




# actual code

seed = 42;
Random.seed!(seed);

n = 10;
# A_original: alleged end result of applying `gebal` transformations
A, A_original = gebal_example(n, seed=seed);
cpA = copy(A)


E = 0.001 * rand(n,n);
(A_bal, (ilo, ihi, scale)), (∂A_bal__∂E, _) = 
    ChainRules.frule((NoTangent(), E), _balance, A)

print("The rule worked. Is it correct, though?\n")
ApE_bal, (ilo_ApE, ihi_ApE, scale_ApE) = _balance(A+E);
remainder_fw = ApE_bal - A_bal - ∂A_bal__∂E;
print("|| remainder_fw || = $(norm(remainder_fw))\n")
print("|| remainder_fw || / || E || = $(norm(remainder_fw)/norm(E))\n")
print("What a disaster! The issue is that in defining the rule we somewhat "*
"assumed that the permutation and the diagonal scaling are constant in the differentiation. " *
"As if they don't depend on A (indeed they do).\n")
print("Simply put, LAPACK.gebal!(A) = PD^{-1}ADP'. But D=D(A) and P=P(A) and we ignored it.\n")
print("I was conscious of this. The issue here is that A has a specific structure that induces " * 
"certain transformations. This structure is completely destroyed in A + E, despite the fact that ||E|| << ||A||. " *
"A + E undergoes very different transformations than the ones of A\n")

print("(ilo, ihi, scale) for A = $((ilo, ihi, scale))\n")
print("(ilo, ihi, scale) for A + E = $((ilo_ApE, ihi_ApE, scale_ApE))\n")

print()
print("This makes differentiating through LAPACK.gebal! pretty hard conceptually, prima ancora che programmatically\n")
print("And is another nail in the coffin of the automatic differentiation of the algorithm for the matrix exponential.\n")