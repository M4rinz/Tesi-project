## Imports
using LinearAlgebra, Random, Printf
using Revise

Revise.includet(joinpath(@__DIR__,"..","src","modules","MyMpExponential.jl"))
Revise.includet(joinpath(@__DIR__,"..","src","modules","MyHelper.jl"))
using .MyMpExponential, .MyHelper

Random.seed!(42)


## Define parameters and useful stuff

function run_tayl_err_test(A, m, s=0)
    if eltype(A) == Float64
        Y_true = exp(2^(-s)*A)
    end

    S = AandPowsStruct(A, true)
    Y_tayl = eval_pade!(S, m, s)

    true_err = opnorm(Y_true - Y_tayl, 1)
    nrm1Y = opnorm(Y_true, 1)
    @printf("‖Y_tayl - exp(2^(-s)A)‖₁ / ‖exp(2^(-s)A)‖₁ = %.6g\n", true_err/nrm1Y)

    factorials = FactorialsStruct()
    alpha_vec = zeros(m)
    alpha_vec[1] = opnorm(A, 1)
    k = 0
    α = alpha!(alpha_vec, S, s, k, m)

    δ, ψ, _ = eval_error(S, α, m, 0, true, factorials)

    @printf("‖ Y_tayl - exp(A) ‖₁ = %.4g,\tδ = |Tₘ(α) - exp(α)| = %.4g\n", true_err, δ)
    true_err ≤ δ || @warn "δ should be an upper bound...\n"

    @printf("‖ exp(A) ‖₁ = %.6g,\tψ = ‖ Tₗ(A) ‖₁ = %.6g\n", nrm1Y, ψ)
    isapprox(ψ, nrm1Y, rtol=0.1) || @warn "ψ should approximate ‖ exp(A) ‖₁..."
end



## First numerical test
n = 10;
m = 6;
A = 0.1rand(n,n);
A -= (tr(A)/n) * I(n);

m_tayl = opt_degs_tayl(21)
for m in m_tayl[3:end]
    print("m = $m\n")
    run_tayl_err_test(A, m, 0)
    print("\n")
end

print("Problem: for high m values exp(A) is well approximated by Tₘ. "
 * "the relative error is ≈ eps(), but δ is smaller! ")

