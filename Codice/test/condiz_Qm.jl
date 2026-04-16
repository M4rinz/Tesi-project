## Imports
using LinearAlgebra, LinearMaps, MatrixEquations
using Random, Printf
using Plots, BenchmarkTools
using Revise

Revise.includet(joinpath(@__DIR__,"..","src","modules","MyMpExponential.jl"))
Revise.includet(joinpath(@__DIR__,"..","src","modules","MyHelper.jl"))
using .MyMpExponential, .MyHelper

Random.seed!(42)


## Define parameters and useful stuff
n = 50;

function condQ_naive(Uₑ, Uₒ)
    Q = Uₑ - Uₒ
    n = LinearAlgebra.checksquare(Q)

    tempQinv = inv(Q)
    tempnormQinv = opnorm(tempQinv, 1)          # η
    tempcondQ    = opnorm(Q, 1) * tempnormQinv  # κ (= η⋅‖Q‖₁)

    tempnormExpm = opnorm(2 * (Q \ Uₒ) + I(n))  # ψ

    return tempnormExpm, tempcondQ
end

function condQ_LU(Uₑ, Uₒ; exact::Bool=false)
    Q = Uₑ - Uₒ
    F = lu(Q)
    opinv = InverseMap(F)

    # think: direi che poiché Q è float alla fine va bene questo 
    #        (punto: oprcondest è pensata per precisione singola o doppia, 
    #         mentre opnorm1est maneggia anche le BigFloat per qualche ragione)
    invcondQ = oprcondest(LinearMap(Q), opinv, exact=exact) # 1/κ
    normExpm = opnorm1est(2 * opinv * Uₒ + I(n))            # ψ

    return normExpm, 1/invcondQ
end


## First test: just random matrices
Uₑ, Uₒ = rand(n,n), rand(n,n);

Qₘ = Uₑ - Uₒ;

ψ_naive, κ_naive = condQ_naive(Uₑ, Uₒ);
bench_naive = @benchmark condQ_naive($Uₑ, $Uₒ);
print("\tNaive benchmark:\n")
display(bench_naive)
print("\n")

ψ_est, κ_est = condQ_LU(Uₑ, Uₒ);
bench_est = @benchmark condQ_LU($Uₑ, $Uₒ);
print("\tBenchmark using norm estimate:\n")
display(bench_est)
print("\n")

