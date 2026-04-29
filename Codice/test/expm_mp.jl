## Imports
using LinearAlgebra
using ChainRules
using Random, Printf
using BenchmarkTools
using Revise

Revise.includet(joinpath(@__DIR__,"..","src","modules","MyMpExponential.jl"))
Revise.includet(joinpath(@__DIR__,"..","src","modules","MyHelper.jl"))
using .MyMpExponential, .MyHelper

Random.seed!(42)


## Define parameters and useful stuff
function hadamard(n::Int)
    ispow2(n) || throw(ArgumentError("n must be a power of 2"))

    H = ones(Int, 1, 1)
    while size(H,1) < n
        H = [ H   H;
              H  -H ]
    end
    return H
end

function run_exp_test(
    A::AbstractMatrix{T}, 
    Y_true=nothing
) where {T}
    print("size(A) = $(size(A, 1)),\teltype(A) = $(eltype(A)),\t")
    print("ishermitian(A) is $(ishermitian(A))\n")

    T_low = T <: Complex ? ComplexF64 : Float64
    A_low = convert(Matrix{T_low}, A)

    if isnothing(Y_true)
        print("Using Base.exp as reference solution.\n")
        Y_true = exp(A_low)
    else 
        print("True exponential of A has been provided.\n")
        Y_base = exp(A_low)
        @printf("\t‖ exp(A) - Y_base ‖ / ‖ exp(A) ‖ = %.6g\n", rel_err(Y_base, Y_true))
        print("\teltype(Y_base) = $(eltype(Y_base)).\n")
    end
    print("\n")

    for appx in [:diagonalcheap, :diagonal, :taylor]
        ALGS = [:transfree, :complexschur]
        if T <: Real 
            push!(ALGS, :realschur)
        end
        for alg in ALGS
            print("Approximant = $appx,\talgorithm = $alg.\n")
            Y = exp_mp(A, approximant=appx, algorithm=alg)

            @printf("\t‖ Y - exp(A) ‖ / ‖ exp(A) ‖ = %.6g\n", rel_err(Y, Y_true))
            print("\teltype(Y) = $(eltype(Y)).\n\n")
        end
        print("\n")
    end
    print("\n\n\n")
end


function compute_cond_exp(A)
    n = LinearAlgebra.checksquare(A)
    T = eltype(A)
    T_low = T <: Complex ? ComplexF64 : Float64

    A_low = convert(AbstractMatrix{T_low}, A)
    Y_true, exp_pullback = ChainRules.rrule(exp, A_low)

    K = construct_full_jacobian(x -> exp_pullback(x)[2], n, T_low)

    #κ_exp(A) = (‖K‖₂⋅‖A‖_F) / ‖exp(A)‖_F
    opnorm(K, 2) * norm(A) / norm(Y_true)
end



## First numerical test 
n = 2 << 7;
A = randn(n,n);

#@benchmark exp_mp(A, approximant=:taylor)

run_exp_test(A)


## Second numerical test
n = 2 << 7;
A = randn(BigFloat, n,n);

run_exp_test(A)


## Third numerical test
n = 2 << 4;
H = hadamard(n);
H /= sqrt(n);   # adesso H*H' = I(n)
D = Diagonal(100rand(n).-50 + 100im*rand(n).-50);
# oss: se D è reale, A è Hermitiana (quindi l'alg. diagonalizza)
A = H' * D * H;    

Y_true = H' * exp(D) * H;

run_exp_test(A, Y_true)


## Fourth numerical test
n = 2 << 4;
H = hadamard(n);
H /= sqrt(big(n));
D = Diagonal(100rand(BigFloat, n).-50 + 100im*rand(BigFloat, n).-50);
# oss: se D è reale, A è Hermitiana (quindi l'alg. diagonalizza)
A = H' * D * H;   

Y_true = H' * exp(D) * H;

run_exp_test(A, Y_true)