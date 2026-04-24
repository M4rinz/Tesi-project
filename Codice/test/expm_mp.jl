## Imports
using LinearAlgebra
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

    for appx in [:diagonalcheap, :diagonal, :taylor]
        ALGS = [:transfree, :complexschur]
        if T <: Real 
            push!(ALGS, :realschur)
        end
        for alg in ALGS
            print("Approximant = $appx,\talgorithm = $alg.\n")
            Y = exp_mp(A, approximant=appx, algorithm=alg)

            @printf("\t‖ Y - exp(A) ‖ / ‖ exp(A) ‖ = %.6g\n", rel_err(Y, Y_true))
            print("\teltype(Y) = $(eltype(Y)).\n")
        end
        print("\n")
    end
    print("\n\n")
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
H /= sqrt(n);
D = Diagonal(100rand(n).-50 + 100im*rand(n).-50);
# oss: se D è reale, A è Hermitiana (quindi l'alg. diagonalizza)
A = H' * D * H;    

Y_true = H' * exp.(D) * H;

run_exp_test(A, Y_true)


## Fourth numerical test
n = 2 << 4;
H = hadamard(n);
H /= sqrt(n);
D = Diagonal(100rand(BigFloat, n).-50 + 100im*rand(BigFloat, n).-50);
# oss: se D è reale, A è Hermitiana (quindi l'alg. diagonalizza)
A = H' * D * H;   

Y_true = H' * exp.(D) * H;

run_exp_test(A, Y_true)