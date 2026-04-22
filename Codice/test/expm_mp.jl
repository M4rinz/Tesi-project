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


## First numerical test 
n = 2 << 7;
A = rand(n,n);

Y_true = exp(A);

#@benchmark exp_mp(A, approximant=:taylor)

Y_pade = exp_mp(A, approximant=:diagonal);
@printf("‖ Y_pade - exp(A) ‖ / ‖ exp(A) ‖ = %.6g\n", rel_err(Y_pade, Y_true))
print("\n")

Y_tayl = exp_mp(A, approximant=:taylor);
@printf("‖ Y_tayl - exp(A) ‖ / ‖ exp(A) ‖ = %.6g\n", rel_err(Y_tayl, Y_true))
print("\n")


## Second numerical test
n = 2 << 4;
H = hadamard(n);
H /= sqrt(n);
D = Diagonal(100rand(n).-50 + 100im*rand(n).-50);
# oss: se D è reale, A è Hermitiana (quindi l'alg. diagonalizza)
A = H' * D * H;    

Y_true = H' * exp.(D) * H;
Y_base = exp(A);

@printf("‖ exp(A) - Y_base ‖ / ‖ exp(A) ‖ = %.6g\n", rel_err(Y_base, Y_true))

Y_pade = exp_mp(A, approximant=:diagonal);
@printf("‖ Y_pade - exp(A) ‖ / ‖ exp(A) ‖ = %.6g\n", rel_err(Y_pade, Y_true))

Y_tayl = exp_mp(A, approximant=:taylor);
@printf("‖ Y_tayl - exp(A) ‖ / ‖ exp(A) ‖ = %.6g\n", rel_err(Y_tayl, Y_true))

print("eltype(Y_base) = $(eltype(Y_base)),\teltype(Y_tayl) = $(eltype(Y_tayl)),\teltype(Y_pade) = $(eltype(Y_pade))\n")