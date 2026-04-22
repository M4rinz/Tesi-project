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



## First numerical test 
n = 2 << 7;
A = rand(n,n);

Y_true = exp(A);

@benchmark exp_mp(A, approximant=:taylor)

Y_pade = exp_mp(A, approximant=:diagonal);
@printf("‖ Y_pade - exp(A) ‖ / ‖ exp(A) ‖ = %.6g\n", rel_err(Y_pade, Y_true))
print("\n")

Y_tayl = exp_mp(A, approximant=:taylor);
@printf("‖ Y_tayl - exp(A) ‖ / ‖ exp(A) ‖ = %.6g\n", rel_err(Y_tayl, Y_true))
print("\n")