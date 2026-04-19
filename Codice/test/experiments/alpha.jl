## Imports
using LinearAlgebra, LinearMaps, MatrixEquations
using Random, Printf
#using Plots, BenchmarkTools
using Revise

Revise.includet(joinpath(@__DIR__,"..","..","src","modules","MyMpExponential.jl"))
Revise.includet(joinpath(@__DIR__,"..","..","src","modules","MyHelper.jl"))
using .MyMpExponential, .MyHelper

Random.seed!(42)


## Define parameters and useful stuff
maxdegree = 500;
ms = opt_degs_pade(maxdegree);   # contains all optimal `m` up to `maxdegree`

alpha_vec = zeros(maxdegree);

s = 0;  # no scaling pls

function d(m, k)  
    d = fld(1 + sqrt(4*(m+k) + 5), 2)
    Int64(d)
end

## First test
n = 10;
A = rand(n,n);
S = AandPowsStruct(A, false);
alpha_vec[1] = opnorm(A, 1);

for m in ms[2:10]
    print("m = $m\n")
    α_min = alpha!(alpha_vec, S, s, m,m);  
    d_val = d(m,m); 
    print("d was $(d_val)\n")
    display(alpha_vec[1:d(ms[10], ms[10])])
    print("\n")
end