using LinearAlgebra, Random, Printf
using Plots, BenchmarkTools
using Revise

Revise.includet(joinpath(@__DIR__,"..","..","src","modules","MyMpExponential.jl"))
Revise.includet(joinpath(@__DIR__,"..","..","src","modules","MyHelper.jl"))
Revise.includet(joinpath(@__DIR__,"..","..","src","modules","MyMatMul.jl"))
using .MyMpExponential, .MyHelper, .MyMatMul

Random.seed!(42)


## Define parameters and useful stuff


## First Benchmark
# idea: how the performance scales with the size `n`, keeping 
#       fixed the number of precomputed powers in `Apows`
print("Still TODO!\t(is it important though)?")

# idea: does the accuracy start to dwindle, as `n` increases?

## Second benchmark 
# idea: Let's fix a size and a power `d`. How does the cost change 
#       if we vary the number of elements in `Apows` ?
print("Still TODO!\t(is it important though)?")

# idea: does the accuracy start to dwindle, as `d` increases?