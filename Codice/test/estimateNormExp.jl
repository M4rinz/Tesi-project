## Imports
using LinearAlgebra, Random
using LinearMaps, MatrixEquations
using BenchmarkTools
using Revise

Revise.includet(joinpath(@__DIR__,"..","src","modules","MyMpExponential.jl"))
Revise.includet(joinpath(@__DIR__,"..","src","modules","MyHelper.jl"))
using .MyHelper, .MyMpExponential

Random.seed!(42)


## Define parameters and useful stuff
n = 500;
A = rand(n,n);
S = AandPowsStruct(A, true);
l = 8;  # will be length(S.powers)

for i = length(S.powers):l-1
    push!(S.powers, A^(i));
end
S.powers ≈ [A^k for k=0:l-1] || error("Hai sbagliato a inizializzare…\n")

s = 0;

T_l_action = function (X::AbstractVecOrMat; downcast::Bool=true)
        T = downcast ? Float64 : eltype(S.A)
        coeff = one(eltype(X))
        XR    = copy(X)
        for i=2:length(S.powers)
            coeff *= 2.0^(-s) / (i-1) 
            XR += coeff .* (T.(S.powers[i]) * X)
        end
        return XR
    end;
T_l_p_action = function (X::AbstractVecOrMat; downcast::Bool=true)
        T = downcast ? Float64 : eltype(S.A)    
        coeff = one(eltype(X))
        XR    = copy(X)
        for i=2:length(S.powers)
            coeff *= 2.0^(-s) / (i-1) 
            XR += coeff .* (T.(S.powers[i]') * X)
        end
        return XR
    end;
TLM = LinearMap{eltype(S.A)}(T_l_action, T_l_p_action, size(S.A, 1));

T_l_action_mk2 = function (X::AbstractVecOrMat)
        XR    = copy(X)                 # A^0 * X
        term  = copy(X)                 # current A^k * X
        coeff = one(eltype(X))

        for k in 1:length(S.powers)-1   # k = 1,2,...
            coeff *= 2.0^(-s) / k
            term = S.A * term           # now term = A^k * X
            XR += coeff .* term
        end

        return XR
    end
T_l_p_action_mk2 = function (X::AbstractVecOrMat)
        XR    = copy(X)                 # A^0 * X
        term  = copy(X)                 # current A^k * X
        coeff = one(eltype(X))

        for k in 1:length(S.powers)-1   # k = 1,2,...
            coeff *= 2.0^(-s) / k
            term = (S.A)' * term           # now term = A^k * X
            XR += coeff .* term
        end

        return XR
    end
TLM_mk2 = LinearMap{eltype(S.A)}(T_l_action_mk2, T_l_p_action_mk2, size(S.A, 1));

# la mk2 sembra convenire


## First numerical experiment
n = 100;
A = rand(BigFloat, n,n);
S = AandPowsStruct(A, true);
l = 8;  # will be length(S.powers)

for i = length(S.powers):l-1
    push!(S.powers, A^(i));
end
S.powers ≈ [A^k for k=0:l-1] || error("Hai sbagliato a inizializzare…\n")

numerators = [2.0^(-s*k) for k=0:l-1];
factorials_double = factorial.(0:l-1);
coeffs = numerators ./ factorials_double;
approx = sum(coeffs .* S.powers);
ref = opnorm(approx, 1);

LA = LinearMap{Float64}(T_l_action, T_l_p_action, size(S.A, 1))

est = opnorm1est(LA);

print("\tBenchmarking true norm 1 computation:\n")
@benchmark opnorm($approx, 1)

print("\tBenchmarking norm 1 estimation:\n")
@benchmark opnorm1est($LA)

# molto male, parrebbe...

rel_err(est, ref)


# prova con una lista separata che ha le potenze già in doppia precisione