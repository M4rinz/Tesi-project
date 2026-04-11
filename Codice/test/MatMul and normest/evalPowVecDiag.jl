## Imports
using LinearAlgebra, Random, Printf
using LinearMaps
using Revise

Revise.includet(joinpath(@__DIR__,"..","..","src","modules","MyMpExponential.jl"))
Revise.includet(joinpath(@__DIR__,"..","..","src","modules","MyHelper.jl"))
using .MyMpExponential, .MyHelper
#using .MyMpExponential: AandPowsStruct

Random.seed!(42)


## Define parameters and useful stuff
n = 5;
A = rand(n,n);

A_struct_tayl = AandPowsStruct(A, true);
A_struct_pade = AandPowsStruct(A, false);

function run_evalPowVecDiag_test(d, t=1)
    print("size(A) = $(size(A))\teltype(A) = $(eltype(A))\n")

    X = t == 1 ? rand(n) : rand(n,t)

    Y_true = A^d * X
    
    Ad_action_pade = evalPowVecDiag(d, A_struct_pade)
    Y_pade = Ad_action_pade * X

    if t > 1
        if Y_pade isa LinearMap
            print("\n Y_pade is a LinearMap. We need to materialize it, "
             * "to compute its norm. We do it by running `Y_pade = Matrix(Y_pade)`.\n\n")
            Y_pade = Matrix(Y_pade)
        else 
            print("\ntypeof(Y_pade) = $(typeof(Y_pade))")
        end        
    end

    print("Testing the two ways to apply A^$d on a random X (Padé version)\n")
    @printf("|| Y_pade - Y_true || / || Y_true || = %.4g\n", rel_err(Y_pade, Y_true))
    print("Y_pade ≈ Y_true is $(Y_pade ≈ Y_true)\n")

    Ad_action_tayl = evalPowVecDiag(d, A_struct_tayl)
    Y_tayl = Ad_action_tayl * X

    if t > 1 
        Y_tayl = Matrix(Y_tayl)
    end

    print("Testing the two ways to apply A^$d on a random X (Taylor version)\n")
    @printf("|| Y_tayl - Y_true || / || Y_true || = %.4g\n", rel_err(Y_tayl, Y_true))
    print("Y_tayl ≈ Y_true is $(Y_tayl ≈ Y_true)\n\n\n")
end


## First test: version of evalPowVecDiag that uses the struct (Padé version)
d = 7;
run_evalPowVecDiag_test(d)


## Second test: the mk2 function on a random tall skinny matrix
t = 3;

run_evalPowVecDiag_test(12, t)
run_evalPowVecDiag_test(13, t)

print("""Lessons learnt:
    - It's a good idea to apply a `LinearMap` as `Map * X` instead of `Map(X)`
    - If `X` is a vector, then `Map * X` returns a vector. Otherwise, it returns 
      something strange.
    - This "something strange" can be materialized into a standard `Matrix`, using 
      the constructor
    - If one wants to compute the norm of `Y = Map * X` using the `norm` or `opnorm`
      commands (from `Base`), one should materialize `Y`
      - Idea: use multiple dispatch to actually use `opnorm1` from `MatrixEquations`
              when one calls `opnorm(Map * X, 1)`. [Not implemented. Is it worth?]
""")


## Third test: does evalPowVecDiag also provide us the action of the Adjoint?
print("We perform a small test to check if also the action of the adjoint works.\n")

d = 7;
x = rand(n);

y_p_true = (A')^d * x;

Ad_action_pade = evalPowVecDiag(d, A_struct_pade)
y_p_pade = Ad_action_pade' * x;

print("Testing the two ways to apply (A')^$d on a random x (Padé version)\n")
@printf("|| y_p_pade - y_p_true || / || y_p_true || = %.4g\n", rel_err(y_p_pade, y_p_true))
print("y_p_pade ≈ y_p_true is $(y_p_pade ≈ y_p_true)\n")

Ad_action_tayl = evalPowVecDiag(d, A_struct_tayl)
y_p_tayl = Ad_action_tayl' * x;

print("Testing the two ways to apply (A')^$d on a random x (Padé version)\n")
@printf("|| y_p_tayl - y_p_true || / || y_p_true || = %.4g\n", rel_err(y_p_tayl, y_p_true))
print("y_p_tayl ≈ y_p_true is $(y_p_tayl ≈ y_p_true)\n")