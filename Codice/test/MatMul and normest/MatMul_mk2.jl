using LinearAlgebra, Random, Printf
using Revise

Revise.includet(joinpath(@__DIR__,"..","..","src","modules","MyMpExponential.jl"))
Revise.includet(joinpath(@__DIR__,"..","..","src","modules","MyHelper.jl"))
Revise.includet(joinpath(@__DIR__,"..","..","src","modules","MyMatMul.jl"))
using .MyMpExponential, .MyHelper, .MyMatMul

Random.seed!(42)


## Define parameters and useful stuff
n = 5;
A = rand(n,n);

Apows_pade = [I(n), A^2];
Apows_tayl = [I(n), A];

## First test: the mk2 function (Padé version) on a random vector
d = 7;
X = rand(n);
print("We test the action of A^$d on a random vector (Padé version).\n")

Ad_action_pade = evalPowVecDiag_mk2(d, A, Apows_pade);

y_true = A^d * X;
y_pade = Ad_action_pade * X;

print("length(Apows_pade) = $(length(Apows_pade))\n")
@printf("|| y_pade - y_true || / || y_true || = %.4g\n", rel_err(y_pade, y_true))
print("y_pade ≈ y_true is $(y_pade ≈ y_true)\n")

print("\n")
print("Let's add some elements to Apows_pade\n\n")

push!(Apows_pade, A^4);
push!(Apows_pade, A^6);

Ad_action_pade = evalPowVecDiag_mk2(d, A, Apows_pade);

y_pade = Ad_action_pade * X;

print("length(Apows_pade) = $(length(Apows_pade))\n")
@printf("|| y_pade - y_true || / || y_true || = %.4g\n", rel_err(y_pade, y_true))
print("y_pade ≈ y_true is $(y_pade ≈ y_true)\n")


## Second test: the mk2 function (Taylor version) on a random vector
print("We test the action of A^$d on a random vector (Taylor version).\n")

Ad_action_tayl = evalPowVecDiag_mk2(d, A, Apows_tayl, use_taylor=true)

y_tayl = Ad_action_tayl * X;

print("length(Apows_tayl) = $(length(Apows_tayl))\n")
@printf("|| y_tayl - y_true || / || y_true || = %.4g\n", rel_err(y_tayl, y_true))
print("y_tayl ≈ y_true is $(y_tayl ≈ y_true)\n")

print("\n")
print("Let's add some elements to Apows_tayl\n\n")

push!(Apows_tayl, A^2);
push!(Apows_tayl, A^3);

Ad_action_tayl = evalPowVecDiag_mk2(d, A, Apows_tayl, use_taylor=true);

y_tayl = Ad_action_tayl * X;

print("length(Apows_tayl) = $(length(Apows_tayl))\n")
@printf("|| y_tayl - y_true || / || y_true || = %.4g\n", rel_err(y_tayl, y_true))
print("y_tayl ≈ y_true is $(y_tayl ≈ y_true)\n")


## Third test: the mk2 function on a random tall skinny matrix

function run_mk2_test(d, t=1)
    print("size(A) = $(size(A))\teltype(A) = $(eltype(A))\n")

    X = t == 1 ? rand(n) : rand(n,t)

    Y_true = A^d * X
    
    Ad_action_pade = evalPowVecDiag_mk2(d, A, Apows_pade)
    Y_pade = Ad_action_pade * X

    if t > 1
        print("\ntypeof(Y_pade) = $(typeof(Y_pade)). We need to materialize it, "
         * "to compute its norm. We do it by running `Y_pade = Matrix(Y_pade)`.\n\n")
        Y_pade = Matrix(Y_pade)
    end

    print("Testing the two ways to apply A^$d on a random X (Padé version)\n")
    @printf("|| Y_pade - Y_true || / || Y_true || = %.4g\n", rel_err(Y_pade, Y_true))
    print("Y_pade ≈ Y_true is $(Y_pade ≈ Y_true)\n")

    Ad_action_tayl = evalPowVecDiag_mk2(d, A, Apows_tayl, use_taylor=true)
    Y_tayl = Ad_action_tayl * X

    if t > 1 
        Y_tayl = Matrix(Y_tayl)
    end

    print("Testing the two ways to apply A^$d on a random X (Taylor version)\n")
    @printf("|| Y_tayl - Y_true || / || Y_true || = %.4g\n", rel_err(Y_tayl, Y_true))
    print("Y_tayl ≈ Y_true is $(Y_tayl ≈ Y_true)\n\n")
end

t = 3;
run_mk2_test(12, t)
run_mk2_test(13, t) # odd degree

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

## Fourth test: does the mk2 provide us the action of the transpose conj as well?
print("We perform a small test to check if also the action of the adjoint works.\n")

d = 7;
x = rand(n);

y_p_true = (A')^d * x;

Ad_action_pade = evalPowVecDiag_mk2(d, A, Apows_pade)
y_p_pade = Ad_action_pade' * x;

print("Testing the two ways to apply (A')^$d on a random x (Padé version)\n")
@printf("|| y_p_pade - y_p_true || / || y_p_true || = %.4g\n", rel_err(y_p_pade, y_p_true))
print("y_p_pade ≈ y_p_true is $(y_p_pade ≈ y_p_true)\n")

Ad_action_tayl = evalPowVecDiag_mk2(d, A, Apows_tayl, use_taylor=true)
y_p_tayl = Ad_action_tayl' * x;

print("Testing the two ways to apply (A')^$d on a random x (Padé version)\n")
@printf("|| y_p_tayl - y_p_true || / || y_p_true || = %.4g\n", rel_err(y_p_tayl, y_p_true))
print("y_p_tayl ≈ y_p_true is $(y_p_tayl ≈ y_p_true)\n")
