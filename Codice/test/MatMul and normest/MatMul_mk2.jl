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
y_pade = Ad_action_pade(X);

print("length(Apows_pade) = $(length(Apows_pade))\n")
@printf("|| y_pade - y_true || / || y_true || = %.4g\n", rel_err(y_pade, y_true))
print("y_pade ≈ y_true is $(y_pade ≈ y_true)\n")

print("\n")
print("Let's add some elements to Apows_pade\n\n")

push!(Apows_pade, A^4);
push!(Apows_pade, A^6);

Ad_action_pade = evalPowVecDiag_mk2(d, A, Apows_pade);

y_pade = Ad_action_pade(X);

print("length(Apows_pade) = $(length(Apows_pade))\n")
@printf("|| y_pade - y_true || / || y_true || = %.4g\n", rel_err(y_pade, y_true))
print("y_pade ≈ y_true is $(y_pade ≈ y_true)\n")


## Second test: the mk2 function (Taylor version) on a random vector
print("We test the action of A^$d on a random vector (Taylor version).\n")

Ad_action_tayl = evalPowVecDiag_mk2(d, A, Apows_tayl, use_taylor=true)

y_tayl = Ad_action_tayl(X);

print("length(Apows_tayl) = $(length(Apows_tayl))\n")
@printf("|| y_tayl - y_true || / || y_true || = %.4g\n", rel_err(y_tayl, y_true))
print("y_tayl ≈ y_true is $(y_tayl ≈ y_true)\n")

print("\n")
print("Let's add some elements to Apows_tayl\n\n")

push!(Apows_tayl, A^2);
push!(Apows_tayl, A^3);

Ad_action_tayl = evalPowVecDiag_mk2(d, A, Apows_tayl, use_taylor=true);

y_tayl = Ad_action_tayl(X);

print("length(Apows_tayl) = $(length(Apows_tayl))\n")
@printf("|| y_tayl - y_true || / || y_true || = %.4g\n", rel_err(y_tayl, y_true))
print("y_tayl ≈ y_true is $(y_tayl ≈ y_true)\n")


