using LinearAlgebra, Random, Printf
using Revise

Revise.includet(joinpath(@__DIR__,"..","..","src","modules","MyMpExponential.jl"))
Revise.includet(joinpath(@__DIR__,"..","..","src","modules","MyHelper.jl"))
Revise.includet(joinpath(@__DIR__,"..","..","src","modules","MyMatMul.jl"))
using .MyMpExponential, .MyHelper, .MyMatMul

Random.seed!(42)


## Define parameters and useful stuff

function opnorm_test(d, A, Apows, use_taylor)
    n = LinearAlgebra.checksquare(A)
    T = eltype(A)
    print("Testing the computation of the 1-norm of A.")
    print("\tsize(A) = $n,\t eltype(A)=$T\n")
    method = use_taylor ? "Taylor" : "Padé"
    print("Chosen approximant: " * method * ".\tlength(Apows) = $(length(Apows))\n\n")
    
    γ_d_true    = opnorm(A^d, 1)
    γ_d_approx  = normest1(d, A, Apows, use_taylor=use_taylor)

    @printf("|| γ - ̂γ || / || γ || = %.6g\n", rel_err(γ_d_approx, γ_d_true))
    @printf("Precisely: γ = %6.9g,\t ̂γ = %6.9g\n", γ_d_true, γ_d_approx)
    print("\n")
end 



## First test
print("We check the how well the 1-norm of a very low power of a small random matrix "
 * "with Float64 elements is approximated.\n")

n = 5;
A = rand(n,n);
d = 3;

opnorm_test(d, A, [I(n), A^2], false)   # Padé
opnorm_test(d, A, [I(n), A], true)      # Taylor


## Second test
print("Similar to previous test, but the matrix power `d` and the size `n` "
 * "are slightly higher.\n")

n = 20;
A = rand(n,n);
d = 7; 

opnorm_test(d, A, [I(n), A^2, A^4], false)   # Padé
opnorm_test(d, A, [I(n), A, A^2], true)      # Taylor


## Third test
print("We now check the accuracy of the computation of the 1-norm "
 * "of a small power of a matrix with BigFloat elements.\n")

n = 5;
A = rand(BigFloat, n,n);
d = 3;

opnorm_test(d, A, [I(n), A^2], false)   # Padé
opnorm_test(d, A, [I(n), A], true)      # Taylor


## Fourth test
print("Similar to previous test. In fact, analogous to test n°2, "
 * "but with a BigFloat matrix.\n")

n = 20;
A = rand(BigFloat, n,n);
d = 9; 

opnorm_test(d, A, [I(n), A^2, A^4], false)   # Padé
opnorm_test(d, A, [I(n), A, A^2], true)      # Taylor