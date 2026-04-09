## Imports
using LinearAlgebra, GenericSchur, Polynomials
using Random, BenchmarkTools, Printf
using Revise

Revise.includet(joinpath(@__DIR__,"..","src","modules","MyMpExponential.jl"))
Revise.includet(joinpath(@__DIR__,"..","src","modules","MyHelper.jl"))
using .MyMpExponential, .MyHelper

Random.seed!(42)


## Define parameters and useful stuff



## First numerical check
print("""A small example with a tame matrix, just to see that it works: 
    - Small size, balanced
    - We compute the Schur form of `A`
    - … and replace the elements on the diagonals, using the `recompute_diagonals!`.
    - We check what happens
""")
n = 10;

A = rand(n,n);
A -= (tr(A)/n) * I(n);
F = schur(A);
T = F.T;

Tpows = [I(n), T^2];
# oss: Y is BigFloat (fault: the coefficients of pₘ and qₘ)
Y = expm_diagonal!(T, Tpows, 13, 0);    
Y_old = copy(Y);

# overwrites Y, putting the exponentials of the diagonal blocks in their respective places
recompute_diagonals!(big.(T), Y);   

@printf("|| Y - Y_recomputed || = %.6g\n", norm(Y_old - Y))
@printf("|| Y - Y_recomputed || / (1 + || Y || + || Y_recomputed ||) = %.6g\n", sym_err(Y, Y_old))

Y_true = exp(A);

@printf("|| Y - exp(T) || / || exp(T) || = %.6g\n", rel_err(Y_old, Y_true))
@printf("|| Y_recomputed - exp(T) || / || exp(T) || = %.6g\n", rel_err(Y, Y_true))


## Second numerical check
n = 6;
print("""We test a random matrix of size 2n, with n=$n, constructed in the following way:
    - pick a₁,…,aₙ and b₁,…,bₙ (at random)
    - set (hopefully) about half of the bₖs to zero
    - costruct T = blkdiag([a₁ -b₁; b₁ a₁],…,[aₙ -bₙ; bₙ aₙ])
There's an explicit formula for exp(T). 
We compute `Y=rₘ(T)` where `rₘ(x)≈exp(x)` is the diagonal Padé approximant of degree `m=13`.
We then recompute the diagonals of `Y` using the explicit formulas, as implemented by `recompute_diagonals!(T, Y)`.
Finally, we compare `rₘ(T)` and `Y` with `exp(T)`.
""")
a = rand(BigFloat, n);
b = 1e-2rand(BigFloat, n) .|> x -> rand()>=0.5 ? x : 0;

is_b_zero = b .≈ 0;
print("$(sum(is_b_zero)) imaginary parts are zero. "
 * "This should yield $(2*sum(is_b_zero)) real eigenvalues.\n")

T      = zeros(BigFloat, 2n,2n);
Y_true = zeros(BigFloat, 2n,2n);
for i = 1:n
    T[2i-1:2i, 2i-1:2i] = [a[i] -b[i]; b[i] a[i]];
    if b[i] == 0
        Y_true[2i-1, 2i-1] = exp(a[i])
        Y_true[2i, 2i] = exp(a[i])
    else 
        Y_true[2i-1:2i, 2i-1:2i] = 
            exp(a[i])*[cos(b[i]) -sin(b[i]); sin(b[i]) cos(b[i])]
    end
end

Tpows = [I(2n), T^2];
Y = expm_diagonal!(T, Tpows, 13, 0);
Y_old = copy(Y);

# overwrites Y, putting the exponentials of the diagonal blocks in their respective places
recompute_diagonals!(T, Y);   

@printf("|| Y - Y_recomputed || = %.6g\n", norm(Y_old - Y))
@printf("|| Y - Y_recomputed || / (1 + || Y || + || Y_recomputed ||) = %.6g\n", sym_err(Y, Y_old))

@printf("|| Y - exp(T) || / || exp(T) || = %.6g\n", rel_err(Y_old, Y_true))
@printf("|| Y_recomputed - exp(T) || / || exp(T) || = %.6g\n", rel_err(Y, Y_true))


