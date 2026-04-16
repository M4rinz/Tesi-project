using LinearAlgebra, Random, Printf
using Revise

Revise.includet(joinpath(@__DIR__,"..","..","src","modules","MyMpExponential.jl"))
Revise.includet(joinpath(@__DIR__,"..","..","src","modules","MyHelper.jl"))
using .MyMpExponential, .MyHelper

Random.seed!(42)


## Define parameters and useful stuff
s = 4;      # For example...
m_tayl_arr = [floor(Int64, (i+2)^2/4) for i=1:10];   # Optimal degrees for Taylor
m_pade_arr = [1, 2];
for i=2:length(m_tayl_arr)-1
    bi = 2*ceil(Int64, (i-1)/4)*(i-1-2*div(i-2,4))+1
    push!(m_pade_arr, bi)
end                                                  # Optimal degrees for Padé 
                                
function horner(X, coeffs, s)
    n = LinearAlgebra.checksquare(X)
    scaling = 2^s

    Xsc = X / scaling
    Y = copy(Xsc)

    length(coeffs) == 1 && return coeffs[1] * I(n)

    Y = coeffs[end] * Y + coeffs[end-1] * I(n)
    for k = length(coeffs)-2:-1:1
        Y = Y * Xsc + coeffs[k] * I(n)
    end

    return Y
end


## Check symbolically
# using Symbolics
# @variables a b

# M = Diagonal([a, b])
# Mpows = [I(size(M,1)), M^2];
# m = m_tayl_arr[2]

# M_tayl_true = simplify.(expand.(tayl_exp_horner(M, m, 0)))

# M_pade = expm_diagonal!(M, Mpows, m, 0, cheap_r=false)

print("The Symbolic check is experiencing some issues (and is not particularly revealing anyway)\n")


## First numerical check
print("We check our `expm_pade!` (which in turn calls `expm_diagonal!` internally) "
 * "against what is obtained with the Horner evaluation scheme "
 * "(for the same Padé approximant of course), using a small random BigFloat matrix\n")

n = 5;
A = rand(BigFloat, n,n);

S_pade = AandPowsStruct(A, false);
#S_tayl = AandPowsStruct(A, true);

m = big(m_pade_arr[7])
pade_num_coeffs = [binomial(m,j)*factorial(2m-j)/factorial(2m) for j=0:m];
pade_den_coeffs = (-1).^(0:m) .* pade_num_coeffs;
Pₘ  = horner(A, pade_num_coeffs, 0);
Qₘ  = horner(A, pade_den_coeffs, 0);
Y_h = Qₘ \ Pₘ;

#_    = expm_diagonal!(S_tayl, m, 0);    # Just to check if the error is caught
Y_ps = eval_pade!(S_pade, m, 0);
@printf("|| Y_ps - Y_h || / || Y_h || = %.4g\n", rel_err(Y_ps, Y_h))
print("Y_ps ≈ Y_h is $(Y_ps ≈ Y_h)\n")
print("Y_ps ≈ Y_h (up to machine precision) is $(isapprox(Y_ps, Y_h, rtol=eps(BigFloat)))\n")
print("S.powers contains $(length(S_pade.powers)) matrices.\n")


## Second numerical check
print("We construct a \"tame\" matrix for whom the Padé approx. of the matrix "
 * "exponential works well, and compare our methods to evaluate rational functions (basically)\n")

A  = rand(n, n);
A  = (A + A')/2;
A -= (tr(A)/n) * I(n)   # shift 

S = AandPowsStruct(A, false);

Y_true = exp(Float64.(A));

m = big(m_pade_arr[7])

Pₘ = horner(A, pade_num_coeffs, 0);
Qₘ = horner(A, pade_den_coeffs, 0);
Y_h = Qₘ \ Pₘ;

@printf("|| Y_h - exp(A) || / || exp(A) || = %.4g\n", rel_err(Y_h, Y_true))
print("Y_h ≈ exp(A) is $(Y_h ≈ Y_true)\n")

Y_ps = eval_pade!(S, m, 0);
@printf("|| Y_ps - exp(A) || / || exp(A) || = %.4g\n", rel_err(Y_ps, Y_true))
print("Y_ps ≈ exp(A) is $(Y_ps ≈ Y_true)\n")

print("Ok, we've obtained good approximations of exp(A). But the crucial point is: how close are Y_h and Y_ps?\n")
@printf("|| Y_ps - Y_h || / || Y_h || = %.4g\n", rel_err(Y_ps, Y_h))
print("Y_ps ≈ Y_h (up to (Float64) machine precision) is $(isapprox(Y_ps, Y_h, rtol=eps()))\n")

print("Note that eltype(A) = $(eltype(A)), eltype(Y_h) = $(eltype(Y_h)), eltype(Y_ps) = $(eltype(Y_ps))\n")
print("\n")
