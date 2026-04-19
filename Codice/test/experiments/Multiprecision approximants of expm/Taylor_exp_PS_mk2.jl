## Imports
using LinearAlgebra, Random, Printf
using GenericSchur
using Revise

Revise.includet(joinpath(@__DIR__,"..","..","..","src","modules","MyMpExponential.jl"))
Revise.includet(joinpath(@__DIR__,"..","..","..","src","modules","MyHelper.jl"))
using .MyMpExponential, .MyHelper

Random.seed!(42)

## Define parameters and useful stuff
s = 4;          # For example...
m_array = [floor(Int64, (i+2)^2/4) for i=1:8]   # Optimal degrees for Taylor


## Check symbolically
print("The symbolic check is incompatible with the new signature of `expm_taylor`.\n"
* "Anyway, it was there just to show that the degrees have to be optimal (not all `m`s will do).\n")
# using Symbolics
# @variables a b

# M = Diagonal([a, b])
# m = 5

# Mtrue = simplify.(expand.(tayl_exp_horner(M, m, 0)))
# Mps   = simplify.(expand.(expm_taylor(M, [], m, 0)))
# print("T₅(a) - T̃₅(a) = $(Mtrue[1,1] - Mps[1,1])\n")
# print("This is definitively not the zero polynomial. However, by switching from m = $m to m = $(m_array[3])...\n")

# m = m_array[3]
# Mtrue = simplify.(expand.(tayl_exp_horner(M, m, 0)))
# Mps   = simplify.(expand.(expm_taylor(M, [], m, 0)))
# print("Tₘ(a) - T̃ₘ(a) = $(Mtrue[1,1] - Mps[1,1])\n")
# print("The result is now correct.\n")

# print("""However, checking higher orders reveals that there are differences of negligible magnitude.
# I think it's due to the numerical evaluation of the symbolic expressions. I'm still a rookie on this symbolics stuff...\n""")


## First numerical check
print("We check our scheme `eval_pade!` (which in turn calls `expm_taylor` internally) "
* "against the Horner evaluation scheme (for the same Taylor polynomial of course), "
 * "using a small random matrix.\n")

m = m_array[6]  # i.e. m = 16
n = 5;
A = rand(n,n)
S = AandPowsStruct(A, true);
print("n = $n. m = $m. eltype(A) = $(eltype(A))\n")

Ytrue = tayl_exp_horner(A, m, s)    # Evaluate Tₘ(2^(-s)A) using the Horner scheme
Y     = eval_pade!(S, m, s);

@printf("|| Y - Y_true || / || Y_true || = %.6g\n", rel_err(Y,Ytrue))
@printf("|| exp(A) - Y || / || exp(A) || = %.6g,\t (m = %2.f)\n", rel_err(Y,exp(A)), m)


## Second numerical check
B = rand(BigFloat, n,n)
S = AandPowsStruct(B, true);
print("n = $n. m=$m. eltype(B) = $(eltype(B))\n")

# take Taylor poly computed with Horner in higher precision as reference
Y_ref = setprecision(2*precision(eltype(B))) do 
    tayl_exp_horner(B, m, s)    # this function is in MyHelper
end

Y_horner = tayl_exp_horner(B, m, s)
@printf("|| Y_ref - Y_horner || / || Y_ref || = %.6g\n", rel_err(Y_horner, Y_ref))
@printf("And indeed 2^(1-precision(BigFloat)) = eps(BigFloat) = %.4g\n", eps(BigFloat))
print("We're within a factor ≤ 10 from the unit roundoff. E ci mancherebbe altro! Nothing here surprises us.\n")

Y = eval_pade!(S, m, s)   # Paterson-Stockmeyer in standard BigFloat precision.
                          # calls `expm_taylor(S, m, s)`
@printf("|| Y - Y_ref || / || Y_ref || = %.6g\n", rel_err(Y, Y_ref) )
@printf("|| Y - Y_ref || / (1 + || Y || + || Y_ref ||) = %.6g\n", sym_err(Y, Y_ref) )
# Domanda 2: con cosa va confrontato? eps(BigFloat)? 2^(-precision(BigFloat))? 
#            Con n⋅ϵ ? O con ϵ / n ? (ϵ è uno dei due sopra)
# Domanda 3: Come "mettere tutto insieme" con la funzione `isapprox` (o ≈ insomma)?
#            Mi pare di capire che || Y - Y_ref || ≈ 0 sia l'approccio sbagliato (cfr. ?> ≈)
#            Però isapprox usa come tolleranza relativa √eps(T) dove T è il tipo e dipende dagli operandi
#            Sono incappato in questo : https://discourse.julialang.org/t/approximate-equality/8952
print("\n")


## Third numerical check
n = 50;
B = rand(BigFloat, n, n);
S = AandPowsStruct(B, true);
print("n = $n. m = $m. eltype(B) = $(eltype(B))\n")

Y_ref = setprecision(2*precision(eltype(B))) do 
    tayl_exp_horner(B, m, s)
end

Y_horner = tayl_exp_horner(B, m, s)
@printf("|| Y_ref - Y_horner || / || Y_ref || = %.6g\n", rel_err(Y_horner, Y_ref))

Y = eval_pade!(S, m, s)
@printf("|| Y - Y_ref || / || Y_ref || = %.6g\n", rel_err(Y, Y_ref))
@printf("|| Y - Y_ref || / (1 + || Y || + || Y_ref ||) = %.6g\n", sym_err(Y, Y_ref))
print("Y ≈ Y_ref up to machine precision = $(isapprox(Y, Y_ref, rtol=eps(BigFloat)))\n")
print("\n")

## Fourth numerical check
n = 20
B = rand(BigFloat, n, n);
B = (B + B')/2;
B -= (tr(B)/n) * I(n);
S = AandPowsStruct(B, true);
print("n = $n. m = $m. eltype(B) = $(eltype(B))\n")

# Use diagonalization as reference solution 
F = schur(B);
@printf("|| Q⋅T⋅Q' - B || / || B || = %.6g\n", rel_err(F.Z * F.T * F.Z', B))
Y_ref = F.Z * exp.(Diagonal(F.T)) * F.Z';

Y_horner = tayl_exp_horner(B, m, 0);
@printf("|| Y_ref - Y_horner || / || Y_ref || = %.6g\n", rel_err(Y_horner, Y_ref))

Y = eval_pade!(S, m, 0); # no scaling
@printf("|| Y - Y_ref || / || Y_ref || = %.6g\n", rel_err(Y, Y_ref))
@printf("|| Y - Y_ref || / (1 + || Y || + || Y_ref ||) = %.6g\n", sym_err(Y, Y_ref))
print("Y ≈ Y_ref up to machine precision = $(isapprox(Y, Y_ref, rtol=eps(BigFloat)))\n")
print("\n")
print("Keep in mind that this is not the full algorithm. "
 * "It may be that Taylor is just bad on this problem.\n")

print("What we really care about here (but we've tested multiple times):\n")
@printf("|| Y - Y_horner || / || Y_horner || = %.6g\n", rel_err(Y, Y_horner))

