using LinearAlgebra, Random

include(joinpath(@__DIR__,"..","src","modules","MyMpExponential.jl"))
include(joinpath(@__DIR__,"..","src","modules","MyHelper.jl"))
using .MyMpExponential, .MyHelper

Random.seed!(42)

## Define parameters and useful stuff
s = 4;          # For example...
m_array = [floor(Int64, (i+2)^2/4) for i=1:6]   # Optimal degrees for Taylor

function tayl_exp_horner(X, m, s)
    n = LinearAlgebra.checksquare(X)
    scaling = 2^s
    Y = X / (scaling*m)
    for j = 1:m-1
        Y = (Y + I(n)) * (X / (scaling * (m - j)))
    end
    return Y + I(n)
end


## Check symbolically
using Symbolics
@variables a b

M = Diagonal([a, b])
m = 5

Mtrue = simplify.(expand.(tayl_exp_horner(M, m, 0)))
Mps   = simplify.(expand.(polyvalm_tay_exp(M, m, 0)))
print("T₅(a) - T̃₅(a) = $(Mtrue[1,1] - Mps[1,1])\n")
print("This is definitively not the zero polynomial. However, by switching from m = $m to m = $(m_array[3])...\n")

m = m_array[3]
Mtrue = simplify.(expand.(tayl_exp_horner(M, m, 0)))
Mps   = simplify.(expand.(polyvalm_tay_exp(M, m, 0)))
print("Tₘ(a) - T̃ₘ(a) = $(Mtrue[1,1] - Mps[1,1])\n")
print("The result is now correct.\n")

print("""However, checking higher orders reveals that there are differences of negligible magnitude.
I think it's due to the numerical evaluation of the symbolic expressions. I'm still a rookie on this symbolics stuff...\n""")


## First numerical check
m = m_array[6]  # i.e. m = 16
n = 5;
A = rand(n,n)
print("n = $n. m=$m. eltype(A) = $(eltype(A))\n")

Ytrue = tayl_exp_horner(A, m, s)
Y = polyvalm_tay_exp(A, m, s)

print("|| Y - Y_true || / || Y_true || = $(rel_err(Y,Ytrue))\n")
print("|| exp(A) - Y || / || exp(A) || = $(rel_err(exp(A),Y))\t (m = $m)\n")
print()


## Second numerical check
B = rand(BigFloat, n, n)
print("n = $n. m=$m. eltype(B) = $(eltype(B))\n")

# take Taylor poly computed with Horner in higher precision as reference
Y_ref = setprecision(2*precision(eltype(B))) do 
    tayl_exp_horner(B, m, s)
end

Y_horner = tayl_exp_horner(B, m, s)
print("|| Y_ref - Y_horner || / || Y_ref || = $(rel_err(Y_horner, Y_ref))\n")
print("And indeed 2^(1-precision(BigFloat)) = eps(BigFloat) = $(eps(BigFloat))\n")
print("We're within a factor ≤ 10 from the unit roundoff. E ci mancherebbe altro! Nothing here surprises us.\n")

Y = polyvalm_tay_exp(B, m, s)   # Paterson-Stockmeyer in standard BigFloat precision
print("|| Y - Y_ref || / || Y_ref || = $(rel_err(Y, Y_ref))\n")
print("|| Y - Y_ref || / (1 + || Y || + || Y_ref ||) = $(norm(Y - Y_ref) / (1 + norm(Y) + norm(Y_ref))))\n")
# Domanda 1: qual è più giusto? (Non tra i due, in generale)
# Domanda 2: con cosa va confrontato? eps(BigFloat)? 2^(-precision(BigFloat))? 
#            Con n * ϵ ? O ϵ / n ? (ϵ è uno dei due sopra)
# Domanda 3: Come "mettere tutto insieme" con la funzione `isapprox` (o ≈ insomma)?
#            Mi pare di capire che || Y - Y_ref || ≈ 0 sia l'approccio sbagliato (cfr. ?> ≈)
#            Però isapprox usa come tolleranza relativa √eps(T) dove T è il tipo e dipende dagli operandi
#            Sono incappato in questo : https://discourse.julialang.org/t/approximate-equality/8952
print()


## Third numerical check
n = 50
B = rand(BigFloat, n, n)
print("n = $n. m=$m. eltype(B) = $(eltype(B))\n")

Y_ref = setprecision(2*precision(eltype(B))) do 
    tayl_exp_horner(B, m, s)
end

Y_horner = tayl_exp_horner(B, m, s)
print("|| Y_ref - Y_horner || / || Y_ref || = $(rel_err(Y_horner, Y_ref))\n")

Y = polyvalm_tay_exp(B, m, s)
print("|| Y - Y_ref || / || Y_ref || = $(rel_err(Y, Y_ref))\n")
print("|| Y - Y_ref || / (1 + || Y || + || Y_ref ||) = $(norm(Y - Y_ref) / (1 + norm(Y) + norm(Y_ref))))\n")
print("Y ≈ Y_ref up to machine precision = $(isapprox(Y, Y_ref, rtol=eps(BigFloat)))\n")
print()