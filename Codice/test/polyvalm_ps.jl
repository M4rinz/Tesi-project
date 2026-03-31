using LinearAlgebra, Random
using Revise

Revise.includet(joinpath(@__DIR__,"..","src","modules","MyMpExponential.jl"))
Revise.includet(joinpath(@__DIR__,"..","src","modules","MyHelper.jl"))
using .MyMpExponential, .MyHelper

Random.seed!(42)


## Define parameters and useful stuff
s = 0;          # For now, no scaling

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

# small test
γ = 2;
A = γ*Diagonal(ones(3))
β_vec = [1, 0, 5, 3];
Y = horner(A, β_vec, 0) # should be 45 * I(3)

Y[1,1] == dot(β_vec, 2 .^ (0:length(β_vec)-1)') || error(lazy"Il test con matrice diagonale non torna")

## first test
Apows = [I(size(A,1)), A]
Y_ps = polyvalm_ps!(Apows, s, β_vec);

Y_ps == Y || error("Il risultato di `polyvalm_ps!` non coincide con quello di `horner`\n")


## Check symbolically
print("""A symbolic check is not possible, because of the type constraint in the
function signature (which are there because of the `precision` stuff).
""")


## First numerical check
# We construct a "tame" matrix for whose the Taylor approx. of the matrix 
# exponential works well, and compare our methods to evaluate polynomials

n = 5;
A  = rand(n, n);
A  = (A + A')/2;
A -= (tr(A)/n) * I(n)   # shift 

m = 36  # an "optimal" Taylor degree 
tayl_coeffs = [1/factorial(big(k)) for k=0:m]

Y_true = exp(A)

Y_t = polyvalm_tay_exp(A, m, 0)
@printf("|| Y_t - exp(A) || / || exp(A) || = %.4g", rel_err(Y_t, Y_true))
print("Y_t ≈ exp(A) is $(Y_t ≈ Y_true)\n")

Y_h = horner(A, tayl_coeffs, 0)
@printf("|| Y_h - exp(A) || / || exp(A) || = %.4g", rel_err(Y_h, Y_true))
print("Y_h ≈ exp(A) is $(Y_h ≈ Y_true)\n")

Apows = [I(n), A];

Y_ps = polyvalm_ps!(Apows, 0, tayl_coeffs);
@printf("|| Y_ps - exp(A) || / || exp(A) || = %.4g", rel_err(Y_ps, Y_true))
print("Y_ps ≈ exp(A) is $(Y_ps ≈ Y_true)\n")

Y_ps_f = polyvalm_ps!(Apows, 0, tayl_coeffs, outputclass=eltype(A));
@printf("|| Y_ps_f - exp(A) || / || exp(A) || = %.4g", rel_err(Y_ps_f, Y_true))
print("Y_ps_f ≈ exp(A) is $(Y_ps_f ≈ Y_true)\n")

"""Note that the coefficient of the polynomial are cast to `outputclass` when 
performing linear combinations.
This is because 
- If `outputclass` == BigFloat, then `BigFloat(β)` returns `β` at the precision 
  specified in the `do` block. 
    - If `β_vec` was computed at a lower precision than the one inside the block, 
      then there's no difference, basically
    - Otherwise, there is a difference. Early benchmarks show that the cost of 
      products between `BigFloat`s` depend on the precision
- If `outputclass` == Float64, then `outputclass(β)` "truncates" `β` to double precision.
  It is safe to assume that the coefficients in `β_vec` are computed at arbitrary precision 
  (i.e. are `BigFloat`s): was it not the case, the following considerations would 
  be worthless, as `outputclass(β)` would do nothing. 
    - `mpowers` has `Matrix{Float64}` elements as well. In the `do` blocks, computations 
      are in double precision (much cheaper)
    - From an accuracy standpoint, it still makes sense, because - since `M`∈`mpowers` 
      is a `Matrix{Float64}` - the accuracy of the computation `β * M` is limited by that of `M` anyway.
      There iss (I'd say) no gain in keeping more significant digits in `β` 

Let us turn to `A`
- If `A` is computed at a higher precision than the working precision 
    - Powers of `A` are still computed accurately
    - `mpowers` has the powers of `A` computed at their original precision 
      (i.e. the precision value in the global scope of the function). This is 
      because the `convert` command creates an alias.
    - In the computation of `B`, powers of `A` are used to full accuracy 
      (precisely, the `mpowers`). However, the `β` is the "limiting factor".
        - The usefulness of setting the precision and "truncating" `β` while leaving 
          `mpowers` untouched is questionable
        - Conversely, bringing `mpowers` to the precision of the `do` block 
          tantamounts to a handful of memory operations. Is it worth the savings in 
          the computational complexity of `β * M`?
      I wouldn't say that `B` is "fully" computed at a higher precision than the working precision.
        - Firstly, it depends on how `1.2 * precision` compares to the precision at which 
          `A` and its powers are computed / stored.
        - Then, again, we're not touching `mpowers`, as it has `BigFloat` matrices already
      
- If `A` has `BigFloat` elements, but computed at the working precision 
    - Not much changes. The `setprecision` thing only touches `β`. I'd say the accuracy
      in the computation of `B` is the same as that of `A` and its powers.
    - by doing `outputclass(β)` we're saving some computation (at the expense of a 
      cast operation, a memory operation I would say) without downsides in terms of 
      accuracy of the computed result

- if `A` is a `Matrix{Float64}`
  - 
"""


## Second numerical experiment
# A small toy example, with companion matrices

n = 50;
v = rand(BigFloat, n);

@printf("max(|v|) / min(|v|) = %.6g", maximum(abs.(v)) / minimum(abs.(v)))

C = zeros(eltype(v), n,n);
C[2:end, 1:end-1] .= I(n-1);
C[:, end] .= -v;

Cpows = [I(n), C];
Y_ps = polyvalm_ps!(Cpows, 0, [v..., 1]);
Y_h  = horner(C, [v..., 1], 0);
@printf("The exact result is Y = 0, as Hamilton-Cayley teaches us.\t eps(BigFloat) = %.4g", eps(BigFloat))
@printf("|| Y_ps || = %.4g", norm(Y_ps))
@printf("|| Y_h ||  = %.4g", norm(Y_h))

Y_ps_f = polyvalm_ps!([I(n), Float64.(C)], 0, [Float64.(v)..., 1], outputclass=Float64);
Y_h_f = horner(Float64.(C), [Float64.(v)..., 1], 0);
@printf("Now the algorithms are ran in double precision.\t eps(Float64) = %.4g", eps(Float64))
@printf("|| Y_ps_f || = %.4g", norm(Y_ps_f))
@printf("|| Y_h_f ||  = %.4g", norm(Y_h_f))
print("In double precision our algorithm doesn't look bad.\n")

Y_ps_l = setprecision(53) do 
  Cpows_l = [I(n), BigFloat.(C)];
  Y_ps_l = polyvalm_ps!(Cpows_l, 0, [v..., 1]);
end;
@printf("|| Y_ps_l || = %.4g", norm(Y_ps_l))
print("More or less the same that we got from the algorithm in double precision\n")
