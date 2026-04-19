using LinearAlgebra, GenericSchur, Polynomials
using Random, Revise, Printf

Random.seed!(42)

Revise.includet(joinpath(@__DIR__,"..","..","..","src","modules","MyHelper.jl"))
using .MyHelper


## Define parameters and useful stuff
"""
    exact_eigvals(n::Integer; T::Type=BigFloat) -> Vector{T}

Compute the exact eigenvalues of the tridiagonal matrix of size n×n
that discretizes the Laplacian operator.

The eigenvalues are calculated using the formula: `4 * sin²(π * x)` where 
`x = min(k, n+1-k) / (2*(n+1))` for each index k from 1 to n.

# Arguments
- `n::Integer`: The size of the matrix
- `T::Type=BigFloat` (optional): The numeric type for computation and return values

# Returns
- `Vector{T}`: A vector of n eigenvalues of type T
"""
function exact_eigvals_laplacian(n::Integer; T::Type=BigFloat)::Vector{T}
    nT = T(n)
    vals = Vector{T}(undef, n)

    for k in 1:n
        x = T(k) / (2*(nT + 1))
        vals[k] = 4 * sinpi(x)^2
    end

    return vals
end

function exact_eigvals_blkdiag(a, b; T::Type=BigFloat)::Vector{complex(T)}
    n = length(a)
    length(b) == n || error("a and b must have the same length.\n")
    v = Vector{complex(T)}(undef, 2n)
    for i=1:n
        v[2i]   = a[i] - b[i]im;
        v[2i-1] = a[i] + b[i]im;
    end
    return sort(v, by=abs2)
end

function test_schur(
    A::AbstractMatrix{T};
    exact_eigvals_func=exact_eigvals_laplacian,
    has_real_eigs::Bool=true
) where {T}
    n = LinearAlgebra.checksquare(A);
    if A isa SymTridiagonal 
        type = "SymTridiagonal"
    elseif A isa UpperHessenberg
        type = "Hessemberg"
    else 
        type = "full"
    end

    print("Testing $(type) A (size = $n) with $T elements.")
    Tf = float(T)
    @printf("\tprecision(%s) = %s, eps(%s) = %.4g\n", Tf, precision(Tf), Tf, eps(Tf))
    ref_val = n * eps(Tf)
    @printf("Reference value n * eps = %.6g\n", ref_val)

    F = schur(A);
    @printf("\t|| Q⋅T⋅Q' - A || / || A || = %.6g\t(%1.2g times the reference value)\n", rel_err(F.Z * F.T * F.Z', A), rel_err(F.Z * F.T * F.Z', A)/ref_val)
    @printf("\t|| Q'⋅Q - I || = %.6g\t\t\t(%1.2g times the reference value)\n", norm(F.Z*F.Z' - I(n)), norm(F.Z*F.Z' - I(n))/ref_val)

    λs = setprecision(2*precision(BigFloat)) do
        λs = exact_eigvals_func(n);
        by_func = has_real_eigs ? identity : abs2
        sort(λs, by=by_func);  
    end
    if has_real_eigs 
        truths_real = imag.(F.values) .≈ 0
        all(truths_real) || error("Not all computed eigenvalues are real (up to machine precision)!\n")
        λs_hat = sort(real.(F.values))
    else 
        λs_hat = sort(F.values, by=abs2)
    end

    truths_approx     = λs .≈ λs_hat
    truths_approx_eps = isapprox.(λs, λs_hat, rtol=eps(Tf))
    print("\tλₖ ≈ ̂λₖ (half significant digits) ∀k∈[n] is $(all(truths_approx))\n")
    print("\tλₖ ≈ ̂λₖ (up to machine precision) ∀k∈[n] is $(all(truths_approx_eps))\n")
    @printf("\tMaximum relative error in λs = %.6g\t(%1.2g times the reference value)\n", maximum(rel_err.(λs_hat, λs)), maximum(rel_err.(λs_hat, λs))/ref_val)
    print("\n")
    return F;
end


## first test: discretization of the Laplacian operator
n = 50;

print("We test the Schur decomposition of the discretization of the Laplace operator: L = tridiag(-1,2,1) of size $n.\n"
* "Being SPD, the eigenvalues are real and positive. The Schur form (be it real or complex) is diagonal. "
* "Moreover, there's a known exact formula for its eigenvalues.\n")

# define matrices. Nomenclature: L_{type}_{full/SymTri}
L_d_f = diagm(0 => 2*ones(n), 1 => -1*ones(n-1), -1 => -1*ones(n-1));       # double, full
L_b_f = diagm(0 => big(2)*ones(n), 1 => -1*ones(n-1), -1 => -1*ones(n-1));  # BigFloat, full
L_d_t = SymTridiagonal(2*ones(n), -1*ones(n-1));                            # double, SymTri
L_b_t = SymTridiagonal(big(2)*ones(n), -1*ones(n-1));                       # BigFloat, SymTri

@which schur(L_d_t)

print("Testing the matrices with Float64 elements\n")
_ = test_schur(L_d_f);
_ = test_schur(L_d_t);

print("Testing the matrices with BigFloat elements\n")
_ = test_schur(L_b_f);
_ = test_schur(L_b_t);


## Second numerical test: companion matrix of the Wilkinson polynomial
n = 20;

print("We test the companion matrix of the Wilkinson polynomial of degree $n: "
* "the polynomial whose roots are the integers 1,…,$n.\n"
* "The coefficients of the polynomial and, hence, entries of the companion matrix vary wildly " 
* "in magnitude and cannot be stored exactly in double precision.\n")

# get polynomial coefficients
v = coeffs(fromroots(BigInt.(1:n)));

# verify result
V = BigInt.(1:n) .^ (0:n)';
evaluations = V * v;
all(evaluations .≈ zeros(n)) || error("The evaluation of the polynomial on one of its roots is not (an approximate) zero.\n");

# construct companion matrix
C = zeros(eltype(v), n,n);
C[2:end, 1:end-1] .= I(n-1);
C[:,end] .= -v[1:end-1];

# function that returns exact eigenvalues
function exact_eigvals_Wilkinson(
    n::Integer; ::Type=BigInt)::Vector{T}
    return Vector{T}(1:n)
end

print("Testing the matrices with BigFloat elements.\n")
F = test_schur(C, exact_eigvals_func=exact_eigvals_Wilkinson);
_ = test_schur(UpperHessenberg(C), exact_eigvals_func=exact_eigvals_Wilkinson);

print("How does our Schur form compare with the the rootfinding algorithm of Polynomials.jl?")
schur_λs_rel_errs = rel_err.(real.(F.values), Vector{eltype(C)}(1:n));
roots_λs_rel_errs = rel_err.(real.(roots(Polynomial(v))) , Vector{eltype(C)}(1:n));
@printf("mean accuracy (Schur) = %.6g\n", sum(schur_λs_rel_errs)/n)
@printf("mean accuracy (roots) = %.6g\n", sum(roots_λs_rel_errs)/n)


## Third numerical test: real Schur form of a random matrix
n = 50; 

print("We test a random matrix of size $n.\n"
* " Hopefully the matrix has complex eigenvalues, so that the computed Schur form is "
* " a real Schur form.")

A = rand(BigFloat, n,n);

print("Testing the real Schur form\n")
F = schur(A);
@printf("|| Q⋅T⋅Q' - A || / || A || = %.6g\n", rel_err(F.Z * F.T * F.Z', A))
@printf("|| Q'⋅Q - I || = %.6g\n", norm(F.Z*F.Z' - I(n)))

print("Testing the complex Schur form\n")
Fc = schur(complex(A));
@printf("|| Q⋅T⋅Q' - A || / || A || = %.6g\n", rel_err(Fc.Z * Fc.T * Fc.Z', complex(A)))
@printf("|| Q'⋅Q - I || = %.6g\n", norm(Fc.Z*Fc.Z' - I(n)))


## Fourth numerical test: destroying a block diagonal matrix
n = 20;

print("""We test a random matrix of size 2n, with n=$n, constructed in the following way:
    - pick a₁,…,aₙ and b₁,…,bₙ
    - costruct Â = blkdiag([a₁ -b₁; b₁ a₁],…,[aₙ -bₙ; bₙ aₙ])
    - destroy the block diagonal structure of ̂A with a similarity transformation, producing A = MÂM⁻¹
""")

a, b = rand(n), rand(n);

Â = zeros(BigFloat, 2n,2n);
for i=1:n
    Â[2i-1:2i, 2i-1:2i] .= [a[i] -b[i]; b[i] a[i]];
end

# Let's begin by applying random unitary similarity transformations
Q, _ = qr(randn(BigFloat, 2n,2n));
A = Q * Â * Q';

#λs_blk_orig = exact_eigvals_blkdiag(a, b);

F = test_schur(A, 
               exact_eigvals_func=n -> exact_eigvals_blkdiag(a, b),
               has_real_eigs=false);

# another similarity transform
M = rand(BigFloat, 2n,2n);
A = (M * Â) / M; 

F = test_schur(A,
               exact_eigvals_func=n -> exact_eigvals_blkdiag(a, b),
               has_real_eigs=false);


## Fifth numerical test: companion matrix of a polynomial with complex conjugate roots
n = 20;

print("""We test the companion matrix of a polynomial which we purposely craft:
    - pick a₁,…,aₙ and b₁,…,bₙ (at random)
    - get the coefficients of the polynomial whose roots are aₖ ± ibₖ
    - create the companion matrix of such polynomial
""")

a = 1e-4rand(BigFloat, n);
b = 1 .+ rand(BigFloat, n);

# construct eigenvalues
λs_orig = exact_eigvals_blkdiag(a, b);

v = coeffs(fromroots(λs_orig));

# verify result
V = λs_orig .^ (0:2n)';
evaluations = V * v;
all(evaluations .≈ zeros(2n)) || error("The evaluation of the polynomial on one of its roots is not (an approximate) zero.\n");
print("It may be that the rootfinding algorithm is not so accurate.\n")

# construct companion matrix
C = zeros(eltype(v), 2n,2n);
C[2:end, 1:end-1] .= I(2*n-1);
C[:,end] .= -v[1:end-1];

F = test_schur(C, 
               exact_eigvals_func=n -> exact_eigvals_blkdiag(a, b),
               has_real_eigs=false);


## Sixth numerical test: companion matrix of a polynomial with real and complex roots
n = 6;

print("This test is similar to the previous one, but we impose that (hopefully) about half of the bₖs (chosen at random) are 0.\n")

a = 1e-4rand(BigFloat, n);
b = 1 .+ 1e-1rand(BigFloat, n) .|> x -> rand()>=0.5 ? x : 0;
is_b_zero = b .≈ 0 
print("$(sum(is_b_zero)) imaginary parts are zero. "
 * "This should yield $(2*sum(is_b_zero)) real eigenvalues.\n")

# construct eigenvalues
λs_orig = exact_eigvals_blkdiag(a, b);

v = coeffs(fromroots(λs_orig));

# verify result
V = λs_orig .^ (0:2n)';
evaluations = V * v;
# check disabled because most probably the rootfinding alg is not very accurate
#all(evaluations .≈ zeros(2n)) || error("The evaluation of the polynomial on one of its roots is not (an approximate) zero.\n");
#print("It may be that the rootfinding algorithm is not so accurate.\n")

# construct companion matrix
C = zeros(eltype(v), 2n,2n);
C[2:end, 1:end-1] .= I(2*n-1);
C[:,end] .= -v[1:end-1];

F = test_schur(C, 
               exact_eigvals_func=n -> exact_eigvals_blkdiag(a, b),
               has_real_eigs=false);

is_subdiag_zero = diag(F.T, -1) .≈ 0
num_1x1_blocks, num_2x2_blocks = 0, 0
i = 1
while i <= length(is_subdiag_zero)
    if is_subdiag_zero[i]
        num_1x1_blocks += 1
        i += 1
    else
        num_2x2_blocks += 1
        i += 2
    end
end;
@printf("Number of 1×1 blocks: %d\tNumber of 2×2 blocks: %d\n", num_1x1_blocks, num_2x2_blocks)

num_1x1_blocks == 2*sum(is_b_zero) || error("The number of computed real eigenvalues doesn't match with how many there are.\n")