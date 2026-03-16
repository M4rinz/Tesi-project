using LinearAlgebra

print("Please take a look at `help?> precision`\n")
print("The precision of Float64 is $(precision(Float64))\n")
default_BigFloat_precision = precision(BigFloat);
print("The precision of BigFloat is $(default_BigFloat_precision)\n")
print("But we can change it with the `setprecision(BigFloat, n)` command, where n::Int is the new precision\n")

setprecision(BigFloat, 512);
print("Now, precision(BigFloat) = $(precision(BigFloat))\n")
setprecision(BigFloat, default_BigFloat_precision);

# How one constructs the BigFloats makes a huge different
print("BigFloat(0.1)   = $(BigFloat(0.1))\n")       # a double is promoted
print("BigFloat(\"0.1\") = $(BigFloat("0.1"))\n")   # the closest BigFloat to 0.1

print("BigFloat(1/3) = $(BigFloat(1/3))\n")     # again, a double is "promoted"
print("BigFloat(1) / BigFloat(3) = $(BigFloat(1)/BigFloat(3)) \n")  # arithmetic takes place between BigFloats
print("big(1) / 3 = $(big(1)/3)\n")

# big is not quite the same as the BigFloat / BigInt constructor
print("typeof(big(1.)) = $(typeof(big(1.)))\t typeof(BigFloat(1.)) = $(typeof(BigFloat(1.)))\n")
print("big(1.0im) = $(big(1.0im))\t big(typeof(1.0im)) = $(big(typeof(1.0im)))\n")
print("Please take a look at `help?> big`\n")



## Matrix Multiplication
n = 10;
A = rand(BigFloat, n, n);

@which A^8
# This function ends up calling ^(A, 8). This `Val` is a bit mysterious...
# ... it looks like `Val{p}` is sort of a workaround to pass a value as a type. 
# To specialize a function to a particular value

@which A^(2^3)
# Turns out that `power_by_squaring(A, p)` is called

@edit Base.power_by_squaring(A, 8)
# Basically the binary powering algorithm in [Higham, Alg. 4.1]

## matrix addition
B = rand(BigFloat, size(A)...);
@which A + Base
# the function broadcast_preserving_zero_d(+, A, B) is called

@which Base.Broadcast.broadcast_preserving_zero_d(+, A, B)
# Qui è un po' un casino, si usa il broadcast. Confido che faccia la cosa giusta


## Some proper Linear Algebra
F = lu(A);
print("typeof(F) = $(typeof(F)),\t where F = lu(A)\n")
print("|| L*U - A[p,:] || = $(norm(F.L * F.U - A[F.p,:]))\n")
# The LU factorization is available
# NOTE (for the future) on linear systems solve: iterative refinement.
# A Julia library is https://github.com/ctkelley/MultiPrecisionArrays.jl


try
    λs = eigvals(A)
catch
    print("`eigvals(A)` is not available on matrix `A` of BigFloats\n")
end

try
    F = eigen(A)
catch
    print("`eigen(A)` is not available on matrix `A` of BigFloats\n")
end
# no eigenvalues, no eigendecomposition

try
    F = schur(A)
catch
    print("`schur(A)` is not available on matrix `A` of BigFloats\n")
end
# No Schur form
# NOTE: One can use the Generic Linear Algebra (https://julialinearalgebra.github.io/GenericLinearAlgebra.jl/stable/),
# that implements some NLA algorithms for higher precision
# If this wasn't enough, one can use Arnoldi for eigenvalues using the ArnoldiMethod library
# (https://julialinearalgebra.github.io/ArnoldiMethod.jl/stable/)
#
# Of course, there's the question of how fast these methods are. 