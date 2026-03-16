module MyHelper

using LinearAlgebra, Random

function rel_err(approx, exact)
    """
    Returns the relative error between `approx` (the approximation)
    and `exact` (the exact answer), using the formula
    ``norm(approx - exact) / norm(exact)``
    """

    norm(approx - exact) / norm(exact)
end

# courtesy of ChatGPT (under my detailed instructions)
"""
    gebal_example(n; k1=2, k2=2, do_permutation=true, seed=42)

Generate a test matrix suitable for testing the `LAPACK.gebal!` function.

# Arguments
- `n::Int`: Size of the output matrix (n × n)
- `k1::Int`: Size of the first triangular block (default: 2)
- `k2::Int`: Size of the second triangular block (default: 2)
- `do_permutation::Bool`: Whether to apply random row/column permutation (default: true)
- `seed::Int`: Random seed for reproducibility (default: 42)

# Returns
- `A_unbalanced::Matrix`: The scaled and (optionally) permuted matrix
- `A_bal::Matrix`: The original balanced matrix

# Description
Constructs a block matrix with triangular blocks T1 and T2, a dense block B, and coupling blocks.
The B block is then scaled by exponential factors to introduce numerical imbalance, simulating 
ill-conditioned matrices useful for testing equilibration algorithms.
The matrix is then (optionally) permuted, destroying the block structure.
"""
function gebal_example(n;
    k1=2, k2=2,
    do_permutation=true,
    seed=42)

    Random.seed!(seed)

    @assert k1 + k2 < n
    m = n - k1 - k2   # size of B

    # -----------------------------
    # 1) Construct balanced matrix
    # -----------------------------

    T1 = triu(randn(k1,k1))
    T2 = triu(randn(k2,k2))
    B  = randn(m,m)

    X = randn(k1,m)
    Y = randn(k1,k2)
    Z = randn(m,k2)

    A_bal = zeros(n,n)

    A_bal[1:k1,1:k1] = T1
    A_bal[1:k1,k1+1:k1+m] = X
    A_bal[1:k1,k1+m+1:end] = Y

    A_bal[k1+1:k1+m,k1+1:k1+m] = B
    A_bal[k1+1:k1+m,k1+m+1:end] = Z

    A_bal[k1+m+1:end,k1+m+1:end] = T2

    # -----------------------------------
    # 2) Apply scaling ONLY on B indices
    # -----------------------------------

    # Exponential but shuffled scaling for B block
    scales_B = 2.0 .^ randperm(m)

    D = Diagonal(vcat(
        ones(k1),
        scales_B,
        ones(k2)
    ))

    A_unbalanced = D \ A_bal * D

    # -----------------------------------
    # 3) Optional permutation
    # -----------------------------------

    if do_permutation
        p = randperm(n)
        A_unbalanced = A_unbalanced[p,p]
    end

    return A_unbalanced, A_bal
end

export rel_err, gebal_example




end #module
