module MyHelper

using LinearAlgebra, Random


############ Valutazioni dell'errore ############

"""
    rel_err(approx, exact)

Returns the relative error between `approx` (the approximation)
and `exact` (the exact answer), using the formula
``norm(approx - exact) / norm(exact)``
"""
function rel_err(approx, exact)
    norm(approx - exact) / norm(exact)
end

"""
    sym_err(A, B)

returns `norm(A - B) / (1 + norm(A) + norm(B))`, a measure of error
between ``A`` and ``B``, that is 
- Well defined when the quantity taken as reference is ``0``
- Symmetric in ``A`` and ``B``
"""
function sym_err(A, B)
    norm(A - B) / (1 + norm(A) + norm(B))
end

export rel_err, sym_err



############ Manipolazione di files ############

# Make sure that the first line of the file is CSV_header. If not, write it
function ensure_csv_header(csvfile, header::Vector{String})
    header_line = join(header, ",")

    if !isfile(csvfile)
        open(csvfile, "w") do io
            println(io, header_line)
        end
        return
    end

    contents = read(csvfile, String)
    if isempty(contents)
        open(csvfile, "w") do io
            println(io, header_line)
        end
        return
    end

    lines = readlines(IOBuffer(contents))

    first_row = strip.(split(first(lines), ','))
    if first_row != header || !endswith(contents, "\n")
        open(csvfile, "w") do io
            println(io, header_line)
            for line in lines
                println(io, line)
            end
        end
    end
end


export ensure_csv_header










############ Costruzione di esempi ############


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

export gebal_example









"""
    tayl_exp_horner(X, m, s)

Computes ``T_m(2^{-s}X)``, the Taylor polynomial of degree `m`
in `2^(-s)X`, using the Horner rule.
"""
function tayl_exp_horner(X, m, s)
    n = LinearAlgebra.checksquare(X)
    scaling = 2^s
    Y = X / (scaling*m)
    for j = 1:m-1
        Y = (Y + I(n)) * (X / (scaling * (m - j)))
    end
    return Y + I(n)
end


export tayl_exp_horner






""" 
Given a linear operator ``\\mathbb{C}^{n\\times n}\\to\\mathbb{C}^{n\\times n}`` 
(typically, a fréchet differential ``L_f(A)``), constructs the matrix 'K'
such that 'linear_operator[E] = K * y' where ``y=\\operatorname{Vec}(E)``
"""
function construct_full_jacobian(linear_operator, n, T)
    K = Matrix{T}(undef, n^2, n^2)    # il tipo lo prendiamo in input eh
    e_i = zeros(n); e_i[1] = 1
    e_j = zeros(n); e_j[1] = 1
    for j = 1:n#, i = 1:n
        for i = 1:n
            column = linear_operator(e_i * e_j')
            K[:, (j-1)*n + i] = vec(column)
            
            circshift!(e_i, 1)
        end
        circshift!(e_j, 1)
    end
    return K
end


function cond_exp_exact(A)
    n = LinearAlgebra.checksquare(A)
    T = eltype(A)
    T_low = T <: Complex ? ComplexF64 : Float64

    A_low = convert(AbstractMatrix{T_low}, A)
    Y_true, exp_pullback = ChainRules.rrule(exp, A_low)

    K = construct_full_jacobian(x -> exp_pullback(x)[2], n, T_low)

    #κ_exp(A) = (‖K‖₂⋅‖A‖_F) / ‖exp(A)‖_F
    opnorm(K, 2) * norm(A) / norm(Y_true)
end


export construct_full_jacobian, cond_exp_exact


end #module
