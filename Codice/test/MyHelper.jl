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






# Non voglio averle in mezzo e le metto qui

# Given (ilo, ihi, iscale) returned by LAPACK.gebal!('B', A), apply same balancing to X
function _balance!(X, ilo, ihi, scale)
    println("Balancing!")
    n = size(X, 1)
    if ihi < n
        for j in (ihi + 1):n
            LinearAlgebra.rcswap!(j, Int(scale[j]), X)
        end
    end
    if ilo > 1
        for j in (ilo - 1):-1:1
            LinearAlgebra.rcswap!(j, Int(scale[j]), X)
        end
    end

    for j in ilo:ihi
        scj = scale[j]
        for i in 1:n
            X[j, i] /= scj
        end
        for i in 1:n
            X[i, j] *= scj
        end
    end
    return X
end

# Reverts the balancing performed by `LAPACK.gebal!`
function _unbalance!(X, ilo, ihi, scale)
    for j = ilo:ihi
        scj = scale[j]
        for i = 1:n
            X[j,i] *= scj
        end
        for i = 1:n
            X[i,j] /= scj
        end
    end

    if ilo > 1       # apply lower permutations in reverse order
        for j in (ilo-1):-1:1
            LinearAlgebra.rcswap!(j, Int(scale[j]), X)
        end
    end
    if ihi < n       # apply upper permutations in forward order
        for j in (ihi+1):n
            LinearAlgebra.rcswap!(j, Int(scale[j]), X)
        end
    end
end


# La copia di Base.exp!(A), in cui ho sostituito `LAPACK.gesv!` con `\`
function my_exp!(A::StridedMatrix{T}) where T<:LinearAlgebra.BlasFloat
    n = LinearAlgebra.checksquare(A)
    if isdiag(A)
        for i in diagind(A, IndexStyle(A))
            A[i] = exp(A[i])
        end
        return A
    elseif ishermitian(A)
        return copytri!(parent(exp(Hermitian(A))), 'U', true)
    end
    ilo, ihi, scale = LAPACK.gebal!('B', A)    # modifies A
    nA   = opnorm(A, 1)
    ## For sufficiently small nA, use lower order Padé-Approximations
    if (nA <= 2.1)
        if nA > 0.95
            C = T[17643225600.,8821612800.,2075673600.,302702400.,
                     30270240.,   2162160.,    110880.,     3960.,
                           90.,         1.]
        elseif nA > 0.25
            C = T[17297280.,8648640.,1995840.,277200.,
                     25200.,   1512.,     56.,     1.]
        elseif nA > 0.015
            C = T[30240.,15120.,3360.,
                    420.,   30.,   1.]
        else
            C = T[120.,60.,12.,1.]
        end
        A2 = A * A
        # Compute U and V: Even/odd terms in Padé numerator & denom
        # Expansion of k=1 in for loop
        P = A2
        U = similar(P)
        V = similar(P)
        for ind in CartesianIndices(P)
            U[ind] = C[4]*P[ind] + C[2]*I[ind]
            V[ind] = C[3]*P[ind] + C[1]*I[ind]
        end
        for k in 2:(div(length(C), 2) - 1)
            P *= A2
            for ind in eachindex(P, U, V)
                U[ind] += C[2k + 2] * P[ind]
                V[ind] += C[2k + 1] * P[ind]
            end
        end

        # U = A * U, but we overwrite P to avoid an allocation
        mul!(P, A, U)
        # P may be seen as an alias for U in the following code

        # Padé approximant:  (V-U)\(V+U)
        VminU, VplusU = V, U # Reuse already allocated arrays
        for ind in eachindex(V, U)
            vi, ui = V[ind], P[ind]
            VminU[ind] = vi - ui
            VplusU[ind] = vi + ui
        end
        #X = LAPACK.gesv!(VminU, VplusU)[1]
        X = VminU \ VplusU
    else
        s  = log2(nA/5.4)               # power of 2 later reversed by squaring
        if s > 0
            si = ceil(Int,s)
            twopowsi = convert(T,2^si)
            for ind in eachindex(A)
                A[ind] /= twopowsi
            end
        end
        CC = T[64764752532480000.,32382376266240000.,7771770303897600.,
                1187353796428800.,  129060195264000.,  10559470521600.,
                    670442572800.,      33522128640.,      1323241920.,
                        40840800.,           960960.,           16380.,
                             182.,                1.]
        A2 = A * A
        A4 = A2 * A2
        A6 = A2 * A4
        tmp1, tmp2 = similar(A6), similar(A6)

        # Allocation economical version of:
        # U  = A * (A6 * (CC[14].*A6 .+ CC[12].*A4 .+ CC[10].*A2) .+
        #           CC[8].*A6 .+ CC[6].*A4 .+ CC[4]*A2+CC[2]*I)
        for ind in eachindex(tmp1)
            tmp1[ind] = CC[14]*A6[ind] + CC[12]*A4[ind] + CC[10]*A2[ind]
            tmp2[ind] = CC[8]*A6[ind] + CC[6]*A4[ind] + CC[4]*A2[ind]
        end
        mul!(tmp2, true,CC[2]*I, true, true) # tmp2 .+= CC[2]*I
        U = mul!(tmp2, A6, tmp1, true, true)
        U, tmp1 = mul!(tmp1, A, U), A # U = A * U0

        # Allocation economical version of:
        # V  = A6 * (CC[13].*A6 .+ CC[11].*A4 .+ CC[9].*A2) .+
        #           CC[7].*A6 .+ CC[5].*A4 .+ CC[3]*A2 .+ CC[1]*I
        for ind in eachindex(tmp1)
            tmp1[ind] = CC[13]*A6[ind] + CC[11]*A4[ind] + CC[9]*A2[ind]
            tmp2[ind] = CC[7]*A6[ind] + CC[5]*A4[ind] + CC[3]*A2[ind]
        end
        mul!(tmp2, true, CC[1]*I, true, true) # tmp2 .+= CC[1]*I
        V = mul!(tmp2, A6, tmp1, true, true)

        for ind in eachindex(tmp1)
            tmp1[ind] = V[ind] + U[ind]
            tmp2[ind] = V[ind] - U[ind] # tmp2 already contained V but this seems more readable
        end
        #X = LAPACK.gesv!(tmp2, tmp1)[1] # X now contains r_13 in Higham 2008
        X = tmp2 \ tmp1

        if s > 0
            # Repeated squaring to compute X = r_13^(2^si)
            for t=1:si
                mul!(tmp2, X, X)
                X, tmp2 = tmp2, X
            end
        end
    end

    # Undo the balancing
    
    _unbalance!(X, ilo, ihi, scale)
    return X
end


export _balance!, _unbalance!, my_exp!




end #module
