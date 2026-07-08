module MyBaseExponential

using LinearAlgebra

include("MyHelper.jl")  # very basic helper functions
using .MyHelper


# Helpers tailored for matrix exponentials

# Balancing functions

"""
    Given (ilo, ihi, iscale) returned by LAPACK.gebal!('B', A), apply same balancing to X
"""
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

"""
    Reverts the balancing performed by `LAPACK.gebal!`
"""
function _unbalance!(X, ilo, ihi, scale, n)
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

"""
    Returns a balanced version of X (obtained by `LAPACK.gebal!`)
    and the `ilo`, `ihi`, `scale` parameters
"""
function _balance(X; job::AbstractChar='B')
    cpX = copy(X)
    ilo, ihi, scale = LAPACK.gebal!(job, cpX)
    return cpX, (ilo, ihi, scale)
end


export _balance!, _unbalance!, _balance






# Matrix exponentials

# La copia di Base.exp!(A), in cui ho sostituito `LAPACK.gesv!` con `\`
function my_Base_exp!(A::StridedMatrix{T}) where T<:LinearAlgebra.BlasFloat
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
    
    _unbalance!(X, ilo, ihi, scale, n)
    return X, si
end


"""
# Usage
**Usage**: X = my_Base_exp(A)

Computes the matrix exponential of A, using the "traditional" scaling and squaring algorithm.

# Arguments
## Positional arguments
- `A::StridedMatrix{<:BlasFloat}`: the matrix whose exponential we want to compute (it's *not* mutated)

# Output
- `X`: the matrix exponential of `A`

# Notes (and peculiar features)
Questa è la mia versione modificata di Base.exp!
- *Non* muta `A`
- `LAPACK.gesv!` è stato sotituito con `\\`
- `LAPACK.gebal!` è stato sostituito con un wrapper che restituisce (anche) 
  la matrice bilanciata, non mutando quella originale.


[^Higham_SaS]
    > Higham, Nicholas J. The scaling and squaring method for the matrix exponential revisited, SIAM J. Matrix Anal. Appl., 26 (2005), pp. 1179–1193.
    > doi: https://epubs.siam.org/doi/10.1137/090768539
"""
function my_Base_exp(A::StridedMatrix{T}) where T<:LinearAlgebra.BlasFloat
    n = LinearAlgebra.checksquare(A)
    if isdiag(A)
        for i in diagind(A, IndexStyle(A))
            A[i] = exp(A[i])
        end
        return A, NaN, 0
    # elseif ishermitian(A)
    #     return copytri!(parent(exp(Hermitian(A))), 'U', true)
    end
    A, (ilo, ihi, scale) = _balance(A)
    nA   = opnorm(A, 1)
    ## For sufficiently small nA, use lower order Padé-Approximations
    if (nA <= 2.1)
        if nA > 0.95
            C = T[17643225600.,8821612800.,2075673600.,302702400.,
                     30270240.,   2162160.,    110880.,     3960.,
                           90.,         1.]
            m = 9
        elseif nA > 0.25
            C = T[17297280.,8648640.,1995840.,277200.,
                     25200.,   1512.,     56.,     1.]
            m = 7
        elseif nA > 0.015
            C = T[30240.,15120.,3360.,
                    420.,   30.,   1.]
            m = 5
        else
            C = T[120.,60.,12.,1.]
            m = 3
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

        s = 0
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
        m = 13
    end

    # Undo the balancing    
    _unbalance!(X, ilo, ihi, scale, n) 
    return X, m, s == 0 ? s : ceil(Int, s)
end


export my_Base_exp!, my_Base_exp


end #module