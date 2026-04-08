module MyMpExponential

using LinearAlgebra
using Symbolics


#### Funzioni per valutare polinomi di matrici ####

# Paterson-Stockmeyer on a generic polynomial (`EvalPolyPS` in the article)
"""
"""
function polyvalm_ps!(
        Apows::AbstractVector{<:AbstractMatrix},
        s::Real,
        # think: shall we assume that the coefficient are computed 
        #        in high precision?
        β_vec::AbstractVector;
        outputclass::Union{Type,Nothing}=nothing     # in the MATLAB, that's A's type after some transformations
) 
    length(Apows) > 1 || throw(ArgumentError(lazy"Supply at least the 0th and 1st power of the matrix"))

    # determine outputclass internally
    T = promote_type(eltype.(Apows)...)
    # think: if T <: Float32 then one could think about promoting to 
    #        Float64 (widen(T) would do it) as a default. 
    #        But setting precision is only truly doable with BigFloats
    # WIP: `to_output_type` introdotta per poter usare i `Num`
    to_output_type(::Type{T}) where T = big(T)
    to_output_type(::Type{Symbolics.Num}) = Symbolics.Num
    outputclass = something(outputclass, to_output_type(T))

    n = LinearAlgebra.checksquare(Apows[2])

    m = length(β_vec) - 1      # degree of the polynomial
    ν = ceil(typeof(m), √m)    # the "batch size"

    if ν == 0
        Y = β_vec[1] * Apows[1]
        return Y
    end

    r = fld(m, ν)              # the "degree" of the P.-S. polynomial
    # think: use a struct with a Vector (dictionary) and a length attribute, 
    #        for `Apows`? Maybe a 3-dim array?
    # think: is it actually an invariant, that powers beyond the `length` are not computed?
    #        (in the MATLAB, there's a variable to keep track of the length)
    for i = length(Apows)+1:ν+1
        if isodd(i)
            i_half = div(i - 1, 2) + 1
            push!(Apows, Apows[i_half]^2)
        else
            # Check that the i-th power is not already there
            # something like assert(Apows[i] == zeros(n, n))
            push!(Apows, Apows[2] * Apows[i-1])
        end
    end
    # Apows.length = max(Apows.length, ν+1), if one wants to overdo

    # promotion creates a copy. Can we avoid the use of this copy?
    # oss: the original MATLAB code creates a copy as well. 
    # oss: creating a copy and then converting is smart, as the divisions 
    #      in the `for` loop are flops and not BigFlops
    mpowers = copy(Apows)   
    for i = 1:ν+1
        @. mpowers[i] /= 2^(2*big(s)*(i-1))
    end
    # convert creates an alias, if Apows already contains matrices of `outputclass`.
    # Thus, the original accuracy of A would be preserved
    mpowers = convert(Vector{AbstractMatrix{outputclass}}, mpowers) 

    # Evaluate the last B-term, the one of degree m - ν*r = m mod ν
    # think: how many @. can we shove in here?
    B = setprecision(floor(Int64, 1.2 * precision(BigFloat))) do 
        # oss: by using `outputclass`, we can use only the number of significant
        #      digits specified by the precision in this block. 
        #      (aka the accuracy of β_vec is limited by that allowed in the scope of 
        #      this `do` block). If we use `big`, then we can use them all. 
        #      It makes no difference in practice I would say, 
        #      as they're limited by the accuracy of `mpowers` (most surely smaller than that of the coeffs).
        #      If we used `big` and `Apows` contained matrices computed at a higher precision,  
        #      then B would be more accurate than what the precision would prescribe. However, this 
        #      kind of defeats the purpose of `setprecision(...)`. Doesn't it? 
        # oss: what if `outputclass` is something like `Float64`? :D should be ok
        # think: maybe I should `outputclass.()` the matrices in `mpowers` as well.
        #        However, these are nasty memory operations...
        B = outputclass(β_vec[m+1]) * mpowers[m - ν*r + 1]
        for j = m-1:-1:ν*r
            if j == ν*r
                B += outputclass(β_vec[ν*r + 1]) * I(n)
            else 
                B += outputclass(β_vec[j+1]) * mpowers[m - ν*r - (m-j) + 1]
            end
        end
        B;
    end

    # Evaluate B-terms of degree ν-1, and evaluate the main polynomial using Horner
    Y = convert(AbstractMatrix{outputclass}, B) # to retain original precision 
                                                # (B is computed at higher precision, hopefully)
    for kk = r-1:-1:0
        # compute B coeff. in slightly higher precision
        B = setprecision(floor(Int64, 1.2 * precision(BigFloat))) do 
            # unless β_vec is computed with high accuracy there may be garbage here
            B = outputclass(β_vec[ν*kk + 1]) * I(n)  
            for j = 1:ν-1
                B += outputclass(β_vec[ν*kk + j + 1]) * mpowers[j+1]
            end
            B;
        end
        # nel MATLAB originale, c'è + cast(B, outputclass), che ha senso perché
        # Y è mantenuta in `outputclass`. Però c'è una copia in più.
        # Ho come l'impressione che tutte queste acrobazie con i tipi non servano a niente, 
        # alla fin fine
        Y = Y * mpowers[ν+1] + convert(AbstractMatrix{outputclass}, B)
    end

    # WIP: questo `outputclass` è ben gestito?
    # WIP: questi `setprecision` che senso hanno?

    return Y
end


# Paterson-Stockmeyer on Taylor polynomials (this function doesn't exist in the article).
function polyvalm_tay_exp(
    Apows::AbstractVector{<:AbstractMatrix}, 
    m::Integer, s::Real
)
    length(Apows) > 1 || throw(ArgumentError(lazy"Supply at least the 0th and 1st power of the matrix"))
    n = LinearAlgebra.checksquare(Apows[1])

    ν = ceil(typeof(m), √m)    # the "batch size"
    ν == 0 && throw(DomainError(lazy"Polynomial degree m is $(m). A value greater than 1 is expected"))
    r = floor(m/ν)  # the "degree" of the P.-S. polynomial

    scaling = 2^s

    #Apows = [A^k for k=0:ν] #WIP. This should be in scope for this function.

    Y = Apows[ν+1] / (scaling * m)
    for j = ν-1:-1:1
        Y = (Y + Apows[j+1]) / (scaling * (m - ν + j))
    end

    Y = Apows[ν+1] * Y;
    for k = r-2:-1:1
        for j = ν:-1:1
            Y = (Y + Apows[j+1]) / (scaling * (k * ν + j))
        end
        Y = Apows[ν+1] * Y
    end

    for j = ν:-1:1
        Y = (Y + Apows[j+1]) / (scaling * j)
    end
    
    Y = Y + I(n)
    # return Y    
end


export polyvalm_tay_exp, polyvalm_ps!



#### Approssimanti dell'esponenziale (Taylor e Padé) ####

# approximate expm using Taylor (`EvalPadeTayl` in the article)
"""
    expm_taylor(A, Apows, m, s)

Evaluates ``T_m(2^{-s}A)``, the Taylor polynomial of degree `m` on `2^(-s)A`,
using the Paterson-Stockmeyer scheme.

*OSS*: `Apows` holds powers of ``A``. It should be supplied with all 
       the required powers, but currently the function computes it 
       if one is not supplied. **Supplying a nonempty `Apows` without 
       the necessary powers of ``A`` will break 

*OSS*: the degree `m` must be one of the "optimal ones" (see article [HF19_mpexpm]), 
i.e. obtained by the formula `floor((i+2)^2 / 4)` for some ``i\\ge 0``. 
Otherwise, the result will be incorrect
"""
function expm_taylor(
    A::AbstractMatrix,
    # WIP: type specification is disabled 
    #      to allow the computation of `Apows` inside
    Apows,#::AbstractVector{<:AbstractMatrix},
    m::Integer,
    s::Real
)
    # oss: we just check that we pass something,
    #      not that we pass the right thing (i.e. [I,A,…,Aᴺ])
    if isnothing(Apows) || isempty(Apows)
        # WIP: la funzione prende `Apows` in input
        # Q: il calcolo di `Apows` qui non c'è. Dove viene fatto,
        #    nel codice originale? Perché le potenze di A ci sono tutte??
        ν = ceil(typeof(m), √m)     # the "batch size" of P.-S.
        # think: assegnamo in-place? Importa a qualcuno?
        Apows = [A^k for k=0:ν]     # WIP. This should be in scope for this function.
    end
    try
        return polyvalm_tay_exp(Apows, m, s)
    catch e
       if isa(e, BoundsError)
           msg = "Providing a nonempty `Apows` without the necessary powers gives error.\n" *
                 "Are you sure you initialized `Apows` with powers from 0 to $ν?\n"
           println(e)
           print(msg)
       else
           print("Error caught!\n")
           print(e)
       end
    end
    return
end


# approximate expm using Padé (`EvalPadeDiag` in the article)
"""
"""
function expm_diagonal!(
    A::AbstractMatrix,
    Apows::AbstractVector{<:AbstractMatrix},
    m::Integer,
    s::Real;
    cheap_r::Bool=true
)
    n = LinearAlgebra.checksquare(A)

    # Compute coefficients of the Padé approximants using the formulas
    c_num, c_den = setprecision(floor(Int64, 2*precision(BigFloat))) do 
        mb = big(m)
        ranges = 0:mb
        c_num = (factorial(mb)/factorial(2mb)) ./
                (factorial.(mb .- ranges) .* factorial.(ranges)) .*
                factorial.(2mb .- ranges)
        c_num[1] = big(1.)
        c_den = ((-1).^ranges) .* c_num
        c_num, c_den;
    end

    # compute even and odd parts, using even/odd coefficients
    Uₑ = polyvalm_ps!(Apows, s, c_num[1:2:end])
    if m >= 1 
        Uₒ = polyvalm_ps!(Apows, s, c_num[2:2:end])
        Uₒ = (A / 2^s) * Uₒ
    else 
        Uₒ = zeros(eltype(A), size(A))
    end

    if cheap_r
        Qₘ = Uₑ - Uₒ
        return Qₘ \ (2*Uₒ) + I(n)
    else 
        Pₘ = polyvalm_ps!(Apows, s, c_num)
        Qₘ = polyvalm_ps!(Apows, s, c_den)
        return Qₘ \ Pₘ
    end
end


export expm_taylor, expm_diagonal!



#### Funzioni per il ricalcolo delle diagonali per lo squaring triangolare superiore ####

# Recompute diagonals of a general block triangular matrix (Overwrites Y)
"""
    recompute_diagonals!(T, Y)

Recomputes the main diagonal and first upper diagonal elements of 
``Y ≈ e^T`` using those of ``T``. 
Namely, the aforementioned elements of ``Y`` are replaced with 
quantities computed using exact formulas, computed on the original elements
(those of ``T``).

*Note*: this function overwrites `Y`.
"""
function recompute_diagonals!(
    M::AbstractMatrix{T}, 
    Y::AbstractMatrix{T}
) where {T}
    n = LinearAlgebra.checksquare(M)
    i = 1
    while i <= n
        # invariant: [i,i] is the top-left corner of a block 
        # (be it 1x1, 2x2, or two successive 1x1s (aka triangular 2x2))
        if (i+1 == n) || (i <= n-2 && M[i+2,i+1] == 0)
            # we're in a 2x2 block, eventually triangular: 
            # in such a block, T[i+2,i+1] is always 0
            
            # oss: the authors check for exact equality, while 
            #      I relax it a little bit. Is it ok? Is it good practice?
            # oss: maybe I should check for it to be ≈ 0 w.r.t. the diagonal elements...
            condition = norm(M[i+1,i]) <= √eps(T)*min(norm(M[i,i]), norm(M[i+1,i+1]))
            if condition    # the block is triangular
                Y[i:i+1,i:i+1] = expm2by2_tri(M[i:i+1,i:i+1])
                # oss: elements T[i+1,i] is set to 0 after this. Is it ok?
            else                # the block is full 
                Y[i:i+1,i:i+1] = expm2by2_full(M[i:i+1,i:i+1])
            end
            i += 2      # exit this block and actually go to the next one
        else 
            # we're in a 1x1 block followed by a 2x2 full block (or at the boundary)
            Y[i,i] = exp(M[i,i])
            i += 1
        end
    end
    return Y
end


function sinch(x)
    x == 0 ? 1 : sinh(x)/x
end


"""
    Y = expm2by2_full(B)

Computes ``Y = e^B``, the exponential of a full ``2\\times 2`` 
block `B`, using formula (2.2) from [alhi09n].

[alhi09n]
    > Higham, N. J. and Al-Mohy, A. H. A New Scaling and Squaring Algorithm for the Matrix Exponential
    > SIAM J. Matrix Anal. Appl., Vol 31.3 (2010), pp.970-989
    > doi: 10.1137/09074721X
"""
function expm2by2_full(B)
    Y = similar(B)
    b11, b21, b12, b22 = B[:]
    b11mb22 = b11 - b22
    δ = sqrt((b11mb22)^2 + 4*b12*b21)/2; # μ/2 in the formula
    exp_apd2 = exp((b11+b22)/2);
    coshδ  = cosh(δ);
    sinchδ = sinch(δ);
    Y[1,1] = exp_apd2 * (coshδ + (b11mb22)/2 * sinchδ);
    Y[2,1] = exp_apd2 * b21 * sinchδ;
    Y[1,2] = exp_apd2 * b12 * sinchδ;
    # Q: perché non usare `-(b11mb22)/2` e risparmiare un conto?
    #    Problemi di cancellazione numerica? 
    Y[2,2] = exp_apd2 * (coshδ + (b22-b11)/2 * sinchδ);    
    return Y
end


function expm2by2_full!(B)
    b11, b21, b12, b22 = B[:]
    b11mb22 = b11 - b22;
    δ = sqrt((b11mb22)^2 + 4b12*b21)/2; # μ/2 in the formula
    exp_apd2 = exp((b11+b22)/2);
    coshδ  = cosh(δ);
    sinchδ = sinch(δ);
    B[1,1] = exp_apd2 * (coshδ + (b11mb22)/2 * sinchδ);
    B[2,1] = exp_apd2 * b21 * sinchδ;
    B[1,2] = exp_apd2 * b12 * sinchδ;
    # Q: perché non usare `-(b11mb22)/2` e risparmiare un conto?
    #    Problemi di cancellazione numerica? 
    B[2,2] = exp_apd2 * (coshδ + (b22-b11)/2 * sinchδ);  
end


"""
    expm2by2_tri(T)

Computes ``Y = e^T``, the exponential of a upper triangular ``2\\times 2``
block `T`, using formula (10.42) from [Higham].

[Higham]
    > Higham, N. J. Functions of Matrices, SIAM, 2008
    > doi: https://doi.org/10.1137/1.9780898717778
"""
function expm2by2_tri(M::AbstractMatrix{T}) where {T}
    Y = zeros(size(M)...)
    M₁, M₂ = diag(M)

    Y[1,1] = exp(M₁)
    Y[2,2] = exp(M₂)

    M₁ += M₂        # M₁ ← M[1,1] + M[2,2]
    M₂ -= M[1,1]    # M₂ ← M[2,2] - M[1,1]

    exp_arg   = M₁ / 2
    sinch_arg = M₂ / 2

    if max(exp_arg, abs(sinch_arg)) < log(floatmax(T))    # guard against overflow
        Y[1,2] = M[1,2] * exp(exp_arg) * sinch(sinch_arg)
    else
        # Numerical cancellation if M[2,2] ≈ M[1,1] 
        # we use divided differences in this case
        Y[1,2] = M[1,2] * (Y[2,2] - Y[1,1]) / M₂
    end
    return Y
end


function expm2by2_tri!(M::AbstractMatrix{T}) where {T}
    M₁, M₂ = diag(M)

    M[1,1] = exp(M₁)
    M[2,2] = exp(M₂)

    M₁ += M₂        # M₁ ← M[1,1] + M[2,2]
    M₂ -= M[1,1]    # M₂ ← M[2,2] - M[1,1]

    exp_arg   = M₁ / 2
    sinch_arg = M₂ / 2

    if max(exp_arg, abs(sinch_arg)) < log(floatmax(T)) # guard against overflow
        M[1,2] *= exp(exp_arg) * sinch(sinch_arg)
    else
        # Numerical cancellation if M[1,1] ≈ M[2,2]
        # We use divided differences in this case
        M[1,2] *= (M[2,2] - M[1,1]) / M₂
    end
end


export recompute_diagonals!, expm2by2_full, expm2by2_full! 
export expm2by2_tri, expm2by2_tri!















function exp_mp(A::AbstractMatrix{T};
                precision::Union{AbstractFloat,Integer} = eps(T)    
    ) where {T<:Real}

    print("Precision = $(precision)\n")
    print("eps(eltype(A)) = $(eps(eltype(A)))\n")

    if precision isa AbstractFloat
        print("Floating point!\n")
    else
        print("Intero!\n")
    end



end

export exp_mp



end #module