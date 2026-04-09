module MyMpExponential

using LinearAlgebra
using LinearMaps, MatrixEquations
using Symbolics


############ Valutare polinomi di matrici ############

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



############ Approssimanti dell'esponenziale (Taylor e Padé) ############

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
    expm_diagonal!(A, Apows, m, s; kwargs...)

Approximates ``e^(2^{-s}A)`` using the diagonal Padé approximant of degree `m`.
The approximant is evaluated using the Paterson-Stockmeyer scheme.

# Arguments
- `A::AbstractMatrix`: The matrix whose scaled version we want the exponential of
- `Apows::AbstractVector`: A vector holding the powers of ``A^2``. At least 
                           ``I`` and ``A^2`` must be present. *This vector is mutated
                           by the function*
- `m::Integer`: The degree of the diagonal approximant. *Note*: an optimal 
                degree must be used.
- `s::Real`: The scaling factor

# Keyword arguments:
- `cheap_r::Bool`: whether to use a smart formula for ``Q_m^{-1}P_m``. 
                   Defaults to `true`.
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



########## Ricalcolo delle diagonali per lo squaring triangolare superiore ##########

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
# Q: e se M e Y non avessero stesso tipo? (expm_diagonal! restituisce BigFloat)
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
    #return Y
end


function sinch(x::Real)
    x == 0 ? 1 : sinh(x)/x
end

function sinch(z::Complex)
    if real(z) == 0 
        z == 0 ? 1 : imag(sinh(z)) / imag(z)
    else 
        z == 0 ? 1 : sinh(z) / z
    end
end


"""
    Y = expm2by2_full(B)

Computes ``Y = e^B``, the exponential of a full ``2\\times 2`` 
block `B`, using formula (2.2) from [^alhi09n].

[^alhi09n]
    > N. J. Higham and A. H. Al-Mohy, A New Scaling and Squaring Algorithm for the Matrix Exponential
    > SIAM J. Matrix Anal. Appl., Vol 31.3 (2010), pp.970-989
    > [doi:10.1137/09074721X](https://doi.org/10.1137/09074721X)
"""
function expm2by2_full(B)
    Y = similar(B)
    b11, b21, b12, b22 = B[:]
    b11mb22 = b11 - b22
    μsq = (b11mb22)^2 + 4*b12*b21
    if μsq < 0 
        μsq = Complex(μsq)
    end
    δ = sqrt(μsq)/2     # μ/2 in the formula
    exp_apd2 = exp((b11+b22)/2)
    # oss: even if δ is a pure imaginary number, the result is real, 
    #      as cosh and sinch are even functions. And also the computed 
    #      result is guaranteed to be real, because of how the functions are implemented 
    coshδ  = real(cosh(δ))
    sinchδ = real(sinch(δ))
    Y[1,1] = exp_apd2 * (coshδ + (b11mb22)/2 * sinchδ)
    Y[2,1] = exp_apd2 * b21 * sinchδ
    Y[1,2] = exp_apd2 * b12 * sinchδ
    # Q: perché non usare `-(b11mb22)/2` e risparmiare un conto?
    #    Problemi di cancellazione numerica? 
    Y[2,2] = exp_apd2 * (coshδ + (b22-b11)/2 * sinchδ)   
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
block `T`, using formula (10.42) from [^Higham].

[^Higham]
    > Higham, N. J. Functions of Matrices, SIAM, 2008
    > [doi:10.1137/1.9780898717778](https://doi.org/10.1137/1.9780898717778)
"""
function expm2by2_tri(M::AbstractMatrix{T}) where {T}
    Y = zeros(T, size(M)...)
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



############ Upper bound all'errore in avanti ############

"""
"""
function scalar_error_tayl()

end


"""
"""
function scalar_error_pade()

end


"""
"""
function alpha!(
        # Q: quale struttura dati usare per alpha_vec?
        alpha_vec::AbstractVector, 
        s, k, m
    )
    d = div(1 + sqrt(4*(m+k) + 5), 2)   # sarebbe d^{[k/m]}

    if alpha_vec[d+1] == 0 
        if alpha_vec[d] == 0 
            # uffff tutti i parametri di normest!!!!
            # oss: il codice MATLAB originale usa `A` e `Apows`
            #      in doppia precisione. 
            alpha_vec[d] = normest1(d, A, Apows, use_taylor=use_taylor)^(1/j)
        end
        # Proviamo a evitare il calcolo di ‖A^(j+1)‖
        low  = findfirst(!iszero, alpha_vec) # lowest index of a nonzero α
        high = findlast(!iszero, alpha_vec)  # highest index of a nonzero α
        bin_counter = false
        found_upper_bound = false
        while low < high
            # oss: dettagli sulla ratio dietro questo `if` sul quaderno
            # oss: a me torna se (e solo se) ‖ A^d ‖ > 1. Però questo controllo non c'è
            if low + high == d + 1
                if alpha_vec[d] > alpha_vec[low]*alpha_vec[high]
                    found_upper_bound = true
                    break
                end
            end
            if bin_counter
                low += 1
            else 
                high -= 1
            end
            bin_counter = !bin_counter
        end
        if found_upper_bound
            return alpha_vec[d] / 2^s
        else 
            alpha_vec[d+1] == 0 || error("Uhm qualcosa non torna qua…\n")
            alpha_vec[d+1] = normest1(d+1, A, Apows, use_taylor=use_taylor)
        end
    end
    # oss: at one point, alpha_vec[d+1] was 0, then it has been computed.
    #      When this happened, also alpha_vec[d] had been computed.
    α_min = maximum(alpha_vec[d:d+1])   
    α_min /= 2^s
end


function alpha!(
        alpha_dict::Dict,
        s, k, m
    )

end


export alpha!



############ Calcolo (meglio: stima) di || A^d ||₁ ############

# Function that returns the action X -> AᵈX, using only elements in `Apows`
function evalPowVecDiag(
        d::Integer,
        S::AandPowsStruct
    )

    length(S.Apows) > 1 || throw(ArgumentError(lazy"Supply at least the 0th and 1st power of the matrix"))
    
    # determine the type of the matrices elements
    T = promote_type(eltype.(S.Apows)...)
    
    # oss: the MATLAB check is more convoluted but tantamounts to this
    n = LinearAlgebra.checksquare(S.A)

    if use_taylor
        Ad_action = function (X::AbstractVecOrMat)
            p = d
            l = min(length(S.Apows), p+1)
            while p > 0 
                for _ = 1:div(p, l-1)
                    X = S.Apows[l] * X
                end
                p = mod(p, l-1)         
                l = min(l-1, p+1)
            end 
            return X
        end
        Adp_action = function (X::AbstractVecOrMat)
            p = d
            l = min(length(S.Apows), p+1)
            while p > 0 
                for _ = 1:div(p, l-1)
                    X = S.Apows[l]' * X
                end
                p = mod(p, l-1)         
                l = min(l-1, p+1)
            end 
            return X
        end
    else
        Ad_action = function (X::AbstractVecOrMat)
            p = d
            l = length(S.Apows)
            while p > 1 && l > 1
                for _ = 1:div(p, 2*(l-1))
                    X = S.Apows[l] * X
                end 
                p = mod(p, 2*(l-1))         # d -= ⌊d/2(l-1)⌋ * 2(l-1)
                l = min(l-1, div(p,2)+1)
            end
            if p == 1 
                # extra multiplication by A, in case `d` was odd
                X = S.A * X
            end    
            return X
        end
        Adp_action = function (X::AbstractVecOrMat)
            p = d
            l = length(S.Apows)
            while p > 1 && l > 1
                for _ = 1:div(p, 2*(l-1))
                    X = (S.Apows[l])' * X
                end 
                p = mod(p, 2*(l-1))         # d -= ⌊d/2(l-1)⌋ * 2(l-1)
                l = min(l-1, div(p,2)+1)
            end
            if p == 1 
                # extra multiplication by A', in case `d` was odd
                X = (S.A)' * X
            end    
            return X
        end
    end

    kwargs = (
        issymmetric = issymmetric(S.A),
        ishermitian = ishermitian(S.A),
        isposdef    = isposdef(S.A)
    )
    return LinearMap{T}(Ad_action, Adp_action, n; kwargs...)
end


"""
    normest1(d, S)

Computes an estimate of the 1-norm of ``A^d``. If ``A`` is a real nonnegative 
matrix, the estimate is exact, otherwise the elements in `S.Apows` are used.

The implementation of the action of ``A^d`` is the same as the 
EvalPowVecDiag function in [^hf19_mpexpm]. Precisely, a `LinearMap` object
that implements the action is created, then the norm is estimated using 
the `opnorm1est` function from `MatrixEquations.jl` (which in turn 
basically implements `normest1` from [^higham_normest1]).

# Arguments
- `d::Integer`: the power of `A`
- `S::AandPowsStruct`: a `struct` with the fields `use_taylor`,
                       `A` (the matrix) and `Apows` 
    - `A::AbstractMatrix`: the matrix ``A``
    - `use_taylor::Bool`: whether the chosen approximant is the Taylor one or not.                          
    - `Apows::AbstractVector`:` a vector with the powers of ``A``. 
                                It is assumed that `Apows` is ``[I, A, \\dots, A^l]`` 
                                in the former case, and that it's ``[I, A^2, \\dots, A^(2l)]``
                                in the latter case.

# References 
- [^hf19_mpexpm]:
    > N. J. Higham and M. Fasi, An Arbitrary Precision Scaling and Squaring Algorithm for the Matrix Exponential
    > SiAM J. Matrix Anal. Appl., Vol. 40.4 (2019), pp.1233-1256
    > [doi:10.1137/18M1228876](https://doi.org/10.1137/18M1228876)
- [^higham_normest1]:
    > N. J. Higham and F. Tisseur, A block algorithm for matrix 1-norm estimation, with and application to 1-norm pseudospectra
    > SIAM J. Matrix Anal. Appl., Vol 21.4 (2000), pp. 1185–1201.
    > [doi:10.1137/S0895479899356080](https://doi.org/10.1137/S0895479899356080)
"""
function normest1(
        d::Integer,
        S::AandPowsStruct
    )

    length(S.Apows) > 1 || throw(ArgumentError(lazy"Supply at least the 0th and 1st power of the matrix"))
    n = LinearAlgebra.checksquare(S.A)

    if S.A == abs.(S.A)
        e = ones(n)
        for _ = 1:d
            e = (S.A)' * e
        end
        γ_d = norm(e, Inf)
    else
        Ad_action = evalPowVecDiag_mk2(d, S)
        γ_d = MatrixEquations.opnorm1est(Ad_action)    
    end

    return γ_d
end


export EvalPowVecDiag, normest1





















#### Gradi ottimi delle approssimanti ####

function opt_degs_tayl(max_deg::Integer=500)
    # degs[i] = ⌊(i+2)²/4⌋
    degs = [1,    2,    4,    6,    9,   12,   16,   20,   25,
            30,   36,   42,   49,   56,   64,   72,   81,   90,  100,
            110,  121,  132,  144,  156,  169,  182,  196,  210,  225,
            240,  256,  272,  289,  306,  324,  342,  361,  380,  400,
            420,  441,  462,  484,  506,  529,  552,  576,  600,  625,
            650,  676,  702,  729,  756,  784,  812,  841,  870,  900,
            930,  961,  992, 1024, 1056, 1089, 1122, 1156, 1190, 1225,
            1260, 1296, 1332, 1369, 1406, 1444, 1482, 1521, 1560, 1600,
            1640, 1681, 1722, 1764, 1806, 1849, 1892, 1936, 1980, 2025,
            2070, 2116, 2162, 2209, 2256, 2304, 2352, 2401, 2450, 2500]
    degs[degs .< max_deg]   # return degrees less than the provided one
end

function opt_degs_pade(max_deg::Integer=500)
    # degs[i] = 2⋅⌈(i-1)/4⌉⋅(i-1-2⌊(i-2)/4⌋) + 1
    degs = [1,    2,    3,    5,    7,    9,   13,   17,   21,
            25,   31,   37,   43,   49,   57,   65,   73,   81,   91,
            101,  111,  121,  133,  145,  157,  169,  183,  197,  211,
            225,  241,  257,  273,  289,  307,  325,  343,  361,  381,
            401,  421,  441,  463,  485,  507,  529,  553,  577,  601,
            625,  651,  677,  703,  729,  757,  785,  813,  841,  871,
            901,  931,  961,  993, 1025, 1057, 1089, 1123, 1157, 1191,
            1225, 1261, 1297, 1333, 1369, 1407, 1445, 1483, 1521, 1561,
            1601, 1641, 1681, 1723, 1765, 1807, 1849, 1893, 1937, 1981,
            2025, 2071, 2117, 2163, 2209, 2257, 2305, 2353, 2401, 2451,
            2501]
    degs[degs .< max_deg]
end


export opt_degs_pade, opt_degs_tayl







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