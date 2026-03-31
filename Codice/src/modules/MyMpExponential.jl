module MyMpExponential

using LinearAlgebra




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
    # think: what if I want the output to be (e.g.) in double precision?
    outputclass = something(outputclass, big(T))

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
    # w.r.t. questo `setprecision`, di sicuro ci sarà (i) una formula migliore (ii) un modo più pulito che gli autori avevano in mente
    # think: how many @. can we shove in here?
    B = setprecision(floor(Int64, 1.2 * precision(outputclass, base=10)), base=10) do 
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
        B = setprecision(floor(Int64, 1.2 * precision(outputclass, base=10)), base=10) do 
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

    # WIP: queste outputclass sono un triplo salto carpiato inutile, se 
    #      voglio ritornare una matrice di double. Di fatto sto argomento 
    #      è inutile e ridondante così com'è gestito ora.

    return Y
end




# Paterson-Stockmeyer on Taylor (`polyvalm_tay_exp` in the original MATLAB)
"""
    polyvalm_tay_exp(A, m, s)

Evaluates ``T_m(2^{-s}A)``, the Taylor polynomial of degree `m` on `2^(-s)A`,
using the Paterson-Stockmeyer scheme.

*OSS*: the degree `m` must be one of the "optimal ones" (see article [HF19_mpexpm]), 
i.e. obtained by the formula `floor((i+2)^2 / 4)` for some ``i\\ge 0``. 
Otherwise, the result will be incorrect
"""
function polyvalm_tay_exp(A::AbstractMatrix, m::Integer, s::Integer)
    n = LinearAlgebra.checksquare(A)

    ν = ceil(typeof(m), √m)    # the "batch size"
    ν == 0 && throw(DomainError(lazy"Polynomial degree m is $(m). A value greater than 1 is expected"))
    r = floor(m/ν)  # the "degree" of the P.-S. polynomial

    scaling = 2^s

    Apows = [A^k for k=0:ν] #WIP. This should be in scope for this function.

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