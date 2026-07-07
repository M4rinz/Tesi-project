module MyStructs

using LinearAlgebra

############ Struct per contenere la matrice e le sue potenze ############

"""
    AandPowsStruct(A, Apows, use_taylor)

A struct to hold the data used by the `exp_mp` internal functions.

# Fields 
- `A::AbstractMatrix`: The base matrix
- `powers::AbstractVector{<:AbstractMatrix}`: The powers of `A` or `A^2` (depending on `use_taylor`)
- `use_taylor::Bool`: Whether the struct is used by Taylor-related functions or not 

# Notes 
Besides the type specifications of the struct's record, there is no check. 
In particular, it's not checked that `A` is square, that `powers` and `use_taylor`
are consistent, that the matrices in `powers` are the right ones. 
**This saves a lot of overhead, but the responsibility for a correct use is on the 
implementation**
"""
struct AandPowsStruct
    A::AbstractMatrix
    powers::AbstractVector{<:AbstractMatrix}
    use_taylor::Bool
end

export AandPowsStruct

############ Struct per contenere i fattoriali ############

# struct to hold the computed factorials (aim: cache computed results)
struct FactorialsStruct
    f_vec::Vector{BigInt}
    FactorialsStruct(n::Integer) = 
        begin
            n < 0 && throw(DomainError(lazy"FactorialsStruct expects n to be positive"))
            v = [factorial(big(k)) for k=0:n]
            new(v)
        end
    FactorialsStruct() = new([big(1)])
end


function (f::FactorialsStruct)(
    n::Integer; 
    return_type::DataType=BigInt
)
    n < 0 && throw(DomainError(lazy"FactorialsStruct call expects n to be positive"))
    l = length(f.f_vec)
    if n+1 > l
        nuovi_f = l:n .|> big .|> factorial
        append!(f.f_vec, nuovi_f)
    end
    return return_type(f.f_vec[n+1])
end


function (f::FactorialsStruct)(
    r::AbstractRange{<:Integer}; 
    return_type::DataType=BigInt
)
    fst, lst = first(r), last(r)
    stp = step(r)

    (fst < 0 || lst < 0) && throw(DomainError(lazy"FactorialsStruct call expect ranges that span positive integers"))
    isempty(r) && return return_type[]
    
    l = length(f.f_vec)
    mx = max(fst, lst)
    if mx+1 > l
        nuovi_f = l:mx .|> big .|> factorial
        append!(f.f_vec, nuovi_f)
    end
    return return_type.(f.f_vec[fst+1:stp:lst+1])
end

export FactorialsStruct



############ Struct per tenere assieme i parametri di exp_mp ############

struct ExpMpParams
    m
    s
    δ
    ψ
    κ_A
    ϵ
    Y
end

Base.getproperty(p::ExpMpParams, name::Symbol) = begin
    if name === :delta
        return getfield(p, :δ)
    elseif name === :cond_q
        return getfield(p, :κ_A)
    elseif name === :epsilon
        return getfield(p, :ϵ)
    elseif name === :psi
        return getfield(p, :ψ)
    elseif name === :YbeforeSquaring
        return getfield(p, :Y)
    else
        return getfield(p, name)
    end
end


export ExpMpParams


end #module