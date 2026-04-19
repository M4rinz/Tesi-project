module MyStructs

using LinearAlgebra

############ Struct per contenere la matrice e le sue potenze ############


struct AandPowsStruct
    A::AbstractMatrix
    powers::AbstractVector{<:AbstractMatrix}
    use_taylor::Bool

    AandPowsStruct(
        A::AbstractMatrix,
        Apows::AbstractVector{<:AbstractMatrix},
        use_taylor::Bool
    ) = begin
        _ = LinearAlgebra.checksquare(A)
        if use_taylor
            for i in eachindex(Apows)
                Apows[i] == A^(i-1) || throw(ArgumentError(lazy"The $(i)-th power in Apows is not A^$(i-1)."))
            end
        else
            for i in eachindex(Apows)
                Apows[i] == A^(2*(i-1)) || throw(ArgumentError(lazy"The $(i)-th power in Apows is not A^$(2*(i-1))."))
            end
        end
        new(A, Apows, use_taylor)
    end
end


 ## external constructors

# in case Apows is not provided
AandPowsStruct(
    A::AbstractMatrix, 
    b::Bool
) = begin 
    n = LinearAlgebra.checksquare(A)
    if b
        Apows = [I(n), A]
    else
        Apows = [I(n), A^2]
    end
    AandPowsStruct(A, Apows, b)
end

# in case just zero or one matrix was provided in Apows
# AandPowsStruct(
#     A::AbstractMatrix, 
#     Apows::AbstractVector{T},
#     b::Bool
# ) where {T} = begin 
#     #print("Outer constructor called!\n")
#     if T <: AbstractMatrix || isempty(Apows)
#         if length(Apows) < 2
#             AandPowsStruct(A, b)    # discard the provided list if too short
#         else
#             AandPowsStruct(A, Apows, b)
#         end
#     else 
#         throw(ArgumentError("AandPowsStruct expects Apows to be a vector of matrices."))
#     end
# end

# in case the flag was not provided
AandPowsStruct(
    A::AbstractMatrix, 
    Apows::AbstractVector{T}
) where {T} = begin 
    if T <: AbstractMatrix || isempty(Apows)
        if Apows[2] == A
            AandPowsStruct(A, Apows, true)
        elseif Apows[2] == A^2
            AandPowsStruct(A, Apows, false)
        else 
            throw(ArgumentError(lazy"Apows[2] should either be A or A²"))
        end
    else 
        throw(ArgumentError("AandPowsStruct expects Apows to be a vector of matrices."))
    end
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



end #module