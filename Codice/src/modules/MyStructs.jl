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
                Apows[i] == A^(i-1) || error(lazy"The $(i)-th power in Apows is not A^$(i-1).")
            end
        else
            for i in eachindex(Apows)
                Apows[i] == A^(2*(i-1)) || error(lazy"The $(i)-th power in Apows is not A^$(2*(i-1)).")
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
AandPowsStruct(
    A::AbstractMatrix, 
    Apows::AbstractVector,
    b::Bool
) = begin 
        print("Outer constructor called!\n")
        if length(Apows) < 2
            AandPowsStruct(A, b)    # discard the provided list if too short
        else 
            AandPowsStruct(A, Apows, b)
        end
    end

# in case the flag was not provided
AandPowsStruct(
    A::AbstractMatrix, 
    Apows::AbstractVector
) = begin 
        if Apows[2] == A
            AandPowsStruct(A, Apows, true)
        elseif Apows[2] == A^2
            AandPowsStruct(A, Apows, false)
        else 
            error(lazy"Apows[2] should either be A or A²")
        end
    end


export AandPowsStruct



############ Struct per contenere i fattoriali ############

# struct to hold the computed factorials (aim: cache computed results)
struct Factorials
    f_vec::Vector{BigInt}
    Factorials(n) = 
        begin
            v = [factorial(big(k)) for k=0:n]
            new(v)
        end
    Factorials() = new([big(1)])
end


function (f::Factorials)(
    n::Real; 
    return_type::DataType=BigInt
)
    try
        return_type(f.f_vec[n+1])
    catch BoundsError
        l = length(f.f_vec)
            # n! is computed, together with the missing factorials
        #alternative: nuovi_f = [factorial(big(k)) for k=l:n] 
        nuovi_f = l:n .|> big .|> factorial 
        append!(f.f_vec, nuovi_f)
        return return_type(f.f_vec[end])
    end
end


function (f::Factorials)(
    r::AbstractRange; 
    return_type::DataType=BigInt
)
    try
        fst, _..., lst = r
        return_type.(f.f_vec[fst+1:lst+1])
    catch BoundsError
        fst, _..., lst = r
        l = length(f.f_vec)
        #alternative: nuovi_f = [factorial(big(k)) for k=l:lst]
        nuovi_f = l:lst .|> big .|> factorial
        append!(f.f_vec, nuovi_f)
        return return_type.(f.f_vec[fst+1:lst+1])
    end
end

export Factorials



end #module