module MyStructs

using LinearAlgebra

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

end #module