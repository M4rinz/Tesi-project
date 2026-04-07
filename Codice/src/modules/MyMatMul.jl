module MyMatMul

using LinearAlgebra
using LinearMaps, MatrixEquations


"""
    evalPowVecDiag_mk1(d, A, Apows, X; kwargs...)

Computes ``A^d X`` using the elements of `Apows` (and potentially `A` once)
"""
function evalPowVecDiag_mk1(
        d::Integer, 
        # Vorrei evitare di dover passare A…
        A::AbstractMatrix,
        Apows::AbstractVector{<:AbstractMatrix},
        X::AbstractVecOrMat;
        use_taylor::Bool = false
    )

    length(Apows) > 1 || throw(ArgumentError(lazy"Supply at least the 0th and 1st power of the matrix"))
    m, n = size(Apows[2])

    if use_taylor 
        l = min(length(Apows), d+1)
        while d > 0 
            for _ = 1:div(d, l-1)
                X = Apows[l] * X
            end
            d = mod(d, l-1)         
            l = min(l-1, d+1)
        end
    else 
        l = length(Apows)
        while d > 1 && l > 1
            for _ = 1:div(d, 2*(l-1))
                X = Apows[l] * X
            end 
            d = mod(d, 2*(l-1))         # d -= ⌊d/2(l-1)⌋ * 2(l-1)
            l = min(l-1, div(d,2)+1)
        end
        if d == 1 
            # extra multiplication by A, in case `d` was odd
            X = A * X
        end
    end

    return X
end 

"""
    f = evalPowVecDiag_mk2(d, A, Apows; kwargs...)

Returns a function (a closure) that computes the action of``A^d`` 
using the elements of `Apows`. The returned function is a `LinearMap`

## Keyword arguments
- `use_taylor`: whether the version is the Taylor one or not.
                In the former case, it is assumed that `Apows` contains
                the consecutive powers from `0` to `length(Apows)-1`

## Returns 
- f::`LinearMap`: the action of ``A^d``, i.e. the function such that 
                  `f(X) = A^d * X`.
"""
function evalPowVecDiag_mk2(
        d::Integer,
        # Vorrei evitare di dover passare A…
        A::AbstractMatrix, 
        Apows::AbstractVector{<:AbstractMatrix};
        use_taylor::Bool = false
    )

    length(Apows) > 1 || throw(ArgumentError(lazy"Supply at least the 0th and 1st power of the matrix"))
    
    # determine the type of the matrices elements
    T = promote_type(eltype.(Apows)...)
    
    # oss: the MATLAB check is more convoluted but tantamounts to this
    n = LinearAlgebra.checksquare(A)

    if use_taylor
        Ad_action = function (X::AbstractVecOrMat)
            p = d
            l = min(length(Apows), p+1)
            while p > 0 
                for _ = 1:div(p, l-1)
                    X = Apows[l] * X
                end
                p = mod(p, l-1)         
                l = min(l-1, p+1)
            end 
            return X
        end
    else
        Ad_action = function (X::AbstractVecOrMat)
            p = d
            l = length(Apows)
            while p > 1 && l > 1
                for _ = 1:div(p, 2*(l-1))
                    X = Apows[l] * X
                end 
                p = mod(p, 2*(l-1))         # d -= ⌊d/2(l-1)⌋ * 2(l-1)
                l = min(l-1, div(p,2)+1)
            end
            if p == 1 
                # extra multiplication by A, in case `d` was odd
                X = A * X
            end    
            return X
        end
    end

    kwargs = (
        issymmetric = issymmetric(A),
        ishermitian = ishermitian(A),
        isposdef    = isposdef(A)
    )
    return LinearMap{T}(Ad_action, n; kwargs...)
end

export evalPowVecDiag_mk1, evalPowVecDiag_mk2


"""

"""
function normest1(
        d::Integer,
        A::AbstractMatrix,
        Apows::AbstractVector{<:AbstractMatrix};
        use_taylor::Bool = false
    )

    length(Apows) > 1 || throw(ArgumentError(lazy"Supply at least the 0th and 1st power of the matrix"))
    n = LinearAlgebra.checksquare(A)

    if A == abs.(A)
        e = ones(n)
        for j = 1:d
            e = A' * e
        end
        γ_d = norm(e, Inf)
    else
        Ad_action = evalPowVecDiag_mk2(d, A, Apows, use_taylor=use_taylor)
        γ_d = MatrixEquations.opnorm1est(Ad_action)    
    end

    return γ_d
end

export normest1




end

