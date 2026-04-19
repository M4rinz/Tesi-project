## Imports
using Test
using LinearAlgebra


## Unit tests
@testset "FactorialsStruct - (inner) constructor" begin 
    
    # constructing with no arguments just initializes with 0! 
    F = @test_nowarn FactorialsStruct()
    @test F isa FactorialsStruct
    @test F.f_vec == [1]
    @test eltype(F.f_vec) == BigInt

    # 1-argument constructor, receiving a positive integer
    n = 4
    F = @test_nowarn FactorialsStruct(n)
    @test F isa FactorialsStruct
    @test length(F.f_vec) == n+1
    @test F.f_vec == [factorial(k) for k=0:n]
    @test eltype(F.f_vec) == BigInt

    # 1-argument constructor, failing if argument is not an integer
    n = 3.0
    F = @test_throws MethodError FactorialsStruct(n)

    # 1-argument constructor, rejecting negative integers
    n = -4
    F = @test_throws DomainError FactorialsStruct(n)


end


@testset "calling the FactorialStruct with an integer" begin 
    # base-case call from a void initialization
    F = FactorialsStruct()
    n! = @test_nowarn F(0)
    @test n! == 1
    @test n! isa BigInt

    # calling with uncached n extends f_vec with missing factorials
    n = 3
    F = FactorialsStruct()
    n! = @test_nowarn F(n)
    @test n! == factorial(n)
    @test n! isa BigInt
    @test F.f_vec == [factorial(k) for k=0:n]

    # calling with uncached n, specifying the return type
    n = 3
    F = FactorialsStruct()
    n! = @test_nowarn F(n, return_type=Int64)
    @test n! == factorial(n)
    @test n! isa Int64
    @test F.f_vec == [factorial(k) for k=0:n]
    @test eltype(F.f_vec) == BigInt

    # calling with a cached n works
    n = 3
    F = FactorialsStruct(5)
    n! = @test_nowarn F(n)
    @test n! == factorial(n)

    # calling with a cached n specifying the return type 
    n! = @test_nowarn F(n, return_type=Int64)
    @test n! == factorial(n)  
    @test n! isa Int64  

    # failure when calling with n not being an integer
    n = 3.0
    n! = @test_throws MethodError F(n)

    # failure when calling with n being a negative integer
    n = -1 
    n! = @test_throws DomainError F(n)

    # calling with cached n should not grow f_vec
    F = FactorialsStruct(10)
    l = length(F.f_vec)
    n! = @test_nowarn F(3)
    @test n! == factorial(3)
    @test length(F.f_vec) == l

    # conversion may fail for factorials too large for Int64
    n! = @test_throws InexactError F(21, return_type=Int64)

end


@testset "calling the FactorialsStruct with a range" begin 
    # calling with an uncached range extends f_vec with missing factorials
    r = 2:5
    F = FactorialsStruct()
    r! = @test_nowarn F(r)
    @test r! == [factorial(k) for k in r]
    @test eltype(r!) == BigInt
    @test F.f_vec == [factorial(k) for k=0:last(r)]

    # calling with uncached range, specifying the return type
    r = 2:5
    F = FactorialsStruct()
    r! = @test_nowarn F(r, return_type=Int64)
    @test r! == [factorial(k) for k in r]
    @test eltype(r!) == Int64
    @test F.f_vec == [factorial(k) for k=0:last(r)]
    @test eltype(F.f_vec) == BigInt

    # calling with a cached range works
    r = 1:4
    F = FactorialsStruct(6)
    r! = @test_nowarn F(r)
    @test r! == [factorial(k) for k in r]

    # calling with a cached range specifying the return type
    r! = @test_nowarn F(r, return_type=Int64)
    @test r! == [factorial(k) for k in r]
    @test eltype(r!) == Int64

    # failure when calling with range endpoints not being integers
    r = 1.0:3.0
    r! = @test_throws MethodError F(r)

    # failure when calling with a range containing negative integers
    r = -2:2
    r! = @test_throws DomainError F(r)

    # descending range is accepted and returns the same descending order
    r = 5:-1:2
    F = FactorialsStruct()
    r! = @test_nowarn F(r)
    @test r! == [factorial(k) for k in r]
    @test eltype(r!) == BigInt

    # empty range returns an empty vector
    r = 1:0
    F = FactorialsStruct()
    r! = @test_nowarn F(r)
    @test isempty(r!)

    # empty range with explicit return type keeps the requested eltype
    r! = @test_nowarn F(r, return_type=Int64)
    @test isempty(r!)
    @test eltype(r!) == Int64

    # non-unit positive step is respected
    r = 0:2:8
    F = FactorialsStruct()
    r! = @test_nowarn F(r)
    @test r! == [factorial(k) for k in r]
    @test F.f_vec == [factorial(k) for k=0:8]

    # non-unit negative step is respected
    r = 8:-2:0
    r! = @test_nowarn F(r)
    @test r! == [factorial(k) for k in r]

    # calling with a fully cached range should not grow f_vec
    F = FactorialsStruct(20)
    l = length(F.f_vec)
    r = 3:2:17
    r! = @test_nowarn F(r)
    @test r! == [factorial(k) for k in r]
    @test length(F.f_vec) == l

    # conversion on ranges may fail when one factorial exceeds Int64
    r = 19:22
    r! = @test_throws InexactError F(r, return_type=Int64)

    # empty negative ranges are rejected by domain checks
    r = -2:-3
    r! = @test_throws DomainError F(r)

end