## Imports
using Test
using LinearAlgebra


## Unit tests
@testset "AandPowsStruct - three-arguments constructor (inner)" begin 
    n = 2
    A = rand(n,n)

    # full construction 
    S1 = @test_nowarn AandPowsStruct(A, [A^k for k=0:3], true)
    @test S1 isa AandPowsStruct
    @test S1.powers == [A^k for k=0:3]
    @test S1.use_taylor == true
    S2 = @test_nowarn AandPowsStruct(A, [A^(2k) for k=0:3], false)
    @test S2 isa AandPowsStruct
    @test S2.powers == [A^2k for k=0:3]
    @test S2.use_taylor == false

    # error if a nonsquare matrix is provided
    S1 = @test_throws DimensionMismatch AandPowsStruct(rand(2,3), [A^k for k=0:2], true)
    S2 = @test_throws DimensionMismatch AandPowsStruct(rand(2,3), [A^k for k=0:2], false)

    # error if `S.use_taylor` and `S.powers` don't correspond
    S1 = @test_throws ArgumentError AandPowsStruct(A, [A^(2k) for k=0:2], true)
    S2 = @test_throws ArgumentError AandPowsStruct(A, [A^k for k=0:2], false) 

    # error if the identity matrix is not the first power
    S1 = @test_throws ArgumentError AandPowsStruct(A, [A^k for k=1:3], true)
    S2 = @test_throws ArgumentError AandPowsStruct(A, [A^(2k) for k=1:3], false)

    # either `A^0` and `I(n)` work
    S1 = @test_nowarn AandPowsStruct(A, [A^0, A, A^2], true)
    @test S1 isa AandPowsStruct
    S2 = @test_nowarn AandPowsStruct(A, [A^0, A^2, A^4, A^6], false)
    @test S2 isa AandPowsStruct
    S1 = @test_nowarn AandPowsStruct(A, [I(n), A, A^2], true)
    @test S1 isa AandPowsStruct
    S2 = @test_nowarn AandPowsStruct(A, [I(n), A^2, A^4, A^6], false)
    @test S2 isa AandPowsStruct

end


@testset "AandPowsStruct - two-arguments constructor (without powers)" begin 
    n = 2
    A = rand(n,n)
    Id = I(n)

    # construction goes well (Taylor)
    S1 = @test_nowarn AandPowsStruct(A, true)
    @test S1 isa AandPowsStruct
    @test S1.use_taylor == true
    @test S1.powers == [A^0, A]

    # construction goes well (Padé)
    S1 = @test_nowarn AandPowsStruct(A, false)
    @test S1 isa AandPowsStruct
    @test S1.use_taylor == false
    @test S1.powers == [A^0, A^2]   
    
    # error if a nonsquare matrix is provided
    S1 = @test_throws DimensionMismatch AandPowsStruct(rand(2,3), true)
    S2 = @test_throws DimensionMismatch AandPowsStruct(rand(2,3), false)

end


@testset "AandPowsStruct - three-arguments constructor (outer)" begin
    @test_skip "Disabled: the outer three-argument constructor is intentionally not supported"

    # n = 2
    # A = rand(n,n)
    # emptyList = []

    # the struct is created on a empty list (Taylor)
    # S1 = @test_nowarn AandPowsStruct(A, emptyList, true)
    # @test S1 isa AandPowsStruct
    # @test S1.powers != emptyList

    # the struct is created on a empty list (Padé)
    # S1 = @test_nowarn AandPowsStruct(A, emptyList, false)
    # @test S1 isa AandPowsStruct
    # @test S1.powers != emptyList

    # the provided list is discarded if too short 
    # M = rand(2,2)
    # S1 = @test_nowarn AandPowsStruct(A, [M], true)
    # @test S1 isa AandPowsStruct
    # @test S1.powers != [M]
    # @test S1.powers == [I(n), A]
    # S2 = @test_nowarn AandPowsStruct(A, [M], false)
    # @test S2 isa AandPowsStruct
    # @test S2.powers != [M]
    # @test S2.powers == [I(n), A^2]

    # the provided list is discarded if too short (albeit a correct element is there)
    # LI = [I(n)]
    # S1 = @test_nowarn AandPowsStruct(A, LI, true)
    # @test S1 isa AandPowsStruct
    # @test S1.powers != LI
    # @test S1.powers == [I(n), A]
    # S2 = @test_nowarn AandPowsStruct(A, LI, false)
    # @test S2 isa AandPowsStruct
    # @test S2.powers != LI
    # @test S2.powers == [I(n), A^2]

    # with no flag, too-short lists fail because Apows[2] is accessed
    # S1 = @test_throws BoundsError AandPowsStruct(A, emptyList)
    # S2 = @test_throws BoundsError AandPowsStruct(A, [A])

    # mismatch in matrix size inside Apows is rejected by power checks
    # M3 = rand(3,3)
    # S1 = @test_throws DimensionMismatch AandPowsStruct(A, [I(n), M3], true)
    # S2 = @test_throws DimensionMismatch AandPowsStruct(A, [I(n), M3], false)
end


@testset "AandPowsStruct - two-arguments constructor (without flag)" begin 
    n = 2
    A = rand(2,2)

    # construction fails on a list with wrongly typed elements 
    S1 = @test_throws ArgumentError AandPowsStruct(A, [1, 2, 3])

    # constructor works as expected (Taylor)
    S1 = @test_nowarn AandPowsStruct(A, [I(n), A])
    @test S1 isa AandPowsStruct
    @test S1.use_taylor == true
    @test S1.powers == [I(n), A]

    # constructor works as expected (Padé)
    S2 = @test_nowarn AandPowsStruct(A, [I(n), A^2])
    @test S2 isa AandPowsStruct
    @test S2.use_taylor == false
    @test S2.powers == [I(n), A^2]

    # construction fails if the flag can't be inferred reliably
    S1 = @test_throws ArgumentError AandPowsStruct(A, [A^0, A^3])

    # the flag can be inferred, but construction fails if the list is wrong 
    S1 = @test_throws ArgumentError AandPowsStruct(A, [A, A])

end



@testset "AandPowsStruct - wrong ways to construct" begin 
    n = 2
    A = rand(2,2)

    # providing a list of something else (not matrices) to the three-arguments constructor
    S1 = @test_throws MethodError AandPowsStruct(A, [1,2,3], true)
    S2 = @test_throws MethodError AandPowsStruct(A, [1,2,3], false)

    # providing a list of something else (not matrices) to the two-arguments constructor
    S1 = @test_throws ArgumentError AandPowsStruct(A, [1,2,3])
    S2 = @test_throws ArgumentError AandPowsStruct(A, [1,2,3])

    # Providing a credible list, but a nonsquare matrix 
    S1 = @test_throws DimensionMismatch AandPowsStruct(rand(2,3), [A^k for k=0:2], true)
    S2 = @test_throws DimensionMismatch AandPowsStruct(rand(2,3), [A^(2k) for k=0:2], false)

    # Construction with just the matrix doesn't work
    S = @test_throws MethodError AandPowsStruct(A)

    # Construction with just the flag doesn't work
    S1 = @test_throws MethodError AandPowsStruct(true)
    S2 = @test_throws MethodError AandPowsStruct(false)

    
end