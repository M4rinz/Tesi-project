## Imports
using Test
using LinearAlgebra


## Unit tests
@testset "eval_error with Taylor mode" begin
	A = BigFloat[1 0; 0 2]
	S = AandPowsStruct(A, true)

	x = 0.5
	m = 4
	s = 0
	extra_precision = false

	@test_throws ArgumentError eval_error(S, x, m, s, extra_precision)

	factorials = FactorialsStruct()

	δ, ψ, κ_A = eval_error(S, x, m, s, extra_precision, factorials)

	@test δ >= 0
	@test ψ >= 0
	@test κ_A == 1
end

@testset "eval_error with Diagonal Padé mode" begin 
    # MatrixEquations.opnorm1est (used in the Padé branch) works on Float32/Float64.
    A = rand(2,2)
    S = AandPowsStruct(A, false)

    x = 0.5
    m = 5
    s = 0
    extra_precision = false

	δ₁, ψ₁, κ_A₁ = eval_error(S, x, m, s, extra_precision)

	factorials = FactorialsStruct()
	δ₂, ψ₂, κ_A₂ = eval_error(S, x, m, s, extra_precision, factorials)

	@test isfinite(δ₁) && isfinite(ψ₁) && isfinite(κ_A₁)
	@test isfinite(δ₂) && isfinite(ψ₂) && isfinite(κ_A₂)
	@test δ₁ >= 0 && ψ₁ >= 0 && κ_A₁ >= 0
	@test δ₂ >= 0 && ψ₂ >= 0 && κ_A₂ >= 0

	# In Padé mode, the optional factorials argument is ignored.
	@test δ₁ == δ₂
	@test ψ₁ == ψ₂
	@test κ_A₁ == κ_A₂

end
