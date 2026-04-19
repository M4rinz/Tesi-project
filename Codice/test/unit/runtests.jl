using Test

include(joinpath(@__DIR__, "..", "..", "src", "modules", "MyMpExponential.jl"))
using .MyMpExponential

const UNIT_DIR = @__DIR__

test_files = String[]
for (root, _, files) in walkdir(UNIT_DIR)
	for file in files
		if endswith(file, ".jl") && file != "runtests.jl"
			push!(test_files, joinpath(root, file))
		end
	end
end

sort!(test_files)

@testset "Unit tests" begin
	for test_file in test_files
		rel = relpath(test_file, UNIT_DIR)
		@testset "$rel" begin
			include(test_file)
		end
	end
end
