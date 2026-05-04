"""Questo benchmark è sbagliato (le funzioni non esistono più)
Ma è sempre un utile esempio.
"""
## Imports
using LinearAlgebra, Random, Printf
using BenchmarkTools
using Revise

Revise.includet(joinpath(@__DIR__,"..","..","src","modules","MyMpExponential.jl"))
Revise.includet(joinpath(@__DIR__,"..","..","src","modules","MyHelper.jl"))
using .MyMpExponential, .MyHelper

Random.seed!(42)

function benchmark_ratios(benchStc, benchMut)
    function __return_params__(mean_bench)
        [mean_bench.time, mean_bench.gctime, mean_bench.memory, mean_bench.allocs]
    end
    meanStc = mean(benchStc)
    meanMut = mean(benchMut)

    @printf("""Ratios non_mutating.x / mutating.x for x in:
        time   = %.4g       gctime = %.4g
        memory = %.4g       allocs = %.4g
    
    """,
        (__return_params__(meanStc) ./ __return_params__(meanMut))...
    )
end


## First numerical check: random 2x2 Float64 matrix 
print("We check whether the mutating version of `expm2by2_full` "
 * "is more efficient than the original one, by testing on a random Float64 matrix.\n"
 * "And since we're at it, we also assess the accuracy.\n")

B    = rand(2,2);
expB = exp(B);      # true result

Y = expm2by2_full(B);   # precompile
benchmark = @benchmark expm2by2_full($B);
print("\tStatic (aka non mutating) benchmark:\n")
display(benchmark)
print("\n")

expm2by2_full!(copy(B));    # precompile
benchmark_mut = @benchmark expm2by2_full!(B) setup=(B = copy($B));
print("\tMutating version benchmark:\n")
display(benchmark_mut)
print("\n")
benchmark_ratios(benchmark, benchmark_mut)

print("\tErrors:\n")
@printf("|| Y - exp(B) || / || exp(B) || = %.6g\n", rel_err(Y, expB))
expm2by2_full!(B);      # Overwrites B ← exp(B)
@printf("|| expmB - exp(B) || / || exp(B) || = %.6g\n", rel_err(B, expB))
@printf("|| expmB - Y || / (1 + || expmB || + || Y ||) = %.6g\n", sym_err(B, Y))


## Second numerical check

a, b = rand(2);
A = [a -b; b a]

expA_true = exp(a)*[cos(b) -sin(b); sin(b) cos(b)]; # true result
expA = exp(A);                                      

@printf("|| exp(A) - true_expA || / || true_expA || = %.6g\n", rel_err(expA, expA_true))

Y = expm2by2_full(A);
@printf("|| Y - true_expA || / || true_expA || = %.6g\n", rel_err(Y, expA_true))


## Third numerical check: random 2x2 BigFloat matrix
print("We now check the efficiency in arbitrary precision, "
 * "by running the analogous test as the previous, but with a random 2x2 BigFloat matrix.\n")

B = rand(BigFloat, 2,2);

Y = expm2by2_full(B);       # precompile
print("\tStatic (aka non mutating) benchmark:\n")
benchmark = @benchmark expm2by2_full($B);
display(benchmark)
print("\n")

expm2by2_full!(copy(B));    # precompile

benchmark_mut = @benchmark expm2by2_full!(B) setup=(B = copy($B));
print("\tMutating version benchmark:\n")
display(benchmark_mut)
print("\n")
benchmark_ratios(benchmark, benchmark_mut)

expm2by2_full!(B);  # Overwrites B ← exp(B)

print("\tErrors:\n")
@printf("|| expmB - Y || / (1 + || expmB || + || Y ||) = %.6g\n", sym_err(B, Y))


## Fourth numerical check: how do the two formulas scale with precision?
print("We now run a more extensive benchmark on BigFloat matrices at different precision "
* "to check how well they scale.\n")

using Plots

precs = range(53, 1024; length=10) .|> Int64 ∘ floor

medie_stc, stdevs_stc = [], [];
medie_mut, stdevs_mut = [], [];

for pr in precs
    setprecision(pr) do 
        B = rand(BigFloat, 2,2);

        expm2by2_full(B);   # precompile
        benchStc = @benchmark expm2by2_full($B);
        push!(medie_stc,  mean(benchStc));
        push!(stdevs_stc, std(benchStc));

        expm2by2_full!(copy(B));
        benchMut = @benchmark expm2by2_full!(B) setup=(B = copy($B));
        push!(medie_mut,  mean(benchMut));
        push!(stdevs_mut, std(benchMut));
    end
end

y_stc_times  = time.(medie_stc) ./ 1e6
y_stc_memory = memory.(medie_stc) ./ (2 << 19)
y_stc_allocs = allocs.(medie_stc)

y_mut_times  = time.(medie_mut) ./ 1e6
y_mut_memory = memory.(medie_mut) ./ (2 << 19)
y_mut_allocs = allocs.(medie_mut)

pl1 = plot(precs, y_stc_times,
           yerror=time.(stdevs_stc) ./ 1e6,
           xlabel="Precision (bits)",
           ylabel="Time (ms)",
           label="Static",
           title="Execution time",
           marker = :o,
           linecolor=:auto,
           markerstrokecolor=:auto)
plot!(pl1, precs, y_mut_times,
      yerror=time.(stdevs_mut) ./ 1e6,
      label="Mutating",
      marker = :s,
      linecolor=:auto,
      markerstrokecolor=:auto)

pl2 = plot(precs, y_stc_memory,
           xlabel="Precision (bits)",
           ylabel="Memory (MB)",
           label="Static",
           title="Memory usage",
           marker = :o,
           linecolor=:auto,
           markerstrokecolor=:auto)
plot!(pl2, precs, y_mut_memory,
      label="Mutating",
      marker = :s,
      linecolor=:auto,
      markerstrokecolor=:auto)

pl3 = plot(precs, y_stc_allocs,
           xlabel="Precision (bits)",
           ylabel="Allocations (n°)",
           label="Static",
           title="Number of Allocations",
           marker = :o,
           linecolor=:auto,
           markerstrokecolor=:auto)
plot!(pl3, precs, y_mut_allocs,
      label="Mutating",
      marker = :s,
      linecolor=:auto,
      markerstrokecolor=:auto)


plot(pl1, pl2, pl3;
     layout = (2, 2),
     size = (1000, 900),
     plot_title = "expm2by2_full, expm2by2_full! cost comparison (BigFloat)",
     left_margin = 5Plots.mm,
     bottom_margin = 5Plots.mm)


## Fifth numerical test: random 2x2 triangular Float64 matrix
print("We now check whether the mutating version of `expm2by2_tri` "
 * "is more efficient than the original one, by testing on a random "
 * "triangular 2 by 2 matrix of Float64.\n"
 * "And since we're at it, we also assess the accuracy.\n")

B    = triu(rand(2,2));
expB = exp(B);      # true result

Y = expm2by2_tri(B);    # precompile
benchmark = @benchmark expm2by2_tri($B);
print("\tStatic (aka non mutating) benchmark:\n")
display(benchmark)
print("\n")

expm2by2_tri!(copy(B));     # precompile
benchmark_mut = @benchmark expm2by2_full!(B) setup=(B = copy($B));
print("\tMutating version benchmark:\n")
display(benchmark_mut)
print("\n")
benchmark_ratios(benchmark, benchmark_mut)

print("\tErrors:\n")
@printf("|| Y - exp(B) || / || exp(B) || = %.6g\n", rel_err(Y, expB))
expm2by2_full!(B);          # Overwrites B ← exp(B)
@printf("|| expmB - exp(B) || / || exp(B) || = %.6g\n", rel_err(B, expB))
@printf("|| expmB - Y || / (1 + || expmB || + || Y ||) = %.6g\n", sym_err(B, Y))


## Sixth numerical check: random 2x2 triangular BigFloat matrix
print("We now check the efficiency in arbitrary precision, "
 * "by running the analogous test as the previous, "
 * "but with a random upper triangular 2x2 BigFloat matrix.\n")

B = triu(rand(BigFloat, 2,2));

Y = expm2by2_tri(B);        # precompile
benchmark = @benchmark expm2by2_full($B);
print("\tStatic (aka non mutating) benchmark:\n")
display(benchmark)
print("\n")

expm2by2_full!(copy(B));    # precompile
benchmark_mut = @benchmark expm2by2_full!(B) setup=(B = copy($B));
print("\tMutating version benchmark:\n")
display(benchmark_mut)
print("\n")

benchmark_ratios(benchmark, benchmark_mut)

expm2by2_full!(B);          # Overwrites B ← exp(B)

print("\tErrors:\n")
@printf("|| expmB - Y || / (1 + || expmB || + || Y ||) = %.6g\n", sym_err(B, Y))