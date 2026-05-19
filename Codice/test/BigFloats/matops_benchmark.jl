using LinearAlgebra, Random, BenchmarkTools

Random.seed!(42)


n = 2 << 6

A = rand(BigFloat, n,n);
B = rand(BigFloat, size(A)...);

Af = Float64.(A);
Bf = Float64.(A);

# same data as Af and Bf, but different datatype
Af_B = BigFloat.(Af);
Bf_B = BigFloat.(Bf);


## matrix multiplications

print("A * B\n")
A * B;          # precompile
@benchmark $A * $B

print("float(A) * float(B)\n")
prod_d  = Af * Bf;        
@benchmark $Af * $Bf

print("big(float(A)) * big(float(B))\n")
prod_bf = Af_B * Bf_B;    
@benchmark $Af_B * $Bf_B

relerr = norm(prod_d - prod_bf) / norm(prod_d);
print("|| Af * Bf - Af_B * Bf_B || / || Af * B_f || = $(relerr)\n")
relerr = norm(prod_d - Float64.(prod_bf)) / norm(prod_d);
print("|| Af * B_f - double(Af_B * Bf_B) || / || Af * B_f || = $(relerr)\n")

msg_moral = """Moral of the story:
if the matrices are of type `Float64`, promoting them is a stupid idea.
The data is the same, but the operations are way more expensive.
""";
print(msg_moral)

convert(Matrix{Float64}, A);    
@benchmark convert(Matrix{Float64}, $A)

msg = """Moral of the story:
Converting a `Matrix{BigFloat}` to one of `BigFloat`s has a cost as well,
albeit small.
""";
print(msg * "\n")

## matrix sums
print("Let's move on to sums.\n\n")

print("A + B\n")
A + B;  # precompile
@benchmark $A + $B

print("float(A) + float(B)\n")
Af + Bf;
@benchmark $Af + $Bf

print("big(float(A)) * big(float(B))\n")
Af_B + Bf_B; 
@benchmark $Af_B + $Bf_B

print(msg_moral)


## Does the cost of the product between BigFloat scalars depend on the precision?
using Plots, Printf

precs = range(53, 1024; length=10) .|> Int64 ∘ floor

medie, stdevs = [], []

for pr in precs 
      setprecision(pr) do 
            a = rand(BigFloat)
            b = rand(BigFloat)
            a + b;      # precompile
            benchmark = @benchmark $a + $b;

            push!(medie, mean(benchmark));
            push!(stdevs, std(benchmark));
      end
end 

y_times  = time.(medie) #./ 1e6
y_memory = memory.(medie) #./ (2 << 19)
y_allocs = allocs.(medie)

X = [ones(length(precs)) precs];
β_time   = X \ y_times;
β_memory = X \ y_memory;
β_allocs = X \ y_allocs;

pl1 = plot(precs, y_times,
     yerror=time.(stdevs) ,#./ 1e6,
     xlabel="Precision (bits)",
     ylabel="Time (ns)",
     label="Measured time",
     title="Execution time",
     marker = :o)
plot!(pl1, precs, X * β_time,
      label= @sprintf("Best fit (slope = %.2f)", β_time[2]),
      linewidth=2)

pl2 = plot(precs, y_memory,
           xlabel="Precision (bits)",
           ylabel="Memory (Bytes)",
           label="Measured memory usage",
           title="Memory usage",
           marker=:o)
plot!(pl2, precs, X * β_memory,
      label= @sprintf("Best fit (slope = %.2f)", β_memory[2]),
      linewidth=2)

pl3 = plot(precs, y_allocs,
           xlabel="Precision (bits)",
           ylabel="Allocations (n°)",
           label="Measured allocations",
           title="Number of Allocations",
           marker=:o)
plot!(pl3, precs, X * β_allocs,
      label= @sprintf("Best fit (slope = %.2f)", β_allocs[2]),
      linewidth=2)

plot(pl1, pl2, pl3;
     layout = (2, 2),
     size = (1000, 900),
     plot_title = "BigFloat sum cost",
     left_margin = 5Plots.mm,
     bottom_margin = 5Plots.mm)


## Does the matmul cost depend on the precision?

precs = range(53, 1024; length=10) .|> Int64 ∘ floor

medie, stdevs = [], []

for pr in precs
    setprecision(pr) do 
        Ai = rand(BigFloat, n,n);
        Bi = rand(BigFloat, n,n);
        Ai * Bi;    # precompile
        benchmark = @benchmark $Ai * $Bi;

        push!(medie, mean(benchmark));
        push!(stdevs, std(benchmark));
    end
end 

y_times  = time.(medie) ./ 1e6
y_memory = memory.(medie) ./ (2 << 19)
y_allocs = allocs.(medie)

X = [ones(length(precs)) precs];
β_time   = X \ y_times;
β_memory = X \ y_memory;
β_allocs = X \ y_allocs;

pl1 = plot(precs, y_times,
     yerror=time.(stdevs) ./ 1e6,
     xlabel="Precision (bits)",
     ylabel="Time (ms)",
     label="Measured time",
     title="Execution time",
     marker = :o)
plot!(pl1, precs, X * β_time,
      label= @sprintf("Best fit (slope = %.2f)", β_time[2]),
      linewidth=2)

pl2 = plot(precs, y_memory,
           xlabel="Precision (bits)",
           ylabel="Memory (MB)",
           label="Measured memory usage",
           title="Memory usage",
           marker=:o)
plot!(pl2, precs, X * β_memory,
      label= @sprintf("Best fit (slope = %.2f)", β_memory[2]),
      linewidth=2)

pl3 = plot(precs, y_allocs,
           xlabel="Precision (bits)",
           ylabel="Allocations (n°)",
           label="Measured allocations",
           title="Number of Allocations",
           marker=:o)
plot!(pl3, precs, X * β_allocs,
      label= @sprintf("Best fit (slope = %.2f)", β_allocs[2]),
      linewidth=2)

plot(pl1, pl2, pl3;
     layout = (2, 2),
     size = (1000, 900),
     plot_title = "Matrix multiplication (BigFloat) cost",
     left_margin = 5Plots.mm,
     bottom_margin = 5Plots.mm)

print("""Moral of the story:
the cost of matrix multiplication scales linearly (at a certain rate) with the precision.
At least on my computer.
""")


## Does the matrix addition cost depend on the precision?
print("If the matrix addition cost depends on the precision? I guess so!\n")