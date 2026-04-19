"""OSS: Questi tests sono stupidi perché le matrici che vengono 
        create con `rand` sono non negative (probabilmente), quindi 
        la funzione `normest1` non usa lo stimatore della norma!!!
"""
## Imports
using LinearAlgebra, Random, Printf
using Plots, BenchmarkTools
using Revise

Revise.includet(joinpath(@__DIR__,"..","..","..","src","modules","MyMpExponential.jl"))
Revise.includet(joinpath(@__DIR__,"..","..","..","src","modules","MyHelper.jl"))
using .MyMpExponential, .MyHelper

Random.seed!(42)


## Define parameters and useful stuff
function benchmark_ratios(
    benchNum, 
    benchDen;
    disp_names=("non_mutating", "mutating")
)
    function __return_params__(mean_bench)
        [mean_bench.time, mean_bench.gctime, mean_bench.memory, mean_bench.allocs]
    end
    meanNum = mean(benchNum)
    meanDen = mean(benchDen)

    @printf("""Ratios %s / %s for x in:
        time   = %.4g       gctime = %.4g
        memory = %.4g       allocs = %.4g
    
    """,
        disp_names[1], disp_names[2],
        (__return_params__(meanNum) ./ __return_params__(meanDen))...
    )
end

## First Benchmark
# idea: how the performance scales with the size `n`, keeping 
#       fixed the number of precomputed powers in `Apows`
print("Still TODO!\t(is it important though)?")

# idea: does the accuracy start to dwindle, as `n` increases?

## Second benchmark 
# idea: Let's fix a size and a power `d`. How does the cost change 
#       if we vary the number of elements in `Apows` ?
print("Still TODO!\t(is it important though)?")

# idea: does the accuracy start to dwindle, as `d` increases?


## Third benchmark
n = 50;
d = 11;
A   = 0.01rand(BigFloat, n,n);
A_f = Float64.(A);

# Arbitrary precision 
print("\tTesting accuracy and performance in arbitrary precision.\n")
A_struct = AandPowsStruct(A, [I(n), A, A^2], true);

Anorm_true = opnorm(A^d, 1);
Anorm_appx = normest1(d, A_struct);
@printf("| ‖Aᵈ‖ - normest1(d, A) | / ‖Aᵈ‖ = %.6g\n", rel_err(Anorm_appx, Anorm_true))

bench_Anorm_true = @benchmark opnorm($A^$d, 1);
print("\topnorm(Aᵈ, 1) benchmark:\n")
display(bench_Anorm_true)
print("\n")

bench_Anorm_appx = @benchmark normest1($d, $A_struct);
print("\tnormest1(d, A_struct) benchmark:\n")
display(bench_Anorm_appx)
print("\n")
benchmark_ratios(bench_Anorm_true, bench_Anorm_appx,
    disp_names=("exact_norm","approx_norm"))

# double precision
print("\tTesting accuracy and performance in double precision.\n")
A_f_struct = AandPowsStruct(A_f, [I(n), A_f, A_f^2], true);

A_f_norm_true = opnorm(A_f^d, 1);
A_f_norm_appx = normest1(d, A_struct);
@printf("| ‖(A_f)ᵈ‖ - normest1(d, A_f) | / ‖(A_f)ᵈ‖ = %.6g\n", rel_err(A_f_norm_appx, A_f_norm_true))

bench_A_f_norm_true = @benchmark opnorm($A_f^$d, 1);
print("\topnorm(Aᵈ, 1) benchmark:\n")
display(bench_A_f_norm_true)
print("\n")

bench_A_f_norm_appx = @benchmark normest1($d, $A_f_struct);
print("\tnormest1(d, A_struct) benchmark:\n")
display(bench_A_f_norm_appx)
print("\n")
benchmark_ratios(bench_A_f_norm_true, bench_A_f_norm_appx,
    disp_names=("exact_norm","approx_norm"))


# idea: how cheaper is the `Float64` version? What if we have to 
#       convert all matrices?


## Fourth test
# is it more convenient to keep the powers of float(A)
# or to convert when needed?
# In particular, is it cheaper to convert or to multiply?

#use data from previous experiment
convert(Matrix{Float64}, A_struct.powers[3]);
bench_convert = @benchmark convert(Matrix{Float64}, $(A_struct.powers[3]));
print("\tConvert(A², 'double') benchmark:\n")
display(bench_convert)
print("\n")

bench_mult_d = @benchmark $A_f^2;
print("\t(A_f)² benchmark:\n")
display(bench_mult_d)
print("\n")
benchmark_ratios(bench_convert, bench_mult_d,
    disp_names=("conversion", "multiplication"))

# Q: is it a fair comparison? It makes sense to try A² instead of higher powers
#    (because each insertion into `Apowers` is tied to one matmul)
#    However, do matmuls and conversions match 1-1? Depends on `d` right?
#    Il codice MATLAB va a pescare la matrice dall'equivalente di `Apows` e moltiplica
#    Quindi può fare meno conversioni di quante sono le inserzioni in `Apows`
#    Inoltre, in memoria ci sono meno matrici (perché io terrei una copia di `Apows` in doppia precisione)
#    Però, convertire costa di più...