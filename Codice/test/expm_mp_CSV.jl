## Imports
using LinearAlgebra
using ChainRules
using Random, Printf, CSV
using BenchmarkTools
using Revise

Revise.includet(joinpath(@__DIR__,"..","src","modules","MyMpExponential.jl"))
Revise.includet(joinpath(@__DIR__,"..","src","modules","MyHelper.jl"))
using .MyMpExponential, .MyHelper

Random.seed!(42)


## Define parameters and useful stuff
function hadamard(n::Int)
    ispow2(n) || throw(ArgumentError("n must be a power of 2"))

    H = ones(Int, 1, 1)
    while size(H,1) < n
        H = [ H   H;
              H  -H ]
    end
    return H
end

function compute_cond_exp(A)
    n = LinearAlgebra.checksquare(A)
    T = eltype(A)
    T_low = T <: Complex ? ComplexF64 : Float64

    A_low = convert(AbstractMatrix{T_low}, A)
    Y_true, exp_pullback = ChainRules.rrule(exp, A_low)

    K = construct_full_jacobian(x -> exp_pullback(x)[2], n, T_low)

    #κ_exp(A) = (‖K‖₂⋅‖A‖_F) / ‖exp(A)‖_F
    opnorm(K, 2) * norm(A) / norm(Y_true)
end

############## Files per runnare esperimenti e scrivere su CSV ##############
csvfile = joinpath(@__DIR__, "..", "..", "Dati_benchmarks_et_al", "bench-v0.1.2alpha-29_03.csv")

const CSV_HEADER = [
    "kind", "n", "eltype",
    "approximant", "algorithm",
    "schur_time", "alpha_time", "eval_bound_time",
    "eval_pade_time", "squaring_time", "total_time",
    "rel_err",
    "cond_exp_A", "cond_A",
]

function ensure_csv_header(csvfile, header::Vector{String})
    header_line = join(header, ",")

    if !isfile(csvfile)
        open(csvfile, "w") do io
            println(io, header_line)
        end
        return
    end

    contents = read(csvfile, String)
    if isempty(contents)
        open(csvfile, "w") do io
            println(io, header_line)
        end
        return
    end

    lines = readlines(IOBuffer(contents))

    first_row = strip.(split(first(lines), ','))
    if first_row != header || !endswith(contents, "\n")
        open(csvfile, "w") do io
            println(io, header_line)
            for line in lines
                println(io, line)
            end
        end
    end
end

ensure_csv_header(csvfile, CSV_HEADER)

function write_row(csvfile, row::Vector)
    open(csvfile, "a") do io
        println(io, join(row, ','))
    end
end

format_long_number(x) = isfinite(x) ? @sprintf("%.15g", x) : string(x)

function run_and_record(
    kind::AbstractString, 
    A::AbstractMatrix; 
    Y_true=nothing
)
    n = size(A,1)
    T = eltype(A)
    T_low = T <: Complex ? ComplexF64 : Float64
    A_low = convert(Matrix{T_low}, A)

    # baseline: compute reference with Base.exp (scaling & squaring)
    t_exp = @elapsed Y_base = exp(A_low)

    schur_time = NaN
    alpha_time = NaN
    eval_bound_time = NaN
    eval_pade_time = NaN
    squaring_time = NaN
    total_time = t_exp

    # compute relative error of Base.exp w.r.t. provided Y_true when available
    if isnothing(Y_true)
        # if Y_true is not provided we use Y_base as reference solution
        rel_base = NaN
        Y_true = Y_base 
    else
        #Y_ref_low  = convert(Matrix{T_low}, Y_ref)
        #Y_true_low = convert(Matrix{T_low}, Y_true)
        #rel_base = rel_err(Y_ref_low, Y_true_low)
        rel_base = rel_err(Y_base, Y_true)
    end
    # conditionings (oss: it's darn expensive!)
    cond_E = NaN
    try 
        cond_E = compute_cond_exp(A)
    catch OutOfMemoryError
        cond_E = NaN
    end
    cond_A = cond(A_low)
    # write data relative to Y_base
    row = [kind, string(n), string(T), "scaling_and_squaring", "NaN",
           schur_time, alpha_time, eval_bound_time, eval_pade_time, squaring_time,
            total_time, format_long_number(rel_base), format_long_number(cond_E), cond_A]
    write_row(csvfile, row)

    ## make a low-precision copy of Y_true for error computations
    #Y_true_low = convert(Matrix{T_low}, Y_true)

    # run configurations using exp_mp
    for approximant in (:diagonalcheap, :diagonal, :taylor)
        ALGS = [:transfree, :complexschur]
        if T <: Real
            push!(ALGS, :realschur)
        end
        for alg in ALGS
            print("Running: kind=$kind, n=$n, eltype=$(T), approximant=$approximant, algorithm=$alg\n")
            t = @elapsed Y, times = exp_mp(A; approximant=approximant, algorithm=alg)
            schur_time = times[1]
            alpha_time = times[2]
            eval_bound_time = times[3]
            eval_pade_time = times[4]
            squaring_time = times[5]
            total_time = t
            ## ensure reference and result have comparable eltypes
            #Y_low = convert(Matrix{T_low}, Y)
            err_rel = rel_err(Y, Y_true)

            row = [kind, string(n), string(T), 
                   string(approximant), string(alg),
                   schur_time, alpha_time, eval_bound_time, eval_pade_time, squaring_time,
                   total_time, format_rel_err(err_rel), cond_E, cond_A]
            write_row(csvfile, row)
        end
    end
end


## First experiment: Float64 random 
for n in [16, 64, 256]
    A = rand(n,n);
    run_and_record("randn", A)
end

## Second experiment: BigFloat random
for n in [16, 64, 256]
    A = rand(BigFloat, n,n)
    run_and_record("randn_big", A)
end

## Third experiment: Hadamard diagonalization (ComplexF64)
for n in [16, 64]   #[16, 64, 256]
    H = hadamard(n)
    H = Matrix{Float64}(H) / sqrt(n)
    D = Diagonal(100rand(n).-50 + 100im*rand(n).-50)
    A = H' * D * H
    Y_true = H' * exp(D) * H
    run_and_record("hadamard_complex", A; Y_true=Y_true)
end

## Fourth experiment: Hadamard with BigFloat
for n in [16, 64]   #[16, 64, 256]
    H = hadamard(n)
    H = Matrix{BigFloat}(H) / sqrt(big(n))
    D = Diagonal(100rand(BigFloat, n).-50 + 100im*rand(BigFloat, n).-50)
    A = H' * D * H
    Y_true = H' * exp(D) * H
    run_and_record("hadamard_complex_big", A; Y_true=Y_true)
end
