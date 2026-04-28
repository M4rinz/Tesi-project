## Imports
using LinearAlgebra
using ChainRules
using Random, Printf
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


## Experiment 
n = 2 << 3;
A = rand(n,n);


csvfile = joinpath(@__DIR__, "..", "..", "Dati_benchmarks_et_al", "bench_29_03.csv")

function write_row(csvfile, row::Vector)
    open(csvfile, "a") do io
        println(io, join(row, ','))
    end
end


function run_and_record(kind::AbstractString, A::AbstractMatrix; Y_true=nothing)
    n = size(A,1)
    T = eltype(A)
    T_low = T <: Complex ? ComplexF64 : Float64
    A_low = convert(Matrix{T_low}, A)

    # baseline: compute reference with Base.exp (scaling & squaring)
    t_exp = @elapsed Y_ref = exp(A_low)
    schur_time = NaN
    alpha_time = NaN
    eval_bound_time = NaN
    eval_pade_time = NaN
    squaring_time = t_exp
    total_time = t_exp
    # compute relative error of Base.exp w.r.t. provided Y_true when available
    if isnothing(Y_true)
        rel_base = NaN
        Y_true = Y_ref
    else
        Y_ref_low = convert(Matrix{T_low}, Y_ref)
        Y_true_low = convert(Matrix{T_low}, Y_true)
        rel_base = rel_err(Y_ref_low, Y_true_low)
    end
    cond_E = compute_cond_exp(A)
    cond_A = cond(A_low)
    row = [kind, string(n), string(T), "scaling_and_squaring", "NaN",
           schur_time, alpha_time, eval_bound_time, eval_pade_time, squaring_time,
           total_time, rel_base, cond_E, cond_A]
    write_row(csvfile, row)

    # make a low-precision copy of Y_true for error computations
    Y_true_low = convert(Matrix{T_low}, Y_true)

    # run configurations using exp_mp
    for approximant in (:diagonalcheap, :diagonal, :taylor)
        ALGS = [:transfree, :complexschur]
        if T <: Real
            push!(ALGS, :realschur)
        end
        for alg in ALGS
            print("Running: kind=$kind, n=$n, eltype=$(T), approximant=$approximant, algorithm=$alg\n")
            Y, times = exp_mp(A; approximant=approximant, algorithm=alg)
            schur_time = times[1]
            alpha_time = times[2]
            eval_bound_time = times[3]
            eval_pade_time = times[4]
            squaring_time = times[5]
            total_time = sum(times)
            # ensure reference and result have comparable eltypes
            Y_low = convert(Matrix{T_low}, Y)
            rel = rel_err(Y_low, Y_true_low)
            cond_E = compute_cond_exp(A)
            cond_A = cond(A_low)

            row = [kind, string(n), string(T), string(approximant), string(alg),
                   schur_time, alpha_time, eval_bound_time, eval_pade_time, squaring_time,
                   total_time, rel, cond_E, cond_A]
            write_row(csvfile, row)
        end
    end
end


# First set of experiments
run_and_record("randn", A)

# Second: BigFloat random
n = 2 << 3
A = rand(BigFloat, n, n)
run_and_record("randn_big", A)

# Third: Hadamard diagonalization (complex)
n = 2 << 3
H = hadamard(n)
H = Matrix{Float64}(H) / sqrt(n)
D = Diagonal(100rand(n).-50 + 100im*rand(n).-50)
A = H' * D * H
Y_true = H' * exp.(D) * H
run_and_record("hadamard_complex", A; Y_true=Y_true)

# Fourth: Hadamard with BigFloat
n = 2 << 3
H = hadamard(n)
H = Matrix{BigFloat}(H) / sqrt(big(n))
D = Diagonal(100rand(BigFloat, n).-50 + 100im*rand(BigFloat, n).-50)
A = H' * D * H
Y_true = H' * exp.(D) * H
run_and_record("hadamard_complex_big", A; Y_true=Y_true)
