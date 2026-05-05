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



############## Files per runnare esperimenti e scrivere su CSV ##############
csvfile = joinpath(@__DIR__, "..", "..", "Dati_benchmarks_et_al", "bench-v0.1.2alpha-5_05.csv")

const CSV_HEADER = [
    "kind", "n", "eltype",
    "approximant", "algorithm",
    "schur_time", "alpha_time", "eval_bound_time",
    "eval_pade_time", "squaring_time", "total_time",
    "m", "s", "delta", "psi", "cond_q", "epsilon",
    "rel_err_F", "abs_err_1", "nrm1_Ytrue",
    "cond_expA_F", 
    "condA_1", "condA_2"
]

ensure_csv_header(csvfile, CSV_HEADER)

function write_row(csvfile, row::Vector)
    open(csvfile, "a") do io
        println(io, join(row, ','))
    end
end

format_long_number(x) = isfinite(x) ? @sprintf("%.15g", x) : string(x)


function run_and_record(
    kind::AbstractString, 
    A::AbstractMatrix
)
    n = size(A,1)
    T = eltype(A)
    T_low = T <: Complex ? ComplexF64 : Float64
    A_low = convert(Matrix{T_low}, A)

    # compute baseline with Base.exp (scaling & squaring)
    t_exp = @elapsed Y_base = exp(A_low)

    schur_time = NaN
    alpha_time = NaN
    eval_bound_time = NaN
    eval_pade_time = NaN
    squaring_time = NaN
    total_time = t_exp

    # compute reference with diagonalization (and crossing our fingers)
    Y_true = setprecision(10 * precision(BigFloat)) do 
        d, V = eigen(A);
        V * diagm(d) / V ≈ A || @warn "The diagonalization used for the reference solution is too inaccurate."
        V * diagm(exp.(d)) / V
    end
    nrm1_Ytrue = opnorm(Y_true, 1)

    # errors of Base.exp
    rel_err_F = rel_err(Y_base, Y_true)
    abs_err_1 = opnorm(Y_base - Y_true, 1)

    # conditionings (oss: it's darn expensive!)
    cond_E = NaN
    try 
        cond_E = cond_exp_exact(A)
    catch OutOfMemoryError
        cond_E = NaN
    end
    condA_1 = cond(A_low, 1)
    condA_2 = cond(A_low, 2)

    # write data relative to Y_base
    row = [kind, string(n), string(T), 
            "scaling_and_squaring", "NaN",
            schur_time, alpha_time, eval_bound_time, 
            eval_pade_time, squaring_time, total_time, 
            NaN, NaN, NaN, NaN, NaN, eps(T_low),
            format_long_number(rel_err_F), format_long_number(abs_err_1), format_long_number(nrm1_Ytrue),
            format_long_number(cond_E), 
            condA_1, condA_2]
    write_row(csvfile, row)

    # run configurations using exp_mp
    for approximant in (:diagonalcheap, :taylor)
        ALGS =  [:transfree, :complexschur]
        PRECS = [256, 1024]
        if T <: Real
            push!(ALGS, :realschur)
        end
        if real(T) == Float64
            push!(PRECS, 128)
        else 
            push!(PRECS, 851)
        end
        for alg in ALGS, wrk_p in PRECS
            print("Running: kind=$kind, n=$n, eltype=$(T), approximant=$approximant, algorithm=$alg, precision=$wrk_p\n")
            t = @elapsed Y, times, params = exp_mp(A; approximant=approximant, algorithm=alg, working_precision=wrk_p)
            
            # get times
            schur_time = times[1]
            alpha_time = times[2]
            eval_bound_time = times[3]
            eval_pade_time = times[4]
            squaring_time = times[5]
            total_time = t

            # get algorithm internal parameters
            m       = params.m
            s       = params.s
            delta   = params.delta
            psi     = params.psi
            cond_q  = params.cond_q
            epsilon = params.epsilon

            # compute errors
            rel_err_F = rel_err(Y, Y_true)
            abs_err_1 = opnorm(Y_true - Y, 1)

            # write data
            row = [kind, string(n), string(T), 
                   string(approximant), string(alg),
                   schur_time, alpha_time, eval_bound_time, 
                   eval_pade_time, squaring_time, total_time, 
                   m, s, format_long_number(delta), psi, cond_q, format_long_number(epsilon),
                   format_long_number(rel_err_F), format_long_number(abs_err_1), format_long_number(nrm1_Ytrue),
                   format_long_number(cond_E), 
                   condA_1, condA_2]
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


"""I TEST QUI SOTTO SONO DA AGGIUSTARE
"""

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
