## Imports
using LinearAlgebra
using ChainRules
using Random, Printf, CSV
using BenchmarkTools
using GenericSchur
using Revise

Revise.includet(joinpath(@__DIR__,"..","src","modules","MyMpExponential.jl"))
Revise.includet(joinpath(@__DIR__,"..","src","modules","MyHelper.jl"))
Revise.includet(joinpath(@__DIR__,"..","src","modules","MyMatrixGalleries.jl"))
using .MyMpExponential, .MyHelper, .MyMatrixGalleries

Random.seed!(42)


## Define parameters and useful stuff


############## Files per runnare esperimenti e scrivere su CSV ##############
csvfile = joinpath(@__DIR__, "..", "..", "Dati_benchmarks_et_al", "bench-v0.1.2alpha-15_05.csv")

const CSV_HEADER = [
    "kind", "n", "eltype",
    "approximant", "algorithm",
    "schur_time", "alpha_time", "eval_bound_time",
    "eval_pade_time", "squaring_time", "total_time",
    "m", "s", "delta", "psi", "cond_q", "epsilon",
    "rel_err_F", "abs_err_1", "nrm1_Ytrue",
    "Ytrue_method", "cond_expA_F", 
    "condA_1", "condA_2",
    "precision"
]

ensure_csv_header(csvfile, CSV_HEADER)

format_long_number(x) = isfinite(x) ? @sprintf("%.16g", x) : string(x)

function compute_Ytrue(A)
    n = LinearAlgebra.checksquare(A)

    old_prec = precision(BigFloat)
    setprecision(20 * old_prec)

    Abig = convert(Matrix{big(eltype(A))}, A)
    Y_true, method = try
        d, V = eigen(Abig)
        # V malcondizionata è un problema, perché limita la qualità della nostra approx.
        # Potrebbe essere che A è "molto non-normale", o "quasi non diagonalizzabile".
        # In ogni caso, meglio non usare la diagonalizzazione
        cond(V, 1) < 10^10 ||    # 10 l'ho messo io arbitrariamente. 
            error("The eigenvector matrix is too ill-conditioned")
        isapprox(V * Diagonal(d), Abig * V, rtol=n*eps(BigFloat)) || 
            error("The diagonalization used for the reference solution is too inaccurate")
        V * Diagonal(exp.(d)) / V, "diag"
    catch
        Y_true, _, _ = exp_mp(Abig)
        Y_true, "exp_mp"
    end

    setprecision(old_prec)
    return Y_true, method
end





function run_and_record(
    kind::AbstractString, 
    A::AbstractMatrix;
    Y_true=nothing
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

    # compute reference solution, if not given
    if isnothing(Y_true)
        Y_true, method = compute_Ytrue(A)
    else 
        method = "given"
    end
    
    nrm1_Ytrue = opnorm(Y_true, 1)

    # errors of Base.exp
    rel_err_F = rel_err(Y_base, Y_true)
    abs_err_1 = opnorm(Y_base - Y_true, 1)

    # conditionings (oss: it's darn expensive!)
    cond_E = NaN
    try 
        cond_E = cond_exp_exact(A)
        println("DEBUG: cond_exp_exact returned: $cond_E")
    catch e
        if e isa OutOfMemoryError
            println("DEBUG: OutOfMemoryError caught!")
        else
            println("DEBUG: Exception caught: $(typeof(e)) - $e")
        end
        cond_E = NaN
    end
    condA_1 = cond(A_low, 1)
    condA_2 = cond(A_low, 2)

    # write data relative to Y_base
    row = [kind, string(n), string(T), 
            "scaling_and_squaring", "NaN",
            schur_time, alpha_time, eval_bound_time, 
            eval_pade_time, squaring_time, total_time, 
            NaN, NaN, NaN, NaN, NaN, eps(float(real(T_low))),
            format_long_number(rel_err_F), format_long_number(abs_err_1), format_long_number(nrm1_Ytrue),
            format_long_number(cond_E), 
            condA_1, condA_2,
            53]
    write_row(csvfile, row)

    # run configurations using exp_mp
    for approximant in (:diagonalcheap, :taylor)
        ALGS =  [:transfree, :complexschur]
        PRECS = [53, 256, 851]
        if T <: Real
            push!(ALGS, :realschur)
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
                   m, s, format_long_number(delta), format_long_number(psi), cond_q, format_long_number(epsilon),
                   format_long_number(rel_err_F), format_long_number(abs_err_1), format_long_number(nrm1_Ytrue),
                   format_long_number(cond_E), 
                   condA_1, condA_2,
                   wrk_p]
            write_row(csvfile, row)
        end
    end
end


## First experiment: Float64 random 
for n in [16, 64] #[16, 64, 256]
    A = rand(n,n);
    run_and_record("randn", A)
end

## Second experiment: BigFloat random
for n in [16, 64] #[16, 64, 256]
    A = rand(BigFloat, n,n)
    run_and_record("randn_big", A)
end


## Third experiment: Fasi's matrices
_, _, n_matrices = FasiMatrices(-42) 
for k=1:n_matrices
    A, k, _ = FasiMatrices(k)
    run_and_record("Fasi_$k", A)
end


## Fourth experiment: matrices from `expm_testmats` 
_, _, n_matrices = expm_testmats(-42)
for k=1:n_matrices
    if k in [11, 12, 32] # dipendono da n
        for n in [16, 64]
            A, Y_true, id, _ = expm_testmats(k, n)
            run_and_record(id, A, Y_true)
        end
    else 
        A, Y_true, id, _ = expm_testmats(k)
        run_and_record(id, A, Y_true)
    end
end





## Fourth experiment: Hadamard + diagonalization (ComplexF64)
"""I TEST QUI SOTTO SONO DA AGGIUSTARE
"""
for n in [16, 64]   #[16, 64, 256]
    H = hadamard(n)
    H = Matrix{Float64}(H) / sqrt(n)
    D = Diagonal(100rand(n).-50 + 100im*rand(n).-50)
    A = H' * D * H
    #Y_true = H' * exp(D) * H
    run_and_record("hadamard_diag", A)
end

## Fifth experiment: Hadamard + diagonalization (BigFloat)
for n in [16, 64]   #[16, 64, 256]
    H = hadamard(n)
    H = Matrix{BigFloat}(H) / sqrt(big(n))
    D = Diagonal(100rand(BigFloat, n).-50 + 100im*rand(BigFloat, n).-50)
    A = H' * D * H
    #Y_true = H' * exp(D) * H
    run_and_record("hadamard_diag_big", A)
end


## Sixth experiment: Hadamard + Jordan form (ComplexF64)
"""The Hadamard similarity + Jordan form test is disabled (for now)
"""
# for n in [16, 64]
#     H = hadamard(n)
#     H = Matrix{Float64}(H) / sqrt(n)
#     J = create_J(n)
#     A = H' * J * H
#     #Y_true = H' * exp(J) * H
#     run_and_record("hadamard_jord", A)
# end


## Fifth experiment: 
