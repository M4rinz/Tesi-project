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

function compute_Ytrue(A)
    n = LinearAlgebra.checksquare(A)

    Y_true = nothing 
    method = ""

    #print("Precision at the beginning = $(precision(BigFloat))\n")

    Y_true, method = setprecision(20*precision(BigFloat)) do 
        Abig = convert(Matrix{big(eltype(A))}, A)

        #print("eigensolving...\n")
        try
            d, V = eigen(Abig)
            #print("...finished eigensolving\n")

            condition_V = cond(V, 1) < 10^12
            good_approx = isapprox(V * Diagonal(d), Abig * V, rtol=n^2*eps(BigFloat))

            if condition_V && good_approx
                #print("computing exp\n")
                p = precision(BigFloat)
                return V * Diagonal(exp.(d)) / V, "diag_$p"
            elseif !condition_V
                @warn @sprintf("the eigenvector matrix is too ill-conditioned (%.4e)", cond(V, 1))
            else
                @warn "The diagonalization used for the reference solution is too inaccurate"
            end
        catch err
            @warn "Reference solution via diagonalization failed" exception=(err, catch_backtrace())
        end

        nothing, ""
    end #setprecision
    #print("Prec after the setprecision = $(precision(BigFloat))\n")
    if isnothing(Y_true)
        #print("exp_mp start...\n")
        wrk_prc = 20*precision(BigFloat)
        Y_true, _, _ = exp_mp(A, working_precision=wrk_prc)
        method = "exp_mp_$(wrk_prc)"
    end

    #print("precision before returning = $(precision(BigFloat))\n")

    return Y_true, method
end




############## Files per runnare esperimenti e scrivere su CSV ##############
csvfile = joinpath(@__DIR__, "..", "..", "Dati_benchmarks_et_al", "bench-v0.1.3alpha-20_05.csv")

const CSV_HEADER = [
    "kind", "n", "eltype", "ishermitian",
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

function run_and_record(
    kind::AbstractString, 
    A::AbstractMatrix,
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
        #print("precision before compute_Ytrue = $(precision(BigFloat))\n")
        Y_true, Ytrue_method = compute_Ytrue(A)
        #print("precision after compute_Ytrue = $(precision(BigFloat))\n")
    else 
        Ytrue_method = "given"
    end
    
    nrm1_Ytrue = opnorm(Y_true, 1)

    # errors of Base.exp
    rel_err_F = rel_err(Y_base, Y_true)
    abs_err_1 = opnorm(Y_base - Y_true, 1)

    # conditionings (oss: it's darn expensive!)
    cond_E = NaN
    try 
        cond_E = cond_exp_exact(Matrix(A))
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
    row = [kind, string(n), string(T), ishermitian(A),
            "scaling_and_squaring", "NaN",
            schur_time, alpha_time, eval_bound_time, 
            eval_pade_time, squaring_time, total_time, 
            NaN, NaN, NaN, NaN, NaN, eps(float(real(T_low))),
            format_long_number(rel_err_F), format_long_number(abs_err_1), format_long_number(nrm1_Ytrue),
            Ytrue_method, format_long_number(cond_E), 
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
            
            # Initialize with NaN values in case of error
            schur_time = NaN
            alpha_time = NaN
            eval_bound_time = NaN
            eval_pade_time = NaN
            squaring_time = NaN
            total_time = NaN
            m = NaN
            s = NaN
            delta = NaN
            psi = NaN
            cond_q = NaN
            epsilon = NaN
            rel_err_F = NaN
            abs_err_1 = NaN
            Y = nothing
            
            try
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
            catch err
                @warn "exp_mp failed for kind=$kind, n=$n, eltype=$(T), approximant=$approximant, algorithm=$alg, precision=$wrk_p" exception=(err, catch_backtrace())
                #print("exp_mp broke. Precision is: $(precision(BigFloat))\n")
            finally
                setprecision(BigFloat, 256) # reset to default (emergency measure)
            end

            # write data
            row = [kind, string(n), string(T), ishermitian(A),
                   string(approximant), string(alg),
                   schur_time, alpha_time, eval_bound_time, 
                   eval_pade_time, squaring_time, total_time, 
                   m, s, format_long_number(delta), format_long_number(psi), cond_q, format_long_number(epsilon),
                   format_long_number(rel_err_F), format_long_number(abs_err_1), format_long_number(nrm1_Ytrue),
                   Ytrue_method, format_long_number(cond_E), 
                   condA_1, condA_2,
                   wrk_p]
            write_row(csvfile, row)
        end
    end
end


## First experiment: Float64 random 
for n in [8, 24] #[16, 64, 256]
    A = rand(n,n);
    run_and_record("rand", A)
end

## Second experiment: BigFloat random
for n in [8, 24] #[16, 64, 256]
    A = rand(BigFloat, n,n)
    run_and_record("rand_big", A)
end


## Third experiment: Fasi's matrices
_, _, _, n_matrices = FasiMatrices(-42) 
for k=1:n_matrices
    A, Y_true, k, _ = FasiMatrices(k)
    run_and_record("Fasi_$k", A, Y_true)
end


## Fourth experiment: matrices from `expm_testmats` 
_, _, _, n_matrices = expm_testmats(-42)
for k=1:n_matrices
    if k in [11, 12, 32] # dipendono da n
        for n in [4, 24]
            A, Y_true, id, _ = expm_testmats(k, n)
            run_and_record(id, A, Y_true)
        end
    else 
        A, Y_true, id, _ = expm_testmats(k)
        run_and_record(id, A, Y_true)
    end
end


## fifth experiment: matrices from `gallery_getall_expm`
_, _, n_matrices = gallery_getall_expm(-42)
for k=1:n_matrices 
    A, id, _ = gallery_getall_expm(k)
    run_and_record(id, A, nothing)
end








## Fourth experiment: Hadamard + diagonalization (ComplexF64)
"""I TEST QUI SOTTO SONO DA AGGIUSTARE
"""
for n in [8, 32]   #[16, 64, 256]
    H = hadamard(n)
    H = Matrix{Float64}(H) / sqrt(n)
    D = Diagonal(100rand(n).-50 + 100im*rand(n).-50)
    A = H' * D * H
    #Y_true = H' * exp(D) * H
    run_and_record("hadamard_diag", A)
end

## Fifth experiment: Hadamard + diagonalization (BigFloat)
for n in [8, 32]   #[16, 64, 256]
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
