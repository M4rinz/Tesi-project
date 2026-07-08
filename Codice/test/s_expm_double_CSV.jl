## Imports
using LinearAlgebra
using Random, Printf, CSV
using GenericSchur
using Revise

Revise.includet(joinpath(@__DIR__,"..","src","modules","MyMpExponential.jl"))
Revise.includet(joinpath(@__DIR__,"..","src","modules","MyBaseExponential.jl"))
Revise.includet(joinpath(@__DIR__,"..","src","modules","MyHelper.jl"))
Revise.includet(joinpath(@__DIR__,"..","src","modules","MyMatrixGalleries.jl"))
using .MyMpExponential, .MyBaseExponential, .MyHelper, .MyMatrixGalleries

Random.seed!(42)

## Setup for Python, mpmath etc
const Y_TRUE_PREC = 1792


############## Files per runnare esperimenti e scrivere su CSV ##############
csvfile = joinpath(@__DIR__, "..", "..", "Dati_benchmarks_et_al", "s-bench-v0.1.5alpha-08_07.csv")

const CSV_HEADER = [
    "kind", "n", "eltype", "ishermitian",
    "approximant", "algorithm",
    "m", "s", "delta", "psi", "cond_q", "epsilon",
    "m_base", "s_base",
    "precision"
]

ensure_csv_header(csvfile, CSV_HEADER)

format_long_number(x) = isfinite(x) ? @sprintf("%.16g", x) : string(x)

function run_and_record(
    kind::AbstractString, 
    A::AbstractMatrix
)
    n = size(A,1)
    T = eltype(A)
    T_low = T <: Complex ? ComplexF64 : Float64
    A_low = convert(Matrix{T_low}, A)

    for approximant in [:diagonalcheap, :taylor]
        ALGS =  [:transfree, :complexschur]
        if T <: Real
            push!(ALGS, :realschur)
        end
        wrk_p = 53
        for alg in ALGS
            print("Running: kind=$kind, n=$n, eltype=$(T), approximant=$approximant, algorithm=$alg, precision=$wrk_p\n")
            
            # Initialize with NaN values in case of error
            m = NaN
            s = NaN
            m_base = NaN
            s_base = NaN
            delta = NaN
            psi = NaN
            cond_q = NaN
            epsilon = NaN
            
            try
                _, _, params = exp_mp(A; approximant=approximant, algorithm=alg, working_precision=53)
                
                m       = params.m
                s       = params.s
                delta   = params.delta
                psi     = params.psi
                cond_q  = params.cond_q
                epsilon = params.epsilon

                _, m_base, s_base = my_Base_exp(A_low)
            catch err
                @warn "exp_mp failed for kind=$kind, n=$n, eltype=$(T), approximant=$approximant, algorithm=$alg, precision=$wrk_p" exception=(err, catch_backtrace())
            finally 
                setprecision(BigFloat, 256) # reset to default (emergency measure)
            end

            # write data
            row = [kind, string(n), string(T), ishermitian(A),
                    string(approximant), string(alg),
                    m, s, format_long_number(delta), format_long_number(psi), cond_q, format_long_number(epsilon),
                    m_base, s_base,
                    wrk_p]
            write_row(csvfile, row)
        end
    end

end

# ## First experiment: Float64 random 
# for n in [8, 24] #[16, 64, 256]
#     A = rand(n,n);
#     run_and_record("rand_$n", A)
# end
# print("randn experiment completed\n")

# ## Second experiment: BigFloat random
# for n in [8, 24] #[16, 64, 256]
#     A = rand(BigFloat, n,n)
#     run_and_record("rand_$(n)_big", A)
# end
# print("randn_big experiment completed\n")

# ## Third experiment: Fasi's matrices
# _, _, _, n_matrices = FasiMatrices(-42) 
# for k=1:n_matrices
#     A, _, k, _ = FasiMatrices(k, Y_true_precision=Y_TRUE_PREC)
#     run_and_record("Fasi_$k", A)
# end
# print("FasiMatrices experiment completed\n")

# ## Fourth experiment: matrices from `expm_testmats` 
# _, _, _, n_matrices = expm_testmats(-42)
# for k=1:n_matrices
#     if k in [11, 12, 32] # dipendono da n
#         for n in [4, 24]
#             A, _, id, _ = expm_testmats(k, n, Y_true_precision=Y_TRUE_PREC)
#             run_and_record(id, A)
#         end
#     else 
#         A, _, id, _ = expm_testmats(k, Y_true_precision=Y_TRUE_PREC)
#         run_and_record(id, A)
#     end
# end
# print("expm_testmats experiment completed\n")

# ## fifth experiment: matrices from `gallery_getall_expm`
# _, _, n_matrices = gallery_getall_expm(-42)
# for k=1:n_matrices 
#     A, id, _ = gallery_getall_expm(k)
#     run_and_record(id, A)
# end
# print("gallery_getall_expm experiment completed\n")

# ## Sixth experiment: Hadamard + diagonalization (ComplexF64)
# for n in [8]   #[16, 64, 256]
#     H = hadamard(n)
#     H = Matrix{Float64}(H) / sqrt(n)
#     D = Diagonal(100rand(n).-50 + 100im*rand(n).-50)
#     A = H' * D * H
#     run_and_record("hadamard_diag_$n", A)
# end
# print("hadamard_diag experiment completed\n")

# ## Seventh experiment: Hadamard + diagonalization (BigFloat)
# for n in [8]   #[16, 64, 256]
#     H = hadamard(n)
#     H = Matrix{BigFloat}(H) / sqrt(big(n))
#     D = Diagonal(100rand(BigFloat, n).-50 + 100im*rand(BigFloat, n).-50)
#     A = H' * D * H
#     run_and_record("hadamard_diag_big_$n", A)
# end
# print("hadamard_diag_big experiment completed\n")

# ## 8th experiment: Hadamard + Jordan form (ComplexF64)
# for n in [8]     # [16, 64, 256]
#     H = hadamard(n)
#     H = Matrix{Float64}(H) / sqrt(n)
#     J = create_J(n)
#     A = H' * J * H
#     run_and_record("hadamard_jord_$n", A)
# end
# print("hadamard_jord experiment completed \n")

# ## Ninth experiment: Hadamard + Jordan form (ComplexF64)
# for n in [8]     # [16, 64, 256]
#     H = hadamard(n)
#     H = Matrix{Float64}(H) / sqrt(n)
#     J = create_J(n, BigFloat)
#     A = H' * J * H
#     run_and_record("hadamard_jord_big_$n", A)
# end
# print("hadamard_jord_big experiment completed \n")