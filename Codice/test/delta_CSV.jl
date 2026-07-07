## Imports
using LinearAlgebra
using Random, Printf, CSV
using GenericSchur
using Revise

Revise.includet(joinpath(@__DIR__,"..","src","modules","MyMpExponential.jl"))
Revise.includet(joinpath(@__DIR__,"..","src","modules","MyHelper.jl"))
Revise.includet(joinpath(@__DIR__,"..","src","modules","MyMatrixGalleries.jl"))
using .MyMpExponential, .MyHelper, .MyMatrixGalleries

Random.seed!(42)

## Setup for Python, mpmath etc
const Y_TRUE_PREC = 1792

using PythonCall

mpmath = pyimport("mpmath")
numpy  = pyimport("numpy")

mpmath.mp.prec = Y_TRUE_PREC


## Define parameters and useful stuff

function python_matrix_from_julia(
    A::Matrix{T},
    target_precision=Y_TRUE_PREC
) where {T<:Real}
    if T == BigFloat
        A = setprecision(target_precision) do 
            BigFloat.(A)
        end
        Apy = pyrowlist(mpmath.mpf.(string.(A)))
    else 
        Apy = pyrowlist(mpmath.mpf.(A))
    end
    Apy = mpmath.matrix(Apy)

    return Apy
end

function python_matrix_from_julia(
    A::Matrix{T},
    target_precision=Y_TRUE_PREC
) where {T<:Complex}
    if real(T) == BigFloat
        A = setprecision(target_precision) do 
            complex.(BigFloat.(real(A)), BigFloat.(imag(A)))
        end
        Apy = pyrowlist(mpmath.mpc.(string.(real(A)), string.(imag(A))))
    else    
        Apy = pyrowlist(mpmath.mpc.(real(A), imag(A)))
    end
    Apy = mpmath.matrix(Apy)

    return Apy
end

function python_matrix_from_julia end


function julia_matrix_from_python(
    Apy,
    ::Val{false},    # Real version
    target_precision=Y_TRUE_PREC
)
    # ciascun elemento subisce la trasformazione:
    #   mp.mpf -> str (di python) -> String (di Julia) -> BigFloat
    A = setprecision(target_precision) do 
        map(x -> 
            string(mpmath.nstr(x, mpmath.mp.dps + 2)), 
        Apy) .|> x -> parse(BigFloat, x)
    end
    if A isa Vector
        n = sqrt(length(A)) |> Int
        A = reshape(A, (n,n))
        A = permutedims(A)
    end

    return A
end

function julia_matrix_from_python(
    Apy,
    ::Val{true},     # Complex version
    target_precision=Y_TRUE_PREC,
)
    # ciascun elemento subisce la trasformazione:
    #   mp.mpf -> str (di python) -> String (di Julia) -> BigFloat
    Areal, Aimag = setprecision(target_precision) do 
        Areal = map(x -> 
            string(mpmath.nstr(x.real, mpmath.mp.dps + 2)), 
        Apy) .|> x -> parse(BigFloat, x)
        Aimag = map(x -> 
            string(mpmath.nstr(x.imag, mpmath.mp.dps + 2)), 
        Apy) .|> x -> parse(BigFloat, x)
        Areal, Aimag
    end
    A = complex.(Areal, Aimag)
    if A isa Vector
        n = sqrt(length(A)) |> Int
        A = reshape(A, (n,n))
        A = permutedims(A)
    end

    return A
end

function julia_matrix_from_python end



function compute_refsol_python(
    A::Matrix{T},
    target_precision=Y_TRUE_PREC
) where {T}
    Apy = python_matrix_from_julia(A)

    Y_true_py = mpmath.expm(Apy)

    Y_true = julia_matrix_from_python(Y_true_py, Val(T <: Complex), target_precision)

    return Y_true, "mpmath_$(target_precision)"
end



############## Files per runnare esperimenti e scrivere su CSV ##############
csvfile = joinpath(@__DIR__, "..", "..", "Dati_benchmarks_et_al", "delta-bench-v0.1.5alpha-04_07.csv")

const CSV_HEADER = [
    "kind", "n", "eltype", "ishermitian",
    "approximant", "algorithm",
    "m", "s", "delta", "psi", "cond_q", "epsilon",
    "rel_err_F", "abs_err_1", "nrm1_Yb4Sq_true",
    "Ytrue_method", #"cond_expA_F",
    #"condA_1", "condA_2",
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
    #T_low = T <: Complex ? ComplexF64 : Float64
    #A_low = convert(Matrix{T_low}, A)

    # # conditionings 
    # cond_E = NaN
    # try 
    #     cond_E = cond_exp_exact(Matrix(A))
    #     println("DEBUG: cond_exp_exact returned: $cond_E")
    # catch e
    #     if e isa OutOfMemoryError
    #         println("DEBUG: OutOfMemoryError caught!")
    #     else
    #         println("DEBUG: Exception caught: $(typeof(e)) - $e")
    #     end
    #     cond_E = NaN
    # end
    # condA_1 = cond(A_low, 1)
    # condA_2 = cond(A_low, 2)

    ## Compute reference solution (this thing is a bit dirty)
    #Y_true, Y_true_method = compute_refsol_python(Matrix(A));

    # run configurations using exp_mp
    for approximant in [:diagonalcheap, :taylor]
        ALGS =  [:transfree, :complexschur]
        PRECS = [53, 256, 851]
        if T <: Real
            push!(ALGS, :realschur)
        end
        for alg in ALGS, wrk_p in PRECS
            print("Running: kind=$kind, n=$n, eltype=$(T), approximant=$approximant, algorithm=$alg, precision=$wrk_p\n")
            
            # Initialize with NaN values in case of error
            m = NaN
            s = NaN
            delta = NaN
            psi = NaN
            cond_q = NaN
            epsilon = NaN
            rel_err_F = NaN
            abs_err_1 = NaN
            Y = nothing
            Yb4Sq = nothing
            nrm1_Yb4Sq_true = NaN
            Ytrue_method = NaN
            
            try
                Y, _, params = exp_mp(A; approximant=approximant, algorithm=alg, working_precision=wrk_p)

                # get algorithm internal parameters
                m       = params.m
                s       = params.s
                delta   = params.delta
                psi     = params.psi
                cond_q  = params.cond_q
                epsilon = params.epsilon
                Yb4Sq   = params.Y

                # compute reference approx to exp(A/2^s)
                Yb4Sq_true, Ytrue_method = compute_refsol_python(Matrix(A)/2^s)
                #Yb4Sq_true = setprecision(2048) do 
                #    Yb4Sq_true = big.(Y_true)^(2.0^(-s))    # forse non conviene manco
                #end
                
                nrm1_Yb4Sq_true = opnorm(Yb4Sq_true, 1)

                # compute errors
                rel_err_F = rel_err(Yb4Sq, Yb4Sq_true)
                abs_err_1 = opnorm(Yb4Sq_true - Yb4Sq, 1)
            catch err
                @warn "exp_mp failed for kind=$kind, n=$n, eltype=$(T), approximant=$approximant, algorithm=$alg, precision=$wrk_p" exception=(err, catch_backtrace())
                #print("exp_mp broke. Precision is: $(precision(BigFloat))\n")
            finally
                setprecision(BigFloat, 256) # reset to default (emergency measure)
            end

            # write data
            row = [kind, string(n), string(T), ishermitian(A),
                   string(approximant), string(alg),
                   m, s, format_long_number(delta), format_long_number(psi), cond_q, format_long_number(epsilon),
                   format_long_number(rel_err_F), format_long_number(abs_err_1), format_long_number(nrm1_Yb4Sq_true),
                   Ytrue_method, #format_long_number(cond_E), 
                   #condA_1, condA_2,
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