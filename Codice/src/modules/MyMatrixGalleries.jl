module MyMatrixGalleries

using LinearAlgebra
using TypedMatrices
using Polynomials
using Random
using SparseArrays


############### Costruire matrici programmaticamente ###############
function hadamard(n::Int)
    ispow2(n) || throw(ArgumentError("n must be a power of 2"))

    H = ones(Int, 1, 1)
    while size(H,1) < n
        H = [ H   H;
              H  -H ]
    end
    return H
end

function create_J(n::Int, ::Type{T}=Float64) where {T<:AbstractFloat}
    n > 0 || throw(ArgumentError("n must be positive"))

    J = zeros(Complex{T}, n, n)
    first = 1
    remaining = n

    while remaining > 0
        block_size = rand(1:remaining)
        λ = complex(T(100) * rand(T) - T(50), T(100) * rand(T) - T(50))

        last = first + block_size - 1
        for i in first:last
            J[i, i] = λ
            if i < last
                J[i, i + 1] = one(T)
            end
        end

        first = last + 1
        remaining -= block_size
    end

    return J
end

export hadamard, create_J



function Toeppd(n, m=n, x=rand(m,1), theta=rand(m,1))
    A = zeros(n,n)
    for i=1:m
        T = collect(1:n) .- collect(1:n)'
        T = cos.(2*π*theta[i]*T)
        A += x[i]*T
    end
    return A
end

function Toeppen(n, a=1, b=-10, c=0, d=10, e=1)
    A = diagm(
        -2 => a * ones(n-2),
        -1 => b * ones(n-1),
         0 => c * ones(n),
         1 => d * ones(n-1),
         2 => e * ones(n-2)
    )
    return A
end

function create_C_condex(n; k=10)
    C = zeros(n,n)
    # create random vector on the unit simplex
    theta = rand(k)
    theta /= sum(theta)

    # convex combination of permutation matrices
    # incrociamo le dita e speriamo che la stessa 
    # permutazione non capiti più volte
    for i = 1:k
        rp = randperm(n)
        P = Matrix(I(n))[rp,:]
        C += theta[i]*P
    end
    # ora C è una [matrice doppiamente stocastica]
    # (https://en.wikipedia.org/wiki/Doubly_stochastic_matrix)

    return C-I(n)
end

function create_V_ChebVand(x)
    n = length(x)  
    V = zeros(n,n)

    e_i = [1, zeros(n-1)...]
    for i = 1:n
        # V[i,j] = T_{i-1}(x[j])
        V[i,:] = ChebyshevT(e_i).(x)'
        circshift!(e_i, 1)
    end

    return V
end

function create_H_house(x)   
    # Ensure x is a column vector
    if isa(x, Number)
        x = [x]
    else
        x = vec(x)
    end
    
    v = [sign(x[1])*norm(x)+x[1]; x[2:end]]
    
    beta = 2.0 / dot(v, v)
    
    return v, beta
end



############### Funzioni usate da Fasi e Higham ###############

"""
    A, Y_true, id, n_matrices = FasiMatrices(k, T=Float64)

Returns `A`, the `k`th matrix in a test set that has been used 
to evaluate algorithm in [^hf19_mpexpm]. 
`n_matrices` is the total number of matrices in such set (i.e. 18).
`id` is an identifier of `A`: for this function, it coincides with `k`.
For some matrices, the true exponential of ``A`` is available: this is `Y_true`.
In the other cases, `Y_true = nothing`.

The original MATLAB code can be found 
[in this repo](https://github.com/mfasi/mpexpm/blob/master/include/mymatrices.m)

All credits to the original authors (N. J. Higham and M. Fasi).

# References 
> [^hf19_mpexpm] N. J. Higham and M. Fasi, An Arbitrary Precision Scaling and Squaring Algorithm for the Matrix Exponential
> SIAM J. Matrix Anal. Appl., Vol. 40.4 (2019), pp.1233-1256.
> [doi: 10.1137/18M1228876](https://doi.org/10.1137/18M1228876)
"""
function FasiMatrices(
    k::Integer,
    n::Integer=10,
    T::DataType=Float64;
    Y_true_precision=2048
)
    n_mats = 18

    Y_true = nothing
    
    if k < 1 
        return [], Y_true, k, n_mats
    end

    epsilon = eps(real(T))

    setprecision(Y_true_precision) do 
    if k == 1
        A = Tridiagonal([-1], [-2, -2], [1])
    elseif k == 2
        A = [1 1; 1 1 + 10 * epsilon]
    elseif k == 3
        A = [10 0 0; 0 1 1; 0 1 1 + 10 * epsilon]
    elseif k == 4
        A      = zeros(10, 10)
        Y_true = zeros(BigFloat, 10, 10)

        accum = 0
        for i = 1:4
            A[accum+1:accum+i,accum+1:accum+i] = Tridiagonal(0*ones(i-1),i*ones(i),ones(i-1))
            # we use the fact that is a Jordan block
            for k = 1:i 
                expλ = exp(big(A[accum+k,accum+k]))
                #Y_true[accum+k, accum+k] = 
                row = [1/factorial(big.(j)) for j=0:(i-k)]
                Y_true[accum+k, accum+k:accum+i] = expλ * row
            end
            accum += i
        end
    elseif k == 5
        n_local = 10
        D = Diagonal([zeros(n_local - 1); 1])
        Q = triu(ones(n_local, n_local))
        A = Q * D / Q
        #Y_true = Q * exp(big.(D)) / Q
    elseif k == 6
        A = diagm([0, 1, 1e6])
        Y_true = exp(Diagonal([0, 1, big(1e6)]))
    elseif k == 7
        A = [1e-4 0; 0 1e4]
    elseif k == 8
        A = ones(2, 2)
    elseif k == 9
        n_local = 10
        A = Toeplitz(vcat(16-3im, (4+3im)/8, zeros(n_local - 2)),
                     vcat(16-3im, -5, zeros(n_local - 2)))
    elseif k == 10
        A = [1 1; 0 1e2]
        Y_true = [exp(big(1)) (exp(big(1e2))-exp(big(1)))/99 
                    0           exp(big(1e2))]
    elseif k == 11
        A = [1 1e3; 1e3 1]
    elseif k == 12
        A = [1 2 3; 1 2 3; 1 2 3]
        divdiff = (expm1(big(3)+2+1)) / (3+2+1)
        Y_true = I(3) + divdiff*big.(A)
    elseif k == 13
        t = -π / 2
        A = [cos(t) -sin(t); sin(t) cos(t)]
    elseif k == 14
        v = I(n)[:,1]
        A = I(n) - v * v'
    elseif k == 15
        A = [100 2 3; 4 5 6; 7 8 100]
    elseif k == 16
        A = [1 1 1; 1 1 1 + 10 * epsilon; 1 1 1 + 100 * epsilon]
    elseif k == 17
        A = [1 2 3; 4 5 6; 7 8 1e2]
    elseif k == 18
        A = [1 1 1 0.1; 1 1 1 10 * epsilon; 1 1 1 100 * epsilon; 1 1 1 1000 * epsilon]
    else 
        #error("k can be at most $(n_mats)")
        return [], Y_true, k, n_mats
    end
    return A, Y_true, k, n_mats
    end # setprecision
end


"""
    A, Y_true, id, n_mats = expm_testmats(k, n=10)

Returns the `k`th matrix in a test set test set that has been used 
to evaluate algorithm in [^hf19_mpexpm]. Specifically, this is the same as 
`expm_testmats` in the [repo of [^hf19_mpexpm]](https://github.com/mfasi/mpexpm).
`n_matrices` is the total number of matrices in such set (i.e. 38).
`id` is an identifier of `A`: for this function, it is given by the comment
left by the original authors.
For some matrices, the true exponential of ``A`` is available: this is `Y_true`.
In the other cases, `Y_true = nothing`.

The original MATLAB code can be found
[at this link](https://github.com/mfasi/mpexpm/blob/master/include/expm_testmats.m)

All credits to the original authors (N. J. Higham, M. Fasi and A. Al Mohy).


# References 
> [^hf19_mpexpm] N. J. Higham and M. Fasi, An Arbitrary Precision Scaling and Squaring Algorithm for the Matrix Exponential
> SIAM J. Matrix Anal. Appl., Vol. 40.4 (2019), pp.1233-1256.
> [doi: 10.1137/18M1228876](https://doi.org/10.1137/18M1228876)
"""
function expm_testmats(
    k::Integer, 
    n::Integer=10;
    Y_true_precision=2048
)
    n_mats = 38

    Y_true = nothing

    if k < 1
        return [], Y_true, "EMPTY", n_mats
    end

    setprecision(Y_true_precision) do 
    if k == 1
        # \cite[Test 1]{ward77}.
        id = "ward77_test1"
        A = [4 2 0; 1 4 1; 1 1 4]
        V = [-2 4 1; 1 -3 1; 1 0 1]
        #J = [3 1 0; 0 3 0; 0 0 6]
        #Mi sa che è meglio se non la restituisco...
        el = exp(big(3))
        expJ = [el el 0
                0  el 0
                0  0  exp(big(6))]
        Vb = big.(V)
        Y_true = Vb * expJ / Vb
    elseif k == 2
        # \cite[Test 2]{ward77}.
        id = "ward77_test2"
        A = [29.87942128909879     .7815750847907159 -2.289519314033932
             .7815750847907159   25.72656945571064    8.680737820540137
             -2.289519314033932   8.680737820540137  34.39400925519054]
    elseif k == 3
        # \cite[Test 3]{ward77}.
        id = "ward77_test3"
        A = [-131 19 18;
             -390 56 54;
             -387 57 52]
        #V = [1 1 1; 3 3 4; 3 4 3]
        #D = Diagonal([-20, -2, -1])
        #Y_true = V * exp(big.(D)) / V
    elseif k == 4
        # \cite[Test 4]{ward77}.
        id = "ward77_test4"
        A = Forsythe(10, 1e-10, 0)
    elseif k == 5
        # \cite[p. 370]{naha95}.
        id = "naha95_p370"
        T = [1 10 100; 1 9 100; 1 11 99]
        D = [0.001 0 0; 0 1 0; 0 0 100] 
        A = T * D / T
        #Y_true = T * exp(big.(Diagonal(D))) / T
    elseif k == 6
        # \cite[Ex.~2]{kela98}.
        id = "kela98_ex2"
        A = [0.1 1e6; 0 0.1]
        el = exp(big(0.1))
        Y_true = [el 1e6*el; 0 el]
    elseif k == 7
        # \cite[p.~655]{kela98}.
        id = "kela98_p655"
        A = [0  3.8e3 0    0   0
             0 -3.8e3 1    0   0
             0 0     -1  5.5e6 0
             0 0      0 -5.5e6 2.7e7
             0 0      0   0   -2.7e7]
    elseif k == 8
        # \cite[Ex.~3.10]{dipa00}
        id = "dipa00_ex3.10"
        w = 1.3
        x = 1e6
        n_local = 8
        n2 = div(n_local, 2)
        A = (1 / n_local) * [w * ones(n2,n2) x * ones(n2,n2)
                       zeros(n2,n2)  -w * ones(n2,n2)]
    elseif k == 9
        id = "rosser8"
        A = Rosser(8)
        A = 2.05 * A / norm(A, 1)  # Bad case for expm re. cost.
    elseif k == 10
        id = "cos_sin_x100"
        A = [0 1e4;
             -1e4 0]  # exp = [cos(x) sin(x); - sin(x) cos(x)], x = 100;
    elseif k == 11
        id = "nilpotent"
        A = 1e2 * triu(randn(n,n), 1)  # Nilpotent.
    elseif k == 12
        # log of Cholesky factor of Pascal matrix. See \cite{edst03}.
        id = "edst03_pascal_chol"
        A = zeros(n, n)
        A[n+1:n+1:n^2] = 1:n-1
    elseif k == 13
        # \cite[p.~206]{kela89}
        id = "kela89_206"
        A = [48 -49 50 49; 0 -2 100 0; 0 -1 -2 1; -50 50 50 -52]
    elseif k == 14
        # \cite[p.~7, Ex I]{pang85}
        id = "pang85_ex1"
        A = [0    30 1   1  1  1
             -100   0 1   1  1  1
             0     0 0  -6  1  1
             0     0 500 0  1  1
             0     0 0   0  0  200
             0     0 0   0 -15 0]
    elseif k == 15
        # \cite[p.~9, Ex II]{pang85}
        # My interpretation of their matrix for arbitrary n.
        # N = 31 corresponds to the matrix in above ref.
        id = "pang85_ex2"
        A = Triw(n,n, 1)
        m = (n - 1) / 2
        A = A - diagm(diag(A)) + diagm(-m:m) * im
        for i = 1:n-1
            A[i, i+1] = -2 * (n - 1) - 2 + 4 * i
        end
    elseif k == 16
        # \cite[p.~10, Ex III]{pang85}
        id = "pang85_ex3"
        A = Triw(n,n, 1, 1)
        A = A - diagm(diag(A)) + diagm(-(n - 1) / 2:(n - 1) / 2)
    elseif k == 17
        # \cite[Ex.~5]{kela89}.
        id = "kela89_ex5"
        A = [0 1e6; 0 0]  # Same as case 6 but with ei'val 0.1 -> 0.
        Y_true = [big(1) big(1e6); 0 big(1)]
    elseif k == 18
        # \cite[(52)]{jemc05}.
        id = "jemc05_52"
        g = [0.6 0.6 4.0]
        b = [2.0 0.75]
        A = [-g[1]        0     g[1]*b[1]
                0       -g[2]   g[2]*b[2]
             -g[1]*g[3]  g[3]  -g[3]*(1-g[1]*b[1])]
    elseif k == 19
        # \cite[(55)]{jemc05}.
        id = "jemc05_55"
        g = [1.5 0.5 3.0 2.0 0.4 0.03]
        b = [0.6 7.0]
        A1 = [-g[5]     0      0
                0     -g[1]    0
              g[4]     g[4]   -g[3]]
        A2 = [-g[6]    0    g[6]*b[2]
                0    -g[2]  g[2]*b[1]
                0     g[4] -g[4]]
        A = [zeros(3, 3) I(3); A2 A1]
    elseif k == 20
        # \cite[Ex.~3]{kela98}.
        id = "kela98_ex3"
        A = [-1 1e7; 0 -1e7]
    elseif k == 21
        # \cite[(21)]{mopa03}.
        id = "mopa03_21"
        Thalf = [3.8235 * 60 * 24, 3.10, 26.8, 19.9] / 60  # Half lives in seconds/
        a = log.(2) ./ Thalf  # decay constant
        A = diagm(0=>-a, -1 => a[1:end-1])
    elseif k == 22
        # \cite[(26)]{mopa03}.
        id = "mopa03_26"
        a1 = 0.01145
        a2 = 0.2270
        A = [-a1              0  0
             0.3594 * a1    -a2  0
             0.6406 * a1     a2  0]
    elseif k == 23
        # \cite[Table 1]{kase99}.
        id = "kase99_table1"
        a = [4.916e-18
             3.329e-7
             8.983e-14
             2.852e-13
             1.373e-11
             2.098e-6
             9.850e-10
             1.601e-6
             5.796e-8
             0.000]
        A = diagm(-a) + diagm(-1 => a[1:end-1])
    elseif k == 24
        # Jitse Niesen sent me this example.
        id = "niesen_example"
        lambda = 1e6 * 1im
        mu = 1 / 2 * (-1 + sqrt(1 + 4 * lambda))
        A = [0 1; lambda -1] - mu*I(2)
    elseif k == 25
        # Awad
        id = "awad_25"
        A = [1 1e17; 0 1]
    elseif k == 26
        # Awad
        id = "awad_26"
        b = 1e3
        x = 1e10
        A = [1 - b/2   b/2; -b/2   1 + b/2]
        A = [A          x * ones(2, 2);
             zeros(2, 2)       -A]
    elseif k == 27
        # Awad
        id = "awad_27"
        b = 1e4
        A = [1 - b/2   b/2; -b/2   1 + b/2]
        # see \cite[(1.13), p. 9]{higham:matfun_book}
        Y_true = big(ℯ) * [(1-big(b)/2) b/2; -b/2 (1+b/2)]
    elseif k == 28
        # Awad
        id = "awad_28"
        b = 1e2
        A = [1 - b/2   b/2; -b^4/2   1 + b/2]
    elseif k == 29
        # \cite S. K. Godunov, "Modern Aspects of Linear Algebra",
        # \cite EigTool
        id = "godunov_eigtool"
        A = [289   2064  336   128  80   32    16
             1152  30    1312  512  288  128   32
             -29   -2000 756   384  1008 224   48
             512   128   640   0    640  512   128
             1053  2256  -504  -384 -756 800   208
             -287  -16   1712  -128 1968 -30   2032
             -2176 -287  -1565 -512 -541 -1152 -289]
        A /= 100
    elseif k == 30
        # \cite[(14.17), p. 141]{trem05}.
        id = "trem05_14.17"
        A = 10 * [0 1 2; -0.01 0 3; 0 0 0]
    elseif k == 31
        id = "invol_13_complex"
        F = schur(Matrix(Involutory(13)))
        F = Schur{Complex}(F)
        A = triu(F.T, 1)
    elseif k == 32
        # \cite{kuda10}
        id = "kuda10"
        alpha = 1
        beta = 1  # No values are given in the paper, unfortunately.
        A = -I(n) + alpha / 2 * (diagm(1 => ones(n - 1)) + diagm(-1 => ones(n - 1)))
        A[1, 2] = beta
        A[n, n - 1] = beta
    elseif k == 33
        # \cite[Benchmark #1]{lara17}
        # \cite[Problem 1]{zhao17}
        id = "lara17_benchmark1"
        A = [-3.328853448977761e-07 4.915959875924379e-18;
             0    -4.915959875924379e-18]
    elseif k == 34
        # \cite[Benchmark #2]{lara17}
        # \cite[Problem 2]{zhao17}
        id = "lara17_benchmark2"
        A = [-2.974063693062615e-07            0      1.024464026382002e-14;
             2.974063693062615e-07 -1.379680196333551e-13                 0;
             0                0     -1.024464026382002e-14]
    elseif k == 35
        # \cite[Benchmark #3]{lara17}
        # \cite[Problem 3]{zhao17}
        id = "lara17_benchmark3"
        A = [-2.421897905520424e-03            0      5.383443102348909e-03;
             0                -3.200125487349701e-04            0;
             0                 3.200125487349701e-04 -5.398342527725431e-03]
    elseif k == 36
        # \cite[Benchmark #4]{lara17}
        # \cite[Problem 4]{zhao17}
        id = "lara17_benchmark4"
        A = [-1.000000000000312e-04         0                  0      0;
             1.000000000000000e-04 -1.000000000009379e-04     0      0;
             0  1.000000000000000e-04 -1.188523972153541e-06  0;
             0     0          1.188523972153541e-06 -1.024464026382002e-14]
    elseif k == 37
        # \cite[Benchmark #5]{lara17}
        # \cite[Problem 5]{zhao17}
        id = "lara17_benchmark5"
        A = sparse(
            [1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 6,
             7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12],
            [1, 7, 1, 2, 2, 3, 10, 3, 4, 4, 5, 12, 5, 6, 6,
             7, 7, 8, 6, 9, 8, 10, 10, 11, 11, 12],
            [-1.000000000000049e-04
             5.880666420493406e-14
             1.000000000000000e-04
             -4.926419193745169e-04
             4.926419193745169e-04
             -3.405151448232769e-06
             2.980258985838552e-12
             3.405151448232769e-06
             -1.000000009110124e-04
             1.000000000000000e-04
             -1.000000033477380e-04
             1.212838692746004e-09
             1.000000000000000e-04
             -1.000015370544945e-04
             1.000000000000000e-04
             -1.000000000588073e-04
             1.000000000000000e-04
             -3.885005720114481e-05
             1.537023753355886e-09
             -5.077325179294990e-11
             3.885005720114481e-05
             -1.000000029802590e-04
             1.000000000000000e-04
             -1.906345381077957e-05
             1.906345381077957e-05
             -1.212838692746004e-09])
        A = Matrix(A)   # non ho voglia di gestire queste cose
    elseif k == 38
        # \cite[Benchmark #6]{lara17}
        # \cite[Problem 6]{zhao17}
        id = "lara17_benchmark6"
        A = sparse(
            [1,  1,  2,  2,  2,  3,  3,  3,  4,  5,  5,  6,  6,  7,  7,  8,  8],
            [1,  4,  1,  2,  4,  2,  3,  4,  4,  4,  5,  5,  6,  6,  7,  7,  8],
            [-2.930607054625170e-05
             1.292290622141271e-07
             2.446793135977101e-05
             -2.106574217602557e-05
             2.051479647948103e-08
             2.106574217602557e-05
             -9.549786402447881e-15
             1.855074206409039e-12
             -1.100000000000049e-04
             1.000000000000000e-04
             -4.926419193745169e-04
             4.926419193745169e-04
             -3.405151448232769e-06
             3.405151448232769e-06
             -1.000000091101239e-05
             1.000000000000000e-05
             -3.347737955438215e-12])
        A = Matrix(A)
    else
        return [], Y_true, "EMPTY", n_mats
    end

    return A, Y_true, id, n_mats
    end #setprecision
end

"""
    A, id, n_matrices = gallery_getall_expm(k, n=10)
"""
function gallery_getall_expm(
    k::Integer,
    n::Integer=10,
)
    #n_mats = length(vcat(101:130, 201:216, 301:328, 401:402))
    n_mats = 77

    if k < 1 
        return [], "EMPTY", n_mats
    end

    if k == 1
        id = "cauchy_$n"
        A = Cauchy(n)
    elseif k == 2
        #gallery("condex", 2,4,6) # n=2, k=4, alpha=6
        id = "condex__2_4_6"
        #\cite{high88f} (FORTRAN codes etc etc)
        inner_n = 2
        alpha   = 6
        C = create_C_condex(inner_n)    # sicuramente non sarà il modo usato 
                                        # nel MATLAB originale
        A = I(inner_n) + alpha*C
    elseif k == 3
        #gallery("condex", 3,2) # n=3, k=2, alpha=100
        id = "condex__3_2_100"
        #\cite{cline1983} (a set of counterex to 3 cond n° est)
        inner_n = 3
        alpha   = 100 # default in condex
        C = [1  1 - 2*alpha^(-1)    -2
             0      alpha^(-1)   -alpha^(-1) 
             0        0             1]
        A = [C                  zeros(3,inner_n-3)
            zeros(inner_n-3,3)  I(inner_n-3)]
    elseif k == 4
        #gallery("condex", n,3) # k=3, alpha=100
        id = "condex__$(n)_3_100"
        A = UnitLowerTriangular(-ones(n,n))
    elseif k == 5
        # condex (symmetric real)
        #gallery("condex", n,4,100)
        id = "condex__$(n)_4_100"
        #\cite{high88f} (FORTRAN codes etc etc)
        alpha = 100
        C = create_C_condex(n)  # sicuramente non sarà il modo usato
                                # nel MATLAB originale
        A = I(n) + alpha*C
    elseif k == 6
        id = "dorr_$(n)_100.0"
        A = Dorr(n, 100.0)
    elseif k == 7
        id = "dramadah_$(n)_2"
        A = Dramadah(n, 2)
    elseif k == 8
        id = "frank_$n"
        A = Frank(n)
    elseif k == 9
        id = "gcdmat_$n"
        A = GCDMat(n)   # (symmetric real)
    elseif k == 10
        id = "grcar_$n"
        A = Grcar(n)
    elseif k == 11
        id = "hanowa_$n"
        A = Hanowa(n)
    elseif k == 12
        id = "hilbert_$n"
        A = Hilbert(n)  # (symmetric real)
    elseif k == 13
        id = "invhess_$n"
        A = Invhess(n)
    elseif k == 14
        id = "jordbloc_$(n)_1"
        A = JordBloc(n, 1) # (symmetric real)
    elseif k == 15
        id = "kahan_$n"
        A = Kahan(n)
    elseif k == 16
        id = "lehmer_$n"
        A = Lehmer(n)   # (symmetric real)
    elseif k == 17
        id = "minij_$n"
        A = Minij(n)    # (symmetric real)
    elseif k == 18
        id = "moler_$n"
        A = Moler(n)    # (symmetric real)
    elseif k == 19
        id = "parter_$n"
        A = Parter(n)
    elseif k == 20
        id = "pei_$n"
        A = Pei(n)
    elseif k == 21
        id = "poisson_$(ceil(Int, sqrt(n)))"
        A = Poisson(ceil(Int, sqrt(n)))# (symmetric real, n^2)
    elseif k == 22
        id = "prolate_$(n)_1.0"
        A = Prolate(n, 1.)  # (symmetric real Toeplitz)
    elseif k == 23
        id = "randcorr_$n"
        A = Randcorr(n) # (symmetric real)
    elseif k == 24
        id = "sampling_$n"
        A = Sampling(n)
    elseif k == 25
        id = "toeppd_$n"
        A = Toeppd(n)   # (symmetric real)
    elseif k == 26
        id = "symtridiagonal_$(n)_2_-1"
        A = SymTridiagonal(2*ones(n), -ones(n-1))
        #A = Matrix(A)
    elseif k == 27
        # symmetric real
        id = "wathen_$(ceil(Int, n^(1/4)))_$(ceil(Int, n^(1/4)))"
        A = Wathen(ceil(Int, n^(1/4)), ceil(Int, n^(1/4)))
    elseif k == 28
        id = "wilkinson_3"
        A = Wilkinson(3)
    elseif k == 29
        id = "wilkinson_4"
        A = Wilkinson(4)
    elseif k == 30
        id = "wilkinson_5"
        A = Wilkinson(5)
    elseif k == 31
        id = "binomial_$n"
        A = Binomial(n)
    elseif k == 32
        id = "fiedler_$n"
        A = Fiedler(n)
    elseif k == 33
        id = "householder_$n"
        v, beta = create_H_house(n) # v is 1 dimensional!
        A = I(n) .- beta * (v * v')
    elseif k == 34
        id = "jordbloc_$(n)_2"
        A = JordBloc(n, 2)
    elseif k == 35
        id = "kms_$n"
        A = KMS(n)
    elseif k == 36
        id = "lesp_$n"
        A = Lesp(n)
    elseif k == 37
        id = "lotkin_$n"
        A = Lotkin(n)
    elseif k == 38
        id = "orthog_$(n)_1"
        A = Orthog(n, 1)
    elseif k == 39
        id = "orthog_$(n)_2"
        A = Orthog(n, 2)
    elseif k == 40
        id = "orthog_$(n)_5"
        A = Orthog(n, 5)
    elseif k == 41
        id = "orthog_$(n)_6"
        A = Orthog(n, 6)
    elseif k == 42
        id = "orthog_$(n)_-1"
        A = Orthog(n, -1)
    elseif k == 43
        id = "redheff_$n"
        A = Redheff(n)
    elseif k == 44
        id = "riemann_$n"
        A = Riemann(n)
    elseif k == 45
        id = "ris_$(n)"
        A = RIS(n)  # nel MATLAB è gallery("ris",n,1e1) ma 1e1 non fa nulla
    elseif k == 46
        id = "wilkinson_21"
        A = Wilkinson(21)
    elseif k == 47
        id = "clement_$(n)_1"
        A = Clement(n, 1)
    elseif k == 48
        id = "chebspec_$n"
        A = ChebSpec(n)
    elseif k == 49
        id = "chebvand_$n"
        x = collect(range(0,1,length=n))
        A = create_V_ChebVand(x)
    elseif k == 50
        id = "chow_$n"
        A = Chow(n)
    elseif k == 51
        id = "circulant_$n"
        A = Circulant(n)
    elseif k == 52
        id = "cycol_$n"
        A = Cycol(n)
    elseif k == 53
        id = "dramadah_$(n)_1"
        A = Dramadah(n, 1)
    elseif k == 54
        id = "dramadah_$(n)_3"
        A = Dramadah(n, 3)
    elseif k == 55
        id = "forsythe_$n"
        A = Forsythe(n)
    elseif k == 56
        id = "leslie_$(n)_1"
        A = Leslie(n)
    elseif k == 57
        id = "leslie_$(n)_2"
        A = Leslie(n)
    elseif k == 58
        # uso Xoshiro(10) solo perché il codice originale
        # usa 10 come seed. 
        id = "randn_xoshiro10_$(n)x$(n)"
        A = randn(Xoshiro(10), (n,n))
    elseif k == 59
        id = "orthog_complex_$(n)_3"
        A = Orthog{ComplexF64}(n, 3)
    elseif k == 60
        id = "orthog_$(n)_4"
        A = Orthog(n, 4)
    elseif k == 61
        id = "orthog_$(n)_-2"
        A = Orthog(n, -2)
    elseif k == 62
        id = "randcolu_$n"
        A = Randcolu(n)
    elseif k == 63
        id = "randhess_$n"
        # LO SO, sicuramente quella che sto per restituire NON è
        # veramente una matrice di Hessenberg random (diciamo rispetto)
        # a una misura che abbia senso sulle matrici (Hessenberg)
        A = hessenberg(rand(n,n)).H
    elseif k == 64
        id = "rando_$(n)_1"
        A = Rando(n, 1)
    elseif k == 65
        id = "rando_$(n)_2"
        A = Rando(n, 2)
    elseif k == 66
        id = "rando_$(n)_3"
        A = Rando(n, 3)
    elseif k == 67
        id = "randsvd_$(n)_1.0"
        A = RandSVD(n, 1.)
    elseif k == 68
        id = "randsvd_$(n)_2.0"
        A = RandSVD(n, 2.)
    elseif k == 69
        id = "randsvd_$(n)_3.0"
        A = RandSVD(n, 3.)
    elseif k == 70
        id = "randsvd_$(n)_4.0"
        A = RandSVD(n, 4.)
    elseif k == 71
        id = "randsvd_$(n)_5.0"
        A = RandSVD(n, 5.)
    elseif k == 72
        id = "smoke_$n"
        A = Smoke(n)
    elseif k == 73
        id = "smoke_$(n)_1"
        A = Smoke(n, 1)
    elseif k == 74
        id = "toeppen_$n"
        A = Toeppen(n)
    elseif k == 75
        id = "uniformdata_$(n)_1000"
        # uso Xoshiro(1000) solo perché il codice originale
        # usa 1000 come seed.
        A = rand(Xoshiro(1000), n,n)
    elseif k == 76
        id = "gearmat_$n"
        A = GearMat(n)
    elseif k == 77
        id = "neumann_$(ceil(Int, sqrt(n))^2)"
        A = Neumann(ceil(Int, sqrt(n))^2)
    else
        #error("k can be at most $(n_mats)")
        return [], "EMPTY", n_mats
    end

    return A, id, n_mats
end


export FasiMatrices, expm_testmats, gallery_getall_expm






end #module