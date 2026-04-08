using LinearAlgebra, Random, Printf
using Revise

Revise.includet(joinpath(@__DIR__,"..","..","src","modules","MyMpExponential.jl"))
Revise.includet(joinpath(@__DIR__,"..","..","src","modules","MyHelper.jl"))
Revise.includet(joinpath(@__DIR__,"..","..","src","modules","MyMatMul.jl"))
using .MyMpExponential, .MyHelper, .MyMatMul

Random.seed!(42)

## Helper functions
function verbose_mult_pade(d, Apows)
    length(Apows) > 1 || throw(ArgumentError(lazy"Supply at least the 0th and 1st power of the matrix"))

    print("Task: multiply by A^$d, using the powers I, A²")
    if length(Apows) > 2
        print(",…, A²^($(length(Apows)-1))")
    end
    print(" present in `Apows`\n")

    l = length(Apows)
    mults = zeros(1, l)
    while d > 1 && l > 1
        for _ = 1:div(d,2*(l-1))
            mults[l] += 1
        end
        d = mod(d, 2*(l-1))
        l = min(l-1, div(d,2)+1)
    end
    if d == 1
        mults[1] += 1    
    end

    print("mults = $mults\n")
    for i = 2:length(mults)
        print("Multiply $(mults[i]) times by Apows[$i] = (A²)^($(i-1))\n")
    end
    if mults[1] > 0 
        print("… and a multiplication by A\n")
    end
    return
end


function verbose_mult_tayl(d, Apows)
    length(Apows) > 1 || throw(ArgumentError(lazy"Supply at least the 0th and 1st power of the matrix"))

    print("Task: multiply by A^$d, using the powers I, A")
    if length(Apows) > 2
        print(",…, A^($(length(Apows)-1))")
    end
    print(" present in `Apows`\n")

    l = min(length(Apows), d+1)
    mults = zeros(1,l)
    while d > 0 
        for _ = 1:div(d, l-1)
            mults[l] += 1
        end
        d = mod(d, l-1)
        l = min(l-1, d+1)
    end

    print("mults = $mults\n")
    for i = 2:length(mults)
        print("Multiply $(mults[i]) times by Apows[$i] = A^$(i-1)\n")
    end
end


## First test: Padé version, explicit prints
n = 5;
A = rand(n,n);

Apows_pade = [I(n), A^2]

for d in [1, 2, 3, 4, 8]
    verbose_mult_pade(d, Apows_pade)
    print("\n")
end

print("Adding some elements to `Apows`…\n")

push!(Apows_pade, A^4);
push!(Apows_pade, A^6);
push!(Apows_pade, A^8);

for d in [1, 2, 10, 11, 13, 17]
    verbose_mult_pade(d, Apows_pade)
    print("\n")
end


## Second test: Taylor version, explicit prints
Apows_tayl = [I(n), A];

for d in [1,2,4]
    verbose_mult_tayl(d, Apows_tayl)
    print("\n")
end

print("Adding more items to Apows_tayl…\n")

push!(Apows_tayl, A^2);
push!(Apows_tayl, A^3);

for d in [1,2,4,7,8,9, 17]
    verbose_mult_tayl(d, Apows_tayl)
    print("\n")
end


## Third test: testing the mk1 function

d = 12;         # Even power of A

function run_mk1_test(d, t=1)
    print("size(A) = $(size(A))\teltype(A) = $(eltype(A))\n")
    X = t == 1 ? rand(n) : rand(n,t)

    Y_true = A^d * X;
    Y_pade = evalPowVecDiag_mk1(d, A, Apows_pade, X);

    print("Testing the two ways to apply A^$d on a random X (Padé version)\n")
    @printf("|| Y_pade - Y_true || / || Y_true || = %.4g\n", rel_err(Y_pade, Y_true))
    print("Y_pade ≈ Y_true is $(Y_pade ≈ Y_true)\n")

    Y_tayl = evalPowVecDiag_mk1(d, A, Apows_tayl, X, use_taylor=true);

    print("Testing the two ways to apply A^$d on a random X (Taylor version)\n")
    @printf("|| Y_tayl - Y_true || / || Y_true || = %.4g\n", rel_err(Y_tayl, Y_true))
    print("Y_tayl ≈ Y_true is $(Y_tayl ≈ Y_true)\n\n")
end

run_mk1_test(13)    # Odd power of A


## Fourth test: the mk1 function, acting on a skinny matrix
print("Same test as before, but A^d is applied to a skinny matrix, not a vector.\n")
t = 3;
run_mk1_test(12, t)
