using LinearAlgebra

function compute_c_k(k, M = 10000; precision=2048)
    #c_k = setprecision(precision) do 
        #thetas = ((2 .* (big(1):M) .- 1) .* (big(π)/(2M)))
        thetas = ((2 .* (1:M) .- 1) .* (π/(2M)))
        c_k = sum(exp.(cos.(thetas)) .* cos.(k .* thetas))
        2c_k/M
    #end
    #Float64(c_k)
end

function compute_c_k_legacy(k, M = 10000, precision = 2048)
    #c_k = setprecision(precision) do
        c_k = 0
        for j=1:M
            #theta = (2big(j) - 1) * (big(π)/(2M))
            theta = (2j - 1) * (π/(2M))
            c_k += exp(cos(theta)) * cos(k * theta)
        end 
        (2*c_k) / M
    #end 
    #Float64(c_k)
end


function cheby(A; v=nothing, tol=eps(), maxit=1000)
    n = LinearAlgebra.checksquare(A)
    if isnothing(v)
        v = I(n)
    end

    # Um2 = Uₖ₋₃, Um1 = Uₖ₋₂, U0 = Uₖ₋₁
    Um2, Um1, U0 = -v, zero(v), v
    U1 = 2A * v         # U1 = Uₖ

    c_k = compute_c_k_legacy(0)

    Y  = (c_k / 2) * U0
    Yprime = zero(v)
    Ay = (c_k / 4) * U1

    resnorm = norm(Ay - Yprime)

    for k=1:maxit
        U2 = 2A * U1 - U0   # U2 = Uₖ₊₁
        c_k = compute_c_k_legacy(k)

        Y  += (c_k / 2) * (U1 - Um1)
        Yprime += (c_k * (k/2)) * (U1 + Um1)
        Ay += (c_k / 2) * (U2 - Um2)

        Um2 = Um1 
        U1  = U0
        U0  = U1
        U1  = U2

        resnorm = norm(Ay - Yprime)
        print("k = $k\t resnorm = $resnorm\n")

        if resnorm < tol
            print("Iteration $k:\n")
            print("\tresnorm = $(resnorm)\n")
            break 
        end
    end
    print("Finishing with resnorm = $resnorm\n")

    return Y
end