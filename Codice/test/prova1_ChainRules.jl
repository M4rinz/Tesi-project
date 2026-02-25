"""
Intended to be used in a VS Code interactive session
"""

using LinearAlgebra, FiniteDifferences
using ChainRules, ChainRulesCore

# helper functions

function rel_err(approx, exact)
    norm(approx - exact) / norm(exact)
end

function construct_full_jacobian(linear_operator)
    """ 
    Given a linear operator ``\\mathbb{C}^{n\\times n}\\to\\mathbb{C}^{n\\times n}`` 
    (typically, a fréchet differential ``L_f(A)``), constructs the matrix 'K'
    such that 'linear_operator[E] = K * y' where ``y=\\operatorname{Vec}(E)``
    """
    K = Matrix{Float64}(undef, n^2, n^2)    # il tipo lo metto perché lo so e non ho voglia di impazzire
    e_i = zeros(n); e_i[1] = 1
    e_j = zeros(n); e_j[1] = 1
    for j = 1:n#, i = 1:n
        for i = 1:n
            column = linear_operator(e_i * e_j')
            K[:, (j-1)*n + i] = vec(column)
            
            circshift!(e_i, 1)
        end
        circshift!(e_j, 1)
    end
    return K
end


n = 3
A = rand(n,n)

@edit exp(A)

@edit ChainRules.rrule(exp, A)

# compute exp(A) and the pullback via rrule
X, exp_pullback = ChainRules.rrule(exp, A)

print("|| X - exp(A) || / || exp(A) || = $(rel_err(X, exp(A)))\n")
print("The 1st argument is the matrix exponential.\n")

K_back = construct_full_jacobian(x -> exp_pullback(x)[2])

# My finite difference approx
my_fd_function = X -> (exp(A + √eps() * X) - exp(A)) / √eps()
K_fd = construct_full_jacobian(my_fd_function)

print("|| K_back' - K_fd || / || K_fd || = $(rel_err(K_fd, K_back'))\n")
print("Being given by a reverse (adjoint) rule, K_back computes the adjoint of the Fréchet derivative\n")
print("When vectorized, this becomes the K' (I need to check this formally but I'm quite sure)\n")

print("L(A)[E]^{\\top} = L(A')[E']")


# compute the pushforward (i.e. the Fréchet derivative at A in a given direction) via frule
# unlike previous case, an action is not returned, thus we construct the full jacobian
# by passing the frule to `construct_full_jacobian`
dself = ChainRulesCore.NoTangent();
exp_pushforward = E -> ChainRules.frule((dself, E), LinearAlgebra.exp!, A)[2]
K_forward = construct_full_jacobian(exp_pushforward)

K_fd_good = FiniteDifferences.jacobian(central_fdm(5,1), exp, A)[1]

print("||K forward - K_fd_good || / || K_fd_good || = $(rel_err(K_fd_good, K_forward))\n")
print("Also K_forward is a good approximation\n")

