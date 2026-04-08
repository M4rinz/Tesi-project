using LinearAlgebra

default_prec = precision(BigFloat);
print("The original precision of BigFloat is $default_prec\n")

print("Let me introduce... Pi. As we know,")
display(π)
print("In Julia, π is a constant of type $(typeof(π))\n")

msg = """In the following, roughly (lo so che non è proprio corretto, ma è per avere un'idea):
    - precision: refers to the precision of the floating point numbers (i.e. n° of bits in the mantissa). 
                 It relates to the `precision` and `setprecision` commands
    - accuracy: refers to the number of correct significant digits in an expression.
                It relates to how operations are performed at a given precision
""";
print(msg * "\n")
## 0th numerical check
let 
    setprecision(BigFloat, 512)
    print("Inside this `let` block, the precision is $(precision(BigFloat))\n")
end
print("Now outside the `let` block, the precision is $(precision(BigFloat))\n")
setprecision(BigFloat, default_prec);
begin
    setprecision(BigFloat, 512)
    print("Inside this `begin` block, the precision is $(precision(BigFloat))\n")
end
print("Now outside the `begin` block, the precision is $(precision(BigFloat))\n")
print("Let's put things back to normal...\n")
setprecision(BigFloat, default_prec);

print("Moral of the story: the `setprecision(BigFloat, n)` command changes the precision in the global scope."
* " I.e., the effects of the command are visible globally!\n")

## First numerical check
expr1 = π * (1/π);
print("π * π⁻¹ = $(expr1).\t Expression is of type $(typeof(expr1))\n\n")

expr2 = BigFloat(π) * BigFloat(π)^-1;
print("Converting all to BigFloat, π * π⁻¹ = $expr2.\t Expression is of type $(typeof(expr2))\n\n")


expr3 = BigFloat(π) * inv(π);
print("However, if we only convert one of the two factors, we get π * π⁻¹ = $expr3, of type $(typeof(expr3))\n")
print("| 1 - (π * π⁻¹) | = $(abs(1-expr3))\n\n")

expr3_msg = """Here's what I think that happened:
I believe that π was first evaluated (in double precision) and then promoted to `BigFloat`, to match the type of the first factor.
However, when promoting a Float64 to BigFloat, a lot of "junk" remains (since the computation is not accurate anymore after the 16th decimal digit,
promoting to BigFloat exposes the digits that are inaccurately computed).
""";
print(expr3_msg)
print("\n")

expr4 = big(π) - π;
print("big(π) - π = $expr4.\t Expression is of type $(typeof(expr4))\n\n")

expr5 = big(π) - 1.0*π;
print("big(π) - 1.0 * π = $expr5. Expression is of type $(typeof(expr5))\n")
print("Same phenomenon of third example: 1.0 * π is first evaluated in Float64 and then promoted. "
* "Instead, in the previous example the constant π was directly evaluated in BigFloat.\n\n")


## Second numerical check
A = rand(3,3);
B = A;
print("Assignment:\t\tB == A is $(B == A).\t B === A is $(B === A).\t pointer(B) == pointer(A) is $(pointer(B) == pointer(A))\n")

B = convert(Matrix{big(eltype(A))}, A);
print("Conversion:\t\tB == A is $(B == A).\t B === A is $(B === A).\t pointer(B) == pointer(A) is $(pointer(B) == pointer(A))\n")
print("Moral of the story: `convert` creates a copy.\n\n")


## Third numerical experiment
A = π * ones(Float64, 3,3);         # 16 digits accuracy
A = convert(Matrix{BigFloat}, A);   # convert to BigFloat
A /= 4.0;

B = π * ones(BigFloat, 3,3);        # 77 ≈ log₁₀(2⁻²⁵⁶) digits accuracy
B /= 4.0;       

print("A == float(B) is $(A == Float64.(B))\n")

## Fourth numerical experiment
pigreco = big(π);   # 77 ≈ log₁₀(2⁻²⁵⁶) digits accuracy
P_512 = setprecision(512) do 
    print("Inside this block, precision is $(precision(BigFloat)), ")
    print("guaranteeing $(precision(BigFloat, base=10)) correct decimal digits\n\n")
    P_512 = π * ones(BigFloat, size(A)...);
    print("(all in precision 512)\tP_512 - big(π) =\n")
    display(P_512 - big(π)*ones(size(P_512)...))
    print("\n")
    
    print("(π in double precision)\tP_512 - π =\n")
    display(P_512 - π*ones(size(P_512)...))
    print("\n")
    
    print("(π in precision 256)\tP_512 - big(π) =\n")
    display(P_512 - pigreco*ones(size(P_512)...))
    print("\n")
    P_512;
end

π_512 = P_512[1,1];     # correct up to the 154 decimal place (manually checked)

print("π_512 is correct up to the 154th decimal digit. I checked manually. However...\n")
print("π_512 - big(π) = $(π_512 - big(π))\n")
print("π_512 ≈ big(π) is $(π_512 ≈ big(π))\n")
print("Nothing to worry about. Is the same up to the machine precision ($(eps(BigFloat)))\n")
print("π_512 == big(π_512) is $(π_512 == big(π_512)), but π_512 == BigFloat(π_512) is $(π_512 == BigFloat(π_512))\n")

setprecision(92, base=10) do 
    print("Inside block, precision is $(precision(BigFloat)), ")
    print("guaranteeing $(precision(BigFloat, base=10)) correct decimal digits\n")
    print("BigFloat(π_512) = $(BigFloat(π_512))\n")
    print("big(π_512) = $(big(π_512))\n")
end

print("""Moral of the story: there's a subtle difference in how `big` and `BigFloat` behave. """
* """Outside a `do` block, quantities retain the precision at which they were computed, """
* """but this is accessible only if we use `big`, as `BigFloat` uses the precision of the current scope.\n""")


## Fifth numerical experiment
A = ones(BigFloat, 2,2) / big(π);    # 77 correct decimal digits

expr = setprecision(92, base=10) do 
    print("Inside block, precision is $(precision(BigFloat)), ")
    print("guaranteeing $(precision(BigFloat, base=10)) correct decimal digits\n")
    expr = π_512 * A .- big(1.0)
    display(expr)
    print("|| π_512 * A .- 1 || = $(norm(expr))\n")
    expr2 = big(π) * A .- big(1.0)
    display(expr2)
    print("using π with $(precision(BigFloat, base=10)) significant digits, we get "
            * "||π * A .- 1|| = $(norm(expr2))\n")
    expr;
end
print("""Moral of the story: the computation is only as accurate as the least accurate of the operands. """
* """Using a higher-precision constant doesn't seem to reduce the final error.\n""")


## Sixth numerical experiment
# If both operands are BigFloats computed at a higher precision
# than that in the `do` block, what is the accuracy of the computation?
# Is there a difference in behaviour, in using `BigFloat`, or `big`?

π_306 = setprecision(306) do 
    BigFloat(π) 
end

print("π_512 - π_306 = $(π_512 - π_306)\n")
print("whereas precision is = $(precision(BigFloat)), "
* "yielding $(precision(BigFloat, base=10)) significant digits.\n")
print("BigFloat(π_512) - BigFloat(π_306) = $(BigFloat(π_512) - BigFloat(π_306)).\n")
print("BigFloat(π_512) - π_306 = $(BigFloat(π_512) - π_306)\n")
print("\n")

msg = """Moral of the story: without specifying anything, arithmetic operations (results)
between BigFloats maintain an accuracy that depends on the original precision of the operands.
By using `big`, this doesn't change, as discussed in earlier examples. By using 
`BigFloat` to an operand, then the precision of that operand matches the precision of the current
scope, and the accuracy of the computation is impacted by this (it matches the machine epsilon 
of the current precision, if the conditioning doesn't screw us up).
"""
print(msg * "\n")