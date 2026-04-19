using LinearAlgebra

struct Factorials
    f_vec::Vector{BigInt}
    Factorials( ) = new([big(1)])
    Factorials(n) = 
        begin
            v = [factorial(big(k)) for k=0:n]
            new(v)
        end
end


function (f::Factorials)(
    n::Real; 
    return_type::DataType=BigInt
)
    l = length(f.f_vec)
    if n+1 > l
        nuovi_f = l:n .|> big .|> factorial
        append!(f.f_vec, nuovi_f)
    end
    return return_type(f.f_vec[end])
end


function (f::Factorials)(
    r::AbstractRange; 
    return_type::DataType=BigInt
)
    fst, _..., lst = r
    l = length(f.f_vec)
    if lst+1 > l
        nuovi_f = l:lst .|> big .|> factorial
        append!(f.f_vec, nuovi_f)
    end
    return return_type.(f.f_vec[fst+1:lst+1])
end




### WIP: forse non serve a niente questo struct. Pensaci su.

function __diag_pade_coeffs__(
    from, to;
    precision_factor=2
    )
    bfrom, bto = big(from), big(to)
    ranges = bfrom:bto

    old_prec = precision(BigFloat)
    setprecision(BigFloat, floor(Int64, precision_factor*precision(BigFloat)))
    
    numerators = (factorial(bto)/factorial(2bto)) ./
                 (factorial.(bto .- ranges) .* factorial.(ranges)) .*
                 factorial.(2bto .- ranges)
    denominators = ((-1).^ranges) .* numerators

    setprecision(BigFloat, old_prec)

    return numerators, denominators
end


struct DiagPadeCoefficients
    c_num::Vector{BigFloat}
    c_den::Vector{BigFloat}
    DiagPadeCoefficients( ) = new([big(1.)],[big(1.)])
    DiagPadeCoefficients(m) = 
        begin
            nums, dens = __diag_pade_coeffs__(0, m)
            new(nums, dens)
        end
end


function (c::DiagPadeCoefficients)(
    n::Real;
    return_type::DataType=BigFloat
)
    l = length(c.c_num)
    if n+1 > l 
        new_nums, new_dens = __diag_pade_coeffs__(l, n)
        append!(c.c_num, new_nums)
        append!(c.c_den, new_dens)
    end
    return return_type(c.c_num[end]), return_type(c.c_den[end])
end


function (c::DiagPadeCoefficients)(
    r::AbstractRange;
    return_type::DataType=BigFloat
)
    fst, _..., lst = r
    l = length(c.c_num)
    if lst+1 > l
        new_nums, new_dens = __diag_pade_coeffs__(l, n)
        append!(c.c_num, new_nums)
        append!(c.c_den, new_dens)
    end
    nums_ret = c.c_num[fst+1:lst+1]
    dens_ret = c.c_den[fst+1:lst+1]
    return return_type.(nums_ret), return_type.(dens_ret)
end


