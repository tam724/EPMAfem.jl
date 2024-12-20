const EEEO = (
    eee = (true, true, true),
    eoo = (true, false, false),
    oeo = (false, true, false),
    ooe = (false, false, true),
    
    oee = (false, true, true),
    eoe = (true, false, true),
    eeo = (true, true, false),
    ooo = (false, false, false),
)

struct SphericalHarmonic{T<:Integer}
    degree::T # l
    order::T # k (jonas thesis) or m (wikipedia)

    function SphericalHarmonic(degree, order)
        deg = (degree >= 0) ? degree : error("degree = $(degree) must be >= 0")
        ord = (abs(order) <= deg) ? order : error("abs(order = $(order)) must be <= degree")
        return new{typeof(deg)}(deg, ord)
    end
end

degree(sh::SphericalHarmonic) = sh.degree
order(sh::SphericalHarmonic) = sh.order

degreeorder(sh::SphericalHarmonic) = sh.degree, sh.order

"""
    classification of spherical harmonics in even f(Ω) = f(-Ω) and odd f(Ω) = -f(-Ω) functions
"""
is_even(m::SphericalHarmonic) = mod(degree(m), 2) == 0
is_odd(m::SphericalHarmonic) = !is_even(m)

"""
    classification of spherical harmonics with respect to the cartesian unit vectors
    see https://publications.rwth-aachen.de/record/819622/files/819622.pdf (page 72)
"""
is_even_in(m::SphericalHarmonic, D::Int) = (D in (1, 2, 3)) ? is_even_in(m, Val(D)) : error("D must be 1, 2 or 3 (x, y or z)")

function is_even_in(m::SphericalHarmonic, ::Z) # z basis vector
    return iseven(degree(m) + order(m))
end

function is_even_in(m::SphericalHarmonic, ::X) # x basis vector
    k = order(m)
    return (k < 0 && isodd(k)) || (k >= 0 && iseven(k))
end

function is_even_in(m::SphericalHarmonic, ::Y) # y basis vector
    return order(m) >= 0
end

is_odd_in(m::SphericalHarmonic, d::Val) = !is_even_in(m, d)

function get_eee(m::SphericalHarmonic)
    return map(d -> is_even_in(m, d), dimensions())
end

function has_same_eee(m1, m2)
    return all(get_eee(m1) == get_eee(m2))
end

import Base: isless
function isless(m1::SphericalHarmonic, m2::SphericalHarmonic)
    if degree(m1) < degree(m2) return true
    elseif degree(m1) == degree(m2)
        if order(m1) < order(m2) return true end
        return false
    end
    return false
end

function isless_evenodd(m1::SphericalHarmonic, m2::SphericalHarmonic)
    if (is_even(m1) == is_even(m2))
        return isless(m1, m2)
    else
        if is_even(m1)
            return true
        else
            return false
        end
    end
end

function isless_eee(eee1::NTuple{3, Bool}, eee2::NTuple{3, Bool})
    sort_list = EEEO
    i1 = findall(x -> x == eee1, sort_list)
    i2 = findall(x -> x == eee2, sort_list)
    return i1 < i2
end

function isless_eeevenodd(m1::SphericalHarmonic, m2::SphericalHarmonic)
    if (is_even(m1) == is_even(m2))
        if has_same_eee(m1, m2)
            return isless(m1, m2)
        else
            return isless_eee(get_eee(m1), get_eee(m2))
        end
    else
        if is_even(m1)
            return true
        else
            return false
        end
    end
end