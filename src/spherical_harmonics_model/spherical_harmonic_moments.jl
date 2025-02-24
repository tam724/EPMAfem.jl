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

function (sh::SphericalHarmonic)(Ω::VectorValue)
    θ, ϕ = unitsphere_cartesian_to_spherical(VectorValue(Ωz(Ω), -Ωx(Ω), -Ωy(Ω)))
    return SphericalHarmonics.computeYlm(θ, ϕ, lmax=degree(sh), SHType=SphericalHarmonics.RealHarmonics())[(degree(sh), order(sh))]
end

degree(sh::SphericalHarmonic) = sh.degree
order(sh::SphericalHarmonic) = sh.order

degreeorder(sh::SphericalHarmonic) = sh.degree, sh.order

"""
    classification of spherical harmonics in even f(Ω) = f(-Ω) and odd f(Ω) = -f(-Ω) functions
"""
is_even(sh::SphericalHarmonic) = mod(degree(sh), 2) == 0
is_odd(sh::SphericalHarmonic) = !is_even(sh)

"""
    classification of spherical harmonics with respect to the cartesian unit vectors
    see https://publications.rwth-aachen.de/record/819622/files/819622.pdf (page 72)
"""
function is_even_in(sh::SphericalHarmonic, ::Z) # z basis vector
    return iseven(degree(sh) + order(sh))
end

function is_even_in(sh::SphericalHarmonic, ::X) # x basis vector
    k = order(sh)
    return (k < 0 && isodd(k)) || (k >= 0 && iseven(k))
end

function is_even_in(sh::SphericalHarmonic, ::Y) # y basis vector
    return order(sh) >= 0
end

function is_even_in(sh, n::VectorValue)
    # only test a few random directions
    n_rand = 10
    is_even = zeros(Bool, 10)
    is_odd = zeros(Bool, 10)
    for i in 1:n_rand
        Ω = VectorValue(randn(), randn(), randn()) |> normalize
        a = sh(Ω)
        b = sh(Ω - 2.0*dot(n, Ω)*n)
        is_even[i] = a ≈ b
        is_odd[i] = a ≈ -b
    end
    if all(is_even)
        return true
    elseif all(is_odd)
        return false
    else
        @show is_even
        @show is_odd
        error("Spherical harmonic $sh is neither even nor odd in the given direction")
    end
end

is_odd_in(sh::SphericalHarmonic, d::SpaceDimension) = !is_even_in(sh, d)
is_odd_in(sh, n::VectorValue) = !is_even_in(sh, n)

function get_eee(sh::SphericalHarmonic)
    return map(d -> is_even_in(sh, d), dimensions())
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