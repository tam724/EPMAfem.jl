struct RealHarmonic{D, T<:Integer, C}
    degree::T # l
    order::T # k (jonas thesis) or m (wikipedia)
    cache::C # for fast evaluations

    function RealHarmonic{3}(degree, order, cache=nothing)
        deg = (degree >= 0) ? degree : error("degree = $(degree) must be >= 0")
        ord = (abs(order) <= deg) ? order : error("abs(order = $(order)) must be <= degree")
        T = promote_type(typeof(deg), typeof(ord))
        if isnothing(cache)
            @warn "For fast evaluations of spherical harmonics, also pass a cache to the constructor"
        else
            #check cache validity
            cache.P.lmax >= deg || error("cache lmax $(cache.P.lmax) < degree $deg")
        end
        return new{3, T, typeof(cache)}(deg, ord, cache)
    end

    function RealHarmonic{2}(degree, order, cache=nothing)
        deg = (degree >= 0) ? degree : error("degree = $(degree) must be >= 0")
        ord = (order >= 0 && order <= degree && order <=1) ? order : error("order must be ∈{0, 1} (if degree == 0, order = 0)")
        T = promote_type(typeof(deg), typeof(ord))
        return new{2, T, Nothing}(deg, ord, cache)
    end
end

const SphericalHarmonic{T, C} = RealHarmonic{3, T, C}
const CircularHarmonic{T, C} = RealHarmonic{2, T, C}

SphericalHarmonic(degree, order, cache=nothing) = RealHarmonic{3}(degree, order, cache)
CircularHarmonic(degree, order, cache=nothing) = RealHarmonic{2}(degree, order, cache)

is_viable(::SphericalHarmonic, ::Dimensions._3D) = true
is_viable(m::SphericalHarmonic, ::Dimensions._2D) = order(m) >= 0
is_viable(m::SphericalHarmonic, ::Dimensions._1D) = order(m) == 0
is_viable(::CircularHarmonic, ::Dimensions._3D) = error("CircularHarmonic only live in 2D")
is_viable(::CircularHarmonic, ::Dimensions._2D) = true
is_viable(m::CircularHarmonic, ::Dimensions._1D) = order(m) == 0

spherical_harmonics(N, ND=Dimensions._3D(), cache=spherical_harmonics_cache(N)) = [SphericalHarmonic(l, k, cache) for l in 0:N for k in -l:l if is_viable(SphericalHarmonic(l, k, cache), ND)]
circular_harmonics(N, ND=Dimensions._2D()) = [CircularHarmonic(l, k) for l in 0:N for k in (l == 0 ? (0:0) : (0:1)) if is_viable(CircularHarmonic(l, k), ND)]

degree(h::RealHarmonic) = h.degree
order(h::RealHarmonic) = h.order

degreeorder(h::RealHarmonic) = h.degree, h.order

Base.show(io::IO, sh::SphericalHarmonic) = print(io, "SH(deg=$(degree(sh)),ord=$(order(sh)))")
Base.show(io::IO, ch::CircularHarmonic) = print(io, "CH(deg=$(degree(ch)),ord=$(order(ch)))")
Base.show(io::IO, ::MIME"text/plain", h::RealHarmonic) = show(io, h)

ComponentArrays.recursive_length(sh::RealHarmonic) = 1

function spherical_harmonics_cache(N)
    return SphericalHarmonics.cache(Float64, N; SHType=SphericalHarmonics.RealHarmonics())
end

function eval_cache!(cache, θ, ϕ)
    SphericalHarmonics.computePlmcostheta!(cache, θ)
    SphericalHarmonics.computeYlm!(cache, θ, ϕ)
end

function eval_naive(sh::SphericalHarmonic, Ω::VectorValue)
    # see diss (for testing the function definitions with the library implementation)
    l, m = degree(sh), order(sh)
    θ, ϕ = unitsphere_cartesian_to_spherical(VectorValue(Ωz(Ω), -Ωx(Ω), -Ωy(Ω)))
    if m < 0
        return sqrt((2*l+1)/(2π) * factorial(l-abs(m))/factorial(l+abs(m)))*LegendrePolynomials.Plm(cos(θ), l, abs(m))*sin(abs(m)*ϕ)
    elseif m == 0
        return sqrt((2*l+1)/(4π)) * LegendrePolynomials.Plm(cos(θ), l, 0)
    else
        return sqrt((2*l+1)/(2π) * factorial(l-abs(m))/factorial(l+abs(m)))*LegendrePolynomials.Plm(cos(θ), l, abs(m))*cos(abs(m)*ϕ)
    end
end

function eval_naive(ch::CircularHarmonic, Ω::VectorValue{2})
    l, m = degree(ch), order(ch)
    θ = unitcircle_cartesian_to_polar(Ω)
    
    norm = l == 0 ? 1/sqrt(2π) : 1/sqrt(π)
    if m == 0
        return norm * cos(l*θ)
    else # m == 1
        return norm * sin(l*θ)
    end
end

function (ch::CircularHarmonic)(Ω::VectorValue{2})
    return eval_naive(ch, Ω)
end

function (ch_vec::AbstractVector{<:CircularHarmonic})(Ω::VectorValue{2})
    return [ch(Ω) for ch in ch_vec]
end

function (sh::SphericalHarmonic)(Ω::VectorValue)
    # TODO (check): we mirror x and y to fit the definition on wikipedia https://en.wikipedia.org/wiki/Spherical_harmonics
    θ, ϕ = unitsphere_cartesian_to_spherical(VectorValue(Ωz(Ω), -Ωx(Ω), -Ωy(Ω)))
    if isnothing(sh.cache)
        return SphericalHarmonics.computeYlm(θ, ϕ, lmax=degree(sh), SHType=SphericalHarmonics.RealHarmonics())[(degree(sh), order(sh))]
    else
        eval_cache!(sh.cache, θ, ϕ)
        return sh.cache.Y[(degree(sh), order(sh))]
    end
end

function eval_vec!(y, sh_vec::AbstractVector{<:SphericalHarmonic}, Ω::VectorValue)
    # we assume that all sh share the same cache (otherwise the access [(.., ..)] will fail anyways..)
    cache = first(sh_vec).cache
    θ, ϕ = unitsphere_cartesian_to_spherical(VectorValue(Ωz(Ω), -Ωx(Ω), -Ωy(Ω)))
    eval_cache!(cache, θ, ϕ)
    for (i, sh) in enumerate(sh_vec)
        y[i] = cache.Y[(degree(sh), order(sh))]
    end
    return y
end


function (sh_vec::AbstractVector{<:SphericalHarmonic})(Ω::VectorValue)
    eval_vec!(zeros(length(sh_vec)), sh_vec, Ω)
end

"""
    classification of spherical/circular harmonics in even f(Ω) = f(-Ω) and odd f(Ω) = -f(-Ω) functions
"""
is_even(sh::RealHarmonic) = mod(degree(sh), 2) == 0
is_odd(sh::RealHarmonic) = !is_even(sh)

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

"""
    classification of circular harmonics with respect to the cartesian unit vectors
"""

function is_even_in(ch::CircularHarmonic, ::Z)
    return mod(degree(ch), 2) == order(ch)
end

function is_even_in(ch::CircularHarmonic, ::X)
    return order(ch) == 0
end

function is_even_in(sh, n::VectorValue{3})
    @warn "not reliable"
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

is_odd_in(sh::RealHarmonic, d::SpaceDimension) = !is_even_in(sh, d)
is_odd_in(sh, n::VectorValue) = !is_even_in(sh, n)

function get_eee(sh::SphericalHarmonic)
    return map(d -> is_even_in(sh, d), dimensions())
end

function has_same_eee(m1, m2)
    return all(get_eee(m1) == get_eee(m2))
end

Base.:(==)(::RealHarmonic, ::RealHarmonic) = false
Base.:(==)(m1::RealHarmonic{D}, m2::RealHarmonic{D}) where {D} = degree(m1) == degree(m2) && order(m1) == order(m2)

Base.isless(m1::RealHarmonic, m2::RealHarmonic) = (degree(m1) < degree(m2)) || (degree(m1) == degree(m2) && order(m1) < order(m2))
isless_evenodd(m1::RealHarmonic, m2::RealHarmonic) = (is_even(m1) == is_even(m2)) ? isless(m1, m2) : is_even(m1)

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
