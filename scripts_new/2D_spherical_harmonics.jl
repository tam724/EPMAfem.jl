struct SphericalHarmonic2D{T<:Integer}
    degree::T
    order::T

    function SphericalHarmonic2D(degree, order)
        deg = (degree >= 0) ? degree : error("degree = $(degree) must be >= 0")
        ord = (order >= 0 && order <= degree && order <=1) ? order : error("order must be {0, 1} (for deg 0, order = 0)")
        T = promote_type(typeof(deg), typeof(ord))
        return new{T}(deg, ord)
    end
end

SH.degree(sh::SphericalHarmonic2D) = sh.degree
SH.order(sh::SphericalHarmonic2D) = sh.order

SH.degreeorder(sh::SphericalHarmonic2D) = degree(sh), order(sh)

function (sh::SphericalHarmonic2D)(Ω::VectorValue{2})
    norm = 1/sqrt(pi)
    z, x = Ω
    @assert SH.degree(sh) <= 13
    if SH.degree(sh) == 0
        if SH.order(sh) == 0
            return norm/sqrt(2)
        end
    elseif SH.degree(sh) == 1
        if SH.order(sh) == 0
            return norm*z
        elseif SH.order(sh) == 1
            return norm*x
        end
    elseif SH.degree(sh) == 2
        if SH.order(sh) == 0
            return norm * (z^2 - x^2)
        elseif SH.order(sh) == 1
            return norm * 2*z*x
        end
    elseif SH.degree(sh) == 3
        if SH.order(sh) == 0
            return norm * z*(z^2 - 3*x^2)
        elseif SH.order(sh) == 1
            return norm * x*(3*z^2 - x^2)
        end
    elseif SH.degree(sh) == 4
        if SH.order(sh) == 0
            return norm * (z^4 - 6*z^2*x^2 + x^4)
        elseif SH.order(sh) == 1
            return norm * (4*z^3*x - 4*z*x^3)
        end
    elseif SH.degree(sh) == 5
        if SH.order(sh) == 0
            return norm * z*(5*x^4 - 10*z^2*x^2 + z^4)
        elseif SH.order(sh) == 1
            return norm * x*(5*z^4 - 10*z^2*x^2 + x^4)
        end
    elseif SH.degree(sh) == 6
        if SH.order(sh) == 0
            return norm * (z^6 - 15*z^4*x^2 + 15*z^2*x^4 - x^6)
        elseif SH.order(sh) == 1
            return norm * (6*z^5*x - 20*z^3*x^3 + 6*z*x^5)
        end
    elseif SH.degree(sh) == 7
        if SH.order(sh) == 0
            return norm*(z*(z^6 - 21*z^4*x^2 + 35*z^2*x^4 - 7*x^6))
        elseif SH.order(sh) == 1
            return norm*(x*(7*z^6 - 35*z^4*x^2 + 21*z^2*x^4 - x^6))
        end
    elseif SH.degree(sh) == 8
        if SH.order(sh) == 0
            return norm*(z^8 - 28*z^6*x^2 + 70*z^4*x^4 - 28*z^2*x^6 + x^8)
        elseif SH.order(sh) == 1
            return norm*(8*z^7*x - 56*z^5*x^3 + 56*z^3*x^5 - 8*z*x^7)
        end
    elseif SH.degree(sh) == 9
        if SH.order(sh) == 0
            return norm*(z*(z^8 - 36*z^6*x^2 + 126*z^4*x^4 - 84*z^2*x^6 + 9*x^8))
        elseif SH.order(sh) == 1
            return norm*(x*(9*z^8 - 84*z^6*x^2 + 126*z^4*x^4 - 36*z^2*x^6 + x^8))
        end
    elseif SH.degree(sh) == 10
        if SH.order(sh) == 0
            return norm*(z^10 - 45*z^8*x^2 + 210*z^6*x^4 - 210*z^4*x^6 + 45*z^2*x^8 - x^10)
        elseif SH.order(sh) == 1
            return norm*(10*z^9*x - 120*z^7*x^3 + 252*z^5*x^5 - 120*z^3*x^7 + 10*z*x^9)
        end
    elseif SH.degree(sh) == 11
        if SH.order(sh) == 0
            return norm*(z*(z^10 - 55*z^8*x^2 + 330*z^6*x^4 - 462*z^4*x^6 + 165*z^2*x^8 - 11*x^10))
        elseif SH.order(sh) == 1
            return norm*(x*(11*z^10 - 165*z^8*x^2 + 462*z^6*x^4 - 330*z^4*x^6 + 55*z^2*x^8 - x^10))
        end
    elseif SH.degree(sh) == 12
        if SH.order(sh) == 0
            return norm*(z^12 - 66*z^10*x^2 + 495*z^8*x^4 - 924*z^6*x^6 + 495*z^4*x^8 - 66*z^2*x^10 + x^12)
        elseif SH.order(sh) == 1
            return norm*(12*z^11*x - 220*z^9*x^3 + 792*z^7*x^5 - 792*z^5*x^7 + 220*z^3*x^9 - 12*z*x^11)
        end
    elseif SH.degree(sh) == 13
        if SH.order(sh) == 0
            return norm*(z*(z^12 - 78*z^10*x^2 + 715*z^8*x^4 - 1716*z^6*x^6 + 1287*z^4*x^8 - 286*z^2*x^10 + 13*x^12))
        elseif SH.order(sh) == 1
            return norm*(x*(13*z^12 - 286*z^10*x^2 + 1287*z^8*x^4 - 1716*z^6*x^6 + 715*z^4*x^8 - 78*z^2*x^10 + x^12))
        end
    end
    error("Not Implemented.")
end

function SH.is_even(sh::SphericalHarmonic2D)
    return iseven(SH.degree(sh))
end

function SH.is_odd(sh::SphericalHarmonic2D)
    return isodd(SH.degree(sh))
end

function SH.get_all_viable_harmonics_up_to(N)
    return [SphericalHarmonic2D(l, k) for l in 0:N for k in ((l==0) ? (0:0) : (0:1))]
end

Ω2D(θ) = VectorValue(cos(θ), sin(θ))

function plot_spherical_harmonic(sh; cmap=:default, kwargs...)
    z = [cos(θ)*abs(sh(Ω2D(θ))) for θ in 0:0.01:2π]
    x = [sin(θ)*abs(sh(Ω2D(θ))) for θ in 0:0.01:2π]
    cval = [sh(Ω2D(θ)) for θ in 0:0.01:2π]
    cmin, cmax = extrema(cval)
    color = cgrad(cmap)[(cval .- cmin)/(cmax - cmin)]
    # @show color
    
    plot!(z, x, color=color; kwargs...)
end

function Plots.plot!(sh::SphericalHarmonic2D; kwargs...)
    plot_spherical_harmonic(sh; kwargs...)
end

function SH.get_transport_coefficient(m1::SphericalHarmonic2D, m2::SphericalHarmonic2D, ::EPMAfem.Dimensions.Z)
    l, k = SH.degree(m1), SH.order(m1)
    l_, k_ = SH.degree(m2), SH.order(m2)
    if abs(l - l_) != 1
        return 0.0
    elseif l == 0 && l_ == 1
        return k_ == 0 ? 1/sqrt(2) : 0.0
    elseif l_ == 0 && l == 1
        return k == 0 ? 1/sqrt(2) : 0.0
    else
        return k == k_ ? 0.5 : 0.0
    end
    error("Not implemented.")
end

function SH.get_transport_coefficient(m1::SphericalHarmonic2D, m2::SphericalHarmonic2D, ::EPMAfem.Dimensions.X)
    l, k = SH.degree(m1), SH.order(m1)
    l_, k_ = SH.degree(m2), SH.order(m2)
    if abs(l - l_) != 1
        return 0.0
    elseif l == 0 && l_ == 1
        return k_ == 1 ? 1/sqrt(2) : 0.0
    elseif l_ == 0 && l == 1
        return k == 1 ? 1/sqrt(2) : 0.0
    else
        if k != k_
            return (((l - l_) > 0 && iszero(k) || (l - l_) < 0 && isone(k)) ? -0.5 : 0.5)
        else
            return 0.0
        end
    end
    error("Not implemented.")
end

SH.get_transport_coefficient(m1::SphericalHarmonic2D, m2::SphericalHarmonic2D, ::EPMAfem.Dimensions.Y) = 0.0

# using HCubature

# moms = get_all_viable_harmonics_up_to(13)

# [hquadrature(θ -> sh1(Ω2D(θ))*sh2(Ω2D(θ)), 0, 2π; atol=1e-14)[1] for sh1 in moms, sh2 in moms] |> diag

# (([hquadrature(θ -> sh1(Ω2D(θ))*sh2(Ω2D(θ)), 0, 2π; atol=1e-14)[1] for sh1 in moms, sh2 in moms] .|> x -> round(x; digits=13)) - Diagonal(ones(27)) .|> abs) |> maximum
# Bz = [hquadrature(θ -> Ω2D(θ)[1]*sh1(Ω2D(θ))*sh2(Ω2D(θ)), 0, 2π; atol=1e-14)[1] for sh1 in moms, sh2 in moms] .|> x -> round(x; digits=13)
# Bx = [hquadrature(θ -> Ω2D(θ)[2]*sh1(Ω2D(θ))*sh2(Ω2D(θ)), 0, 2π; atol=1e-14)[1] for sh1 in moms, sh2 in moms] .|> x -> round(x; digits=13)

# maximum(abs.(Bz - [get_transport_coefficient(sh1, sh2, EPMAfem.Dimensions.Z()) for sh1 in moms, sh2 in moms]))
# maximum(abs.(Bx - [get_transport_coefficient(sh1, sh2, EPMAfem.Dimensions.X()) for sh1 in moms, sh2 in moms]))

# for i in 1:6
#     for j in (i==0 ? [0] : [0, 1])
#         sh = SphericalHarmonic2D(i, j)
#         @show hquadrature(θ -> sh(Ω2D(θ))*sh(Ω2D(θ)), 0, 2π)
#     end
# end

# plot()
# plot!(moms[6])

# function even_n(f, n)
#     return Ω -> 0.5*(f(Ω - 2*dot(n, Ω)n) + f(Ω))
# end

# function odd_n(f, n)
#     return Ω -> 0.5*(f(Ω - 2*dot(n, Ω)n) - f(Ω))
# end

# begin
#     sh = SphericalHarmonic2D(2, 1)
#     plot(0:0.01:2π, θ -> sh(Ω2D(θ)), label="sh")
#     # plot!(0:0.01:2π, θ -> even_n(sh, VectorValue(1.0, 0.0))(Ω2D(θ)), label="sh even (z)")
#     plot!(0:0.01:2π, θ -> odd_n(sh, VectorValue(1.0, 0.0))(Ω2D(θ)), label="sh odd (z)")
#     # plot!(0:0.01:2π, θ -> even_n(sh, VectorValue(0.0, 1.0))(Ω2D(θ)), label="sh even (x)")
#     plot!(0:0.01:2π, θ -> odd_n(sh, VectorValue(0.0, 1.0))(Ω2D(θ)), label="sh odd (x)")
# end

# moms1 = [moms[2:3]..., moms[5]]
# moms2 = [moms[1], moms[4]]

# [hquadrature(θ -> dot(VectorValue(1.0, 0.0), Ω2D(θ))*sh1(Ω2D(θ))*sh2(Ω2D(θ)), 0, 2π; atol=1e-14)[1] for sh1 in moms[2:3], sh2 in test]
# [hquadrature(θ -> dot(VectorValue(0.0, 1.0), Ω2D(θ))*sh1(Ω2D(θ))*sh2(Ω2D(θ)), 0, 2π; atol=1e-14)[1] for sh1 in moms[2:3], sh2 in test]
# [hquadrature(θ -> dot(VectorValue(1.0, 0.0), Ω2D(θ))*sh1(Ω2D(θ))*sh2(Ω2D(θ)), 0, 2π; atol=1e-14)[1] for sh1 in moms[2:3], sh2 in moms[4:5]]
# [hquadrature(θ -> dot(VectorValue(0.0, 1.0), Ω2D(θ))*sh1(Ω2D(θ))*sh2(Ω2D(θ)), 0, 2π; atol=1e-14)[1] for sh1 in moms[2:3], sh2 in moms[4:5]]

# # [hquadrature(θ -> dot(VectorValue(0.4, 0.6) |> normalize, Ω2D(θ))*sh1(Ω2D(θ))*sh2(Ω2D(θ)), 0, 2π; atol=1e-14)[1] for sh1 in moms2, sh2 in moms2]

# test = [
#     Ω -> Ω[1]*Ω[2],
#     Ω -> Ω[1]^2,
#     Ω -> Ω[2]^2,
# ]
