const ∫S²_uv = Val{:∫S²_uv}()
@concrete struct ∫S²_μuv{F}
    μ::F
end

const ∫S²_Ωzuv = Val{:∫S²_Ωzuv}()
const ∫S²_Ωxuv = Val{:∫S²_Ωxuv}()
const ∫S²_Ωyuv = Val{:∫S²_Ωyuv}()

∫S²_Ωuv(::_1D) = (∫S²_Ωzuv, )
∫S²_Ωuv(::_2D) = (∫S²_Ωzuv, ∫S²_Ωxuv)
∫S²_Ωuv(::_3D) = (∫S²_Ωzuv, ∫S²_Ωxuv, ∫S²_Ωyuv)

const ∫S²_absΩzuv = Val{:∫S²_absΩzuv}()
const ∫S²_absΩxuv = Val{:∫S²_absΩxuv}()
const ∫S²_absΩyuv = Val{:∫S²_absΩyuv}()

∫S²_absΩuv(::_1D) = (∫S²_absΩzuv, )
∫S²_absΩuv(::_2D) = (∫S²_absΩzuv, ∫S²_absΩxuv)
∫S²_absΩuv(::_3D) = (∫S²_absΩzuv, ∫S²_absΩxuv, ∫S²_absΩyuv)

dim(::Val{:∫S²_Ωzuv}) = Z()
dim(::Val{:∫S²_Ωxuv}) = X()
dim(::Val{:∫S²_Ωyuv}) = Y()

dim(::Val{:∫S²_absΩzuv}) = Z()
dim(::Val{:∫S²_absΩxuv}) = X()
dim(::Val{:∫S²_absΩyuv}) = Y()

int_func(::Val{:∫S²_uv}, Ω) = 1
int_func(int::∫S²_μuv, Ω) = int.μ(Ω)
int_func(::Val{:∫S²_Ωxuv}, Ω) = Ωx(Ω)
int_func(::Val{:∫S²_Ωyuv}, Ω) = Ωy(Ω)
int_func(::Val{:∫S²_Ωzuv}, Ω) = Ωz(Ω)
int_func(::Val{:∫S²_absΩxuv}, Ω) = abs(Ωx(Ω))
int_func(::Val{:∫S²_absΩyuv}, Ω) = abs(Ωy(Ω))
int_func(::Val{:∫S²_absΩzuv}, Ω) = abs(Ωz(Ω))
struct ∫∫S²_kuv{F}
    k::F
end

struct ∫S²_kuv{F}
    k::F
end

const IntFuncIntegral = Union{Val{:∫S²_uv}, ∫S²_μuv, Val{:∫S²_Ωzuv}, Val{:∫S²_Ωxuv}, Val{:∫S²_Ωyuv}, Val{:∫S²_absΩxuv}, Val{:∫S²_absΩyuv}, Val{:∫S²_absΩzuv}}

"""
    Abstract type for quadratures on the n-sphere (n = D-1)
"""
abstract type NSphericalQuadrature{D} end
struct LebedevQuadrature{D} <: NSphericalQuadrature{D}
    order::Int64
end

LebedevQuadrature{3}() = LebedevQuadrature{3}(getavailableorders()[end])
LebedevQuadrature{2}() = LebedevQuadrature{2}(4*80)

lebedev_quadrature_max() = LebedevQuadrature{3}()

function lebedev_points(quad::LebedevQuadrature{3})
    x, y, z, w = lebedev_by_order(quad.order)
    Ω = to_Ω.(z, x, y)
    return Ω, 4π*w
end

function lebedev_points(quad::LebedevQuadrature{2})
    θ = (π/quad.order).* (2 .* (0:quad.order-1) .+ 1) # shift the points away from the axis.
    Ω = unitcircle_polar_to_cartesian.(θ)
    return Ω, fill(2π/quad.order, quad.order)
end

struct HCubatureQuadrature{D} <: NSphericalQuadrature{D}
    atol::Float64
    rtol::Float64
    maxevals::Int64
end

HCubatureQuadrature{D}(atol, rtol) where {D} = HCubatureQuadrature{D}(atol, rtol, typemax(Int))

struct ExactQuadrature{D} <: NSphericalQuadrature{D} end

function (quad::LebedevQuadrature)(f!, cache)
    Ω, w = lebedev_points(quad)
    I = zero(cache)
    for (Ω_, w_) in zip(Ω, w)
        f!(cache, Ω_)
        I .+= w_ .* cache
    end
    return I
end

function (quad::HCubatureQuadrature{3})(f!, cache)
    function integrand((θ, ϕ))
        Ω = unitsphere_spherical_to_cartesian((θ, ϕ))
        f!(cache, Ω)
        return cache.*sin(θ)
    end
    return hcubature(integrand, (0, 0), (π, 2π), atol=quad.atol, rtol=quad.rtol, maxevals=quad.maxevals)[1]
end

function (quad::HCubatureQuadrature{2})(f!, cache)
    function integrand(θ)
        Ω = unitcircle_polar_to_cartesian(θ)
        f!(cache, Ω)
        return cache
    end
    return hquadrature(integrand, 0, 2π, atol=quad.atol, rtol=quad.rtol, maxevals=quad.maxevals)[1]
end

function (quad::NSphericalQuadrature{D})(f::Function) where {D}
    # evaluate once to compute cache size
    Ω = VectorValue((randn() for _ in 1:D)...) |> normalize
    y = f(Ω)
    isscalar = !(y isa AbstractArray)
    cache = zeros(size(y))
    function f!(cache, Ω)
        cache[:] .= f(Ω)
    end
    I = quad(f!, cache)
    if isscalar
        return I[1]
    else
        return I
    end
end

# 1D expansions

abstract type Abstract1DBasisExp end
struct LegendreBasisExp <: Abstract1DBasisExp
    coeffs::Vector{Float64}
end
coef(basis_exp::LegendreBasisExp, _, u) = basis_exp.coeffs[degree(u) + 1]
function (f::LegendreBasisExp)(μ)
    inf_it = LegendrePolynomials.LegendrePolynomialIterator(μ)
    val = zero(μ)
    l = 0
    for (c_l, Pl) in zip(f.coeffs, inf_it)
        val += (2*l+1)/(4π)*c_l*Pl
        l+=1
    end
    return val
end

function expand_legendre(f, N, quad::HCubatureQuadrature)
    cache = LegendrePolynomials.OffsetVector{Float64}(undef, 0:N)
    c = hquadrature(μ -> 2π*f(μ).*collectPl!(cache, μ, lmax=N), -1.0, 1.0, rtol=quad.rtol, atol=quad.atol, maxevals=quad.maxevals)[1]
    return LegendreBasisExp(c.parent)
end

struct FourierBasisExp <: Abstract1DBasisExp
    coeffs::Vector{Float64}
end
coef(basis_exp::FourierBasisExp, _, u) = basis_exp.coeffs[degree(u) + 1]
function (f::FourierBasisExp)(μ)
    val = zero(μ)
    for l in 0:length(f.coeffs)-1
        norm = l == 0 ? 2π : π
        val += f.coeffs[l+1]*cos(l*acos(μ))/norm
    end
    return val
end

function collectFourier!(cache, θ, N)
    for l in 0:N-1
        cache[l+1] = cos(l*θ)
    end
    return cache
end
function expand_fourier(f, N, quad::HCubatureQuadrature)
    cache = zeros(N+1) 
    c = hquadrature(θ -> f(cos(θ))*collectFourier!(cache, θ, N+1), 0, 2π, rtol=quad.rtol, atol=quad.atol, maxevals=quad.maxevals)[1]
    return FourierBasisExp(c)
end

expand_1D(f, N, quad::HCubatureQuadrature{3}) = expand_legendre(f, N, quad)
expand_1D(f, N, quad::HCubatureQuadrature{2}) = expand_fourier(f, N, quad)
struct ExpFilter <: Abstract1DBasisExp
    α::Float64
end

function coef(basis_exp::ExpFilter, model, u)
    l = degree(u)
    # TODO: this should be the numerical precision of the discretization (which is not always Float64)
    return log(eps(Float64)) * (l / (max_degree(model) + 1))^basis_exp.α
end

function (f::ExpFilter)(_)
    error("Cannot evaluate the filter!")
end

function assemble_bilinear(integral::∫S²_kuv{<:Function}, model::AbstractHarmonicsModel{D}, U, V, quad::HCubatureQuadrature{D}) where D
    N = max_degree(model)
    expansion_1D = expand_1D(integral.k, N, quad)
    return assemble_bilinear(∫S²_kuv(expansion_1D), model, U, V, quad)
end

function assemble_bilinear(integral::∫S²_kuv{<:Abstract1DBasisExp}, model::AbstractHarmonicsModel{D}, U, V, ::NSphericalQuadrature{D}) where {D}
    A = zeros(length(V), length(U))
    for (i, v) in enumerate(V)
        for (j, u) in enumerate(U)
            if u == v # isotropic scattering is diagonal (in spherical harmonic basis)
                A[i, j] = coef(integral.k, model, u)
            end
        end
    end
    return A
end

function assemble_bilinear(integral::IntFuncIntegral, model::AbstractHarmonicsModel{D}, U, V, quad::NSphericalQuadrature{D}=LedebevQuadrature{Ð}()) where {D}
    cache = zeros(length(V), length(U))
    function f!(cache, Ω)
        Y_U, Y_V = _eval_basis_functions!(model, Ω, U, V)
        mul!(cache, Y_V, transpose(Y_U), int_func(integral, Ω), false)
    end
    return quad(f!, cache)
end

function assemble_bilinear(::Val{:∫S²_uv}, model::AbstractHarmonicsModel{D}, U, V, ::ExactQuadrature{D}) where {D}
    if U == V
        return Diagonal(ones(length(U)))
    end
    A = zeros(length(V), length(U))
    for i in eachindex(V)
        for j in eachindex(U)
            A[i, j] = V[i] == U[j] ? 1.0 : 0.0
        end
    end
    return A
end

function assemble_bilinear(integral::Union{Val{:∫S²_Ωzuv}, Val{:∫S²_Ωxuv}, Val{:∫S²_Ωyuv}}, ::AbstractHarmonicsModel{D}, U, V, ::ExactQuadrature{D}) where {D}
    A = zeros(length(V), length(U))
    for (i, m1) in enumerate(V)
        for (j, m2) in enumerate(U)
            A[i, j] = get_transport_coefficient(m1, m2, dim(integral))
        end
    end
    return A
end

function assemble_bilinear(integral::Union{Val{:∫S²_absΩzuv}, Val{:∫S²_absΩxuv}, Val{:∫S²_absΩyuv}}, model::AbstractHarmonicsModel{D}, U, V, ::ExactQuadrature{D}) where {D}
    try
        A = zeros(length(V), length(U))
        for (i, m1) in enumerate(V)
            for (j, m2) in enumerate(U)
                A[i, j] = get_cached_boundary_coefficient(m1, m2, dim(integral))
            end
        end
        return A
    catch e
        if e isa DomainError
            @warn "Boundary Matrix Value not precomputed, falling back to numerical quadrature"
            return assemble_bilinear(integral, model, U, V)
        else
            error("whats wrong here?")
        end
    end
    return A
end

function assemble_bilinear(integral::∫∫S²_kuv, model::AbstractHarmonicsModel{D}, U, V, quad::NSphericalQuadrature{D}=LedebevQuadrature{D}()) where {D}
    cache1 = zeros(length(V), length(U))
    cache2 = zeros(length(V), length(U))
    function fᵤ!(cache1, Ωᵤ)
        Y_U = _eval_basis_functions!(model, Ωᵤ, U)
        function fᵥ!(cache2, Ωᵥ)
            Y_V = _eval_basis_functions!(model, Ωᵥ, V)
            mul!(cache2, Y_V, transpose(Y_U), integral.k(Ωᵤ, Ωᵥ), false)
        end
        cache1 .= quad(fᵥ!, cache2)
    end
    return quad(fᵤ!, cache1)
end
