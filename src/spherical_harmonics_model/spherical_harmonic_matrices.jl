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

abstract type AbstractLegendreBasisExp end
@concrete struct LegendreBasisExp <: AbstractLegendreBasisExp
    coeffs
end

function legendre_coeff(basis_exp::LegendreBasisExp, _, u)
    l = degree(u)
    return basis_exp.coeffs[l+1]
end

function (f::LegendreBasisExp)(μ)
    inf_it = LegendrePolynomials.LegendrePolynomialIterator(μ)
    val = zero(μ)
    l = 0
    for (c_l, Pl) in zip(f.coeffs, inf_it)
        val += (2.0*l+1.0)/2.0*c_l*Pl
        l+=1
    end
    return val
end

function expand_legendre(f, N, quad=hcubature_quadrature)
    cache = LegendrePolynomials.OffsetVector{Float64}(undef, 0:N)
    c = hquadrature(μ -> f(μ).*collectPl!(cache, μ, lmax=N), -1.0, 1.0, rtol=quad.rtol, atol=quad.atol, maxevals=quad.maxevals)[1]
    return LegendreBasisExp(c)
end

@concrete struct ExpFilter <: AbstractLegendreBasisExp
    α
end

function legendre_coeff(basis_exp::ExpFilter, model, u)
    l = degree(u)
    # TODO: this should be the numerical precision of the discretization (which is not always Float64)
    return log(eps(Float64)) * (l / (max_degree(model) + 1))^basis_exp.α
end

function (f::ExpFilter)(_)
    error("Cannot evaluate the filter!")
end

struct ∫∫S²_kuv{F}
    k::F
end

struct ∫S²_kuv{F}
    k::F
end

const IntFuncIntegral = Union{Val{:∫S²_uv}, ∫S²_μuv, Val{:∫S²_Ωzuv}, Val{:∫S²_Ωxuv}, Val{:∫S²_Ωyuv}, Val{:∫S²_absΩxuv}, Val{:∫S²_absΩyuv}, Val{:∫S²_absΩzuv}}

abstract type SphericalQuadrature end
struct lebedev_quadrature <: SphericalQuadrature
    order::Int64
end

function lebedev_quadrature_max()
    return lebedev_quadrature(getavailableorders()[end])
end

function guess_lebedev_order_from_model(model, fac=3)
    available_orders = getavailableorders()
    return available_orders[end]
    # N = max_degree(model)*fac #TODO: this should be checked somehow..
    # idx = findfirst(o -> o > N, available_orders)
    # if isnothing(idx)
    #     return available_orders[end]
    # end
    # return available_orders[idx]
end

function lebedev_points(quad)
    x, y, z, w = lebedev_by_order(quad.order)
    Ω = to_Ω.(z, x, y)
    return Ω, 4π*w
end

struct hcubature_quadrature <: SphericalQuadrature
    atol::Float64
    rtol::Float64
    maxevals::Int64
end
hcubature_quadrature(atol, rtol) = hcubature_quadrature(atol, rtol, typemax(Int))

struct exact_quadrature <: SphericalQuadrature end

function (quad::lebedev_quadrature)(f!, cache)
    Ω, w = lebedev_points(quad)
    I = zero(cache)
    for (Ω_, w_) in zip(Ω, w)
        f!(cache, Ω_)
        I .+= w_ .* cache
    end
    return I
end

function (quad::hcubature_quadrature)(f!, cache)
    function integrand((θ, ϕ))
        Ω = unitsphere_spherical_to_cartesian((θ, ϕ))
        f!(cache, Ω)
        return cache.*sin(θ)
    end
    return hcubature(integrand, (0, 0), (π, 2π), atol=quad.atol, rtol=quad.rtol, maxevals=quad.maxevals)[1]
end

function (quad::SphericalQuadrature)(f::Function)
    # evaluate once to compute cache size
    Ω = VectorValue(randn(), randn(), randn()) |> normalize
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

function assemble_bilinear(integral::∫S²_kuv{<:Function}, model, U, V, quad::hcubature_quadrature)
    N = max_degree(model)
    # TODO: cleanup! Here we can use the LegendreBasisExp and remove the duplicate assemble_bilinear routine.
    Σl = 2*π*hquadrature(μ -> integral.k(μ).*collectPl.(μ, lmax=N), -1.0, 1.0, rtol=quad.rtol, atol=quad.atol, maxevals=quad.maxevals)[1]
    A = zeros(length(V), length(U))
    for (i, v) in enumerate(V)
        for (j, u) in enumerate(U)
            if u == v # isotropic scattering is diagonal (in spherical harmonic basis)
                l = degree(u) #  == degree(v)
                A[i, j] = Σl[l]
            else
                A[i, j] = 0.0
            end
        end
    end
    return A
end

function assemble_bilinear(integral::∫S²_kuv{<:AbstractLegendreBasisExp}, model, U, V, _)
    A = zeros(length(V), length(U))
    for (i, v) in enumerate(V)
        for (j, u) in enumerate(U)
            if u == v # isotropic scattering is diagonal (in spherical harmonic basis)
                A[i, j] = legendre_coeff(integral.k, model, u)
            end
        end
    end
    return A
end

function assemble_bilinear(integral::IntFuncIntegral, model, U, V, quad::SphericalQuadrature=lebedev_quadrature(guess_lebedev_order_from_model(model)))
    cache = zeros(length(V), length(U))
    function f!(cache, Ω)
        Y_U, Y_V = _eval_basis_functions!(model, Ω, U, V)
        mul!(cache, Y_V, transpose(Y_U), int_func(integral, Ω), false)
    end
    return quad(f!, cache)
end

function assemble_bilinear(::Val{:∫S²_uv}, model, U, V, ::exact_quadrature)
    if U == V
        A = Diagonal(ones(length(U)))
        return A
    end
    A = zeros(length(V), length(U))
    for i in eachindex(V)
        for j in eachindex(U)
            A[i, j] = V[i] == U[j] ? 1.0 : 0.0
        end
    end
    return A
end

function assemble_bilinear(integral::Union{Val{:∫S²_Ωzuv}, Val{:∫S²_Ωxuv}, Val{:∫S²_Ωyuv}}, model, U, V, ::exact_quadrature)
    A = zeros(length(V), length(U))
    for (i, m1) in enumerate(V)
        for (j, m2) in enumerate(U)
            A[i, j] = get_transport_coefficient(m1, m2, dim(integral))
        end
    end
    return A
end

function assemble_bilinear(integral::Union{Val{:∫S²_absΩzuv}, Val{:∫S²_absΩxuv}, Val{:∫S²_absΩyuv}}, model, U, V, ::exact_quadrature)
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

function assemble_bilinear(integral::∫∫S²_kuv, model, U, V, quad::SphericalQuadrature=lebedev_quadrature(guess_lebedev_order_from_model(model)))
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
