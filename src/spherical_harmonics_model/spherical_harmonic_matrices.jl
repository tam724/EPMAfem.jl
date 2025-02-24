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

@concrete struct ∫S²_kuv
    k
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
    N = max_degree(model)*fac #TODO: this should be checked somehow..
    available_orders = getavailableorders()
    idx = findfirst(o -> o > N, available_orders)
    if isnothing(idx)
        return available_orders[end]
    end
    return available_orders[idx]
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

function assemble_bilinear(integral::∫S²_kuv, model, U, V, quad::hcubature_quadrature)
    N = max_degree(model)
    Σl = 2*π*hquadrature(μ -> integral.k(μ).*Pl.(μ, 0:N), -1.0, 1.0, rtol=quad.rtol, atol=quad.atol, maxevals=quad.maxevals)[1]
    A = zeros(length(V), length(U))

    for (i, v) in enumerate(V)
        for (j, u) in enumerate(U)
            if u == v # isotropic scattering is diagonal (in spherical harmonic basis)
                l = degree(u) #  == degree(v)
                A[i, j] = Σl[l+1]
            else
                A[i, j] = 0.0
            end
        end
    end
    return A
    # return Diagonal([Σl[l+1] for (l, k) in get_even_moments(N, nd)]), Diagonal([Σl[l+1] for (l, k) in get_odd_moments(N, nd)])
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
    A = zeros(length(V), length(U))
    for (i, m1) in enumerate(V)
        for (j, m2) in enumerate(U)
            A[i, j] = get_cached_boundary_coefficient(m1, m2, dim(integral))
        end
    end
    return A
end