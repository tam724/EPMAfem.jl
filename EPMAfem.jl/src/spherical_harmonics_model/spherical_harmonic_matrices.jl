const ∫S²_uv = Val{:∫S²_uv}()
const ∫S²_Ωzuv = Val{:∫S²_Ωzuv}()
const ∫S²_Ωxuv = Val{:∫S²_Ωxuv}()
const ∫S²_Ωyuv = Val{:∫S²_Ωyuv}()

const ∫S²_absΩzuv = Val{:∫S²_absΩzuv}()
const ∫S²_absΩxuv = Val{:∫S²_absΩxuv}()
const ∫S²_absΩyuv = Val{:∫S²_absΩyuv}()

dim(::Val{:∫S²_Ωzuv}) = Z()
dim(::Val{:∫S²_Ωxuv}) = X()
dim(::Val{:∫S²_Ωyuv}) = Y()

dim(::Val{:∫S²_absΩzuv}) = Z()
dim(::Val{:∫S²_absΩxuv}) = X()
dim(::Val{:∫S²_absΩyuv}) = Y()

Ωz(Ω::VectorValue{3}) = Ω[1]
Ωx(Ω::VectorValue{3}) = Ω[2]
Ωy(Ω::VectorValue{3}) = Ω[3]

int_func(::Val{:∫S²_uv}, Ω) = 1
int_func(::Val{:∫S²_Ωxuv}, Ω) = Ωx(Ω)
int_func(::Val{:∫S²_Ωyuv}, Ω) = Ωy(Ω)
int_func(::Val{:∫S²_Ωzuv}, Ω) = Ωz(Ω)
int_func(::Val{:∫S²_absΩxuv}, Ω) = abs(Ωx(Ω))
int_func(::Val{:∫S²_absΩyuv}, Ω) = abs(Ωy(Ω))
int_func(::Val{:∫S²_absΩzuv}, Ω) = abs(Ωz(Ω))

const IntFuncIntegral = Union{Val{:∫S²_uv}, Val{:∫S²_Ωzuv}, Val{:∫S²_Ωxuv}, Val{:∫S²_Ωyuv}, Val{:∫S²_absΩxuv}, Val{:∫S²_absΩyuv}, Val{:∫S²_absΩzuv}}

function lebedev_quadrature(f!, cache, model)
    N = max_degree(model)*3 #TODO: this should be checked somehow..
    available_orders = getavailableorders()
    idx = findfirst(o -> o > N, available_orders)
    x, y, z, w = lebedev_by_order(available_orders[idx])
    I = zero(cache)
    for (x_, y_, z_, w_) in zip(x, y, z, w)
        f!(cache, VectorValue(z_, x_, y_))
        I .+= w_ .* cache
    end
    return 4π.*I
end

function hcubature_quadrature(f!, cache, model, tol=1e-10)
    function integrand((θ, ϕ))
        Ω = VectorValue(sin(θ)*cos(ϕ), sin(θ)*sin(ϕ), cos(θ))
        f!(cache, Ω)
        return cache.*sin(θ)
    end
    return hcubature(integrand, (0, 0), (π, 2π), atol=tol, rtol=tol)[1]
end

## dummy function to dispatch on
function exact_quadrature() end 

const Quadrature = Union{typeof(lebedev_quadrature), typeof(hcubature_quadrature)}

function assemble_bilinear(integral::IntFuncIntegral, model, U, V, quad::Quadrature=lebedev_quadrature)
    cache = zeros(length(V), length(U))
    function f!(cache, Ω)
        Y_U, Y_V = _eval_basis_functions!(model, Ω, U, V)
        mul!(cache, Y_V, transpose(Y_U), int_func(integral, Ω), 0.0)
    end
    return quad(f!, cache, model)
end

function assemble_bilinear(::Val{:∫S²_uv}, model, U, V, ::typeof(exact_quadrature))
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

function assemble_bilinear(integral::Union{Val{:∫S²_Ωzuv}, Val{:∫S²_Ωxuv}, Val{:∫S²_Ωyuv}}, model, U, V, ::typeof(exact_quadrature))
    all_harmonics = SphericalHarmonics.ML(0:max_degree(model)) |> collect
    A = zeros(length(V), length(U))

    for i in eachindex(V)
        for j in eachindex(U)
            m1 = SphericalHarmonic(all_harmonics[V[i]]...)
            m2 = SphericalHarmonic(all_harmonics[U[j]]...)
            A[i, j] = get_transport_coefficient(m1, m2, dim(integral))
        end
    end
    return A
end

function assemble_bilinear(integral::Union{Val{:∫S²_absΩzuv}, Val{:∫S²_absΩxuv}, Val{:∫S²_absΩyuv}}, model, U, V, ::typeof(exact_quadrature))
    all_harmonics = SphericalHarmonics.ML(0:max_degree(model)) |> collect
    A = zeros(length(V), length(U))

    for i in eachindex(V)
        for j in eachindex(U)
            m1 = SphericalHarmonic(all_harmonics[V[i]]...)
            m2 = SphericalHarmonic(all_harmonics[U[j]]...)
            A[i, j] = get_cached_boundary_coefficient(m1, m2, dim(integral))
        end
    end
    return A
end