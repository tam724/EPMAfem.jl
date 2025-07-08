using Revise
using EPMAfem
using EPMAfem.Gridap
using LinearAlgebra
include("plot_overloads.jl")

lazy(A) = EPMAfem.lazy(A)
kron_AXB = EPMAfem.kron_AXB

space_model = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((-1, 0, -1, 1, -1, 1), (10, 10, 10)))
direction_model = EPMAfem.SphericalHarmonicsModels.EOSphericalHarmonicsModel(5, 3)

# space_model = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((-1, 0), 10))
# direction_model = EPMAfem.SphericalHarmonicsModels.EOSphericalHarmonicsModel(5, 1)

equations = EPMAfem.PNEquations()
model = EPMAfem.DiscretePNModel(space_model, 0:0.01:1.0, direction_model)

problem = EPMAfem.discretize_problem(equations, model, EPMAfem.cpu())

system = EPMAfem.implicit_midpoint(problem, EPMAfem.PNKrylovMinresSolver)

## switch off the coefficients in A
system.A.sym[] = false
system.A.Δ[] = 1.0
system.A.γ[] = 1.0
system.A.δ[] = 1.0
system.A.δT[] = 1.0

function build_blockmatrix(problem)
    nd, ne, nσ = EPMAfem.n_sums(problem)
    T = EPMAfem.base_type(problem.arch)
    coeff_a = [Ref(zero(T)) for _ in 1:ne]
    coeff_c = [[Ref(zero(T)) for _ in 1:nσ] for _ in 1:ne]
    coeff_b = [Ref(zero(T)) for _ in 1:nd]

    lz = EPMAfem.lazy

    ρp = lz.(problem.space_discretization.ρp)
    ρm = lz.(problem.space_discretization.ρm)
    ∇pm = lz.(problem.space_discretization.∇pm)
    ∂p = lz.(problem.space_discretization.∂p)

    Ip = lz(problem.direction_discretization.Ip)
    Im = lz(problem.direction_discretization.Im)
    kp = [lz.(problem.direction_discretization.kp[i]) for i in 1:ne]
    km = [lz.(problem.direction_discretization.km[i]) for i in 1:ne]

    Ωpm = lz.(identity.(problem.direction_discretization.Ωpm))
    absΩp = lz.(identity.(problem.direction_discretization.absΩp))


    U = lz(rand(size(ρp[1], 1), 10))
    Ut = transpose(U)
    V = lz(rand(size(Ip, 1), 10))
    Vt = transpose(V)

    kron_aXb(A, B) = kron_AXB(B, A)
    m(A) = EPMAfem.materialize(A)
    c(A) = EPMAfem.cache(A)

    A = sum(kron_aXb(c(Ut*∂p[i]*U), c(Vt*absΩp[i]*V)) for i in 1:nd) + sum(kron_aXb(c(Ut*ρp[i]*U), c(Vt*(coeff_a[i]*Ip + sum(coeff_c[i][j]*kp[i][j] for j in 1:nσ))*V)) for i in 1:ne)
    C = sum(kron_aXb(ρm[i], m(coeff_a[i]*Im + sum(coeff_c[i][j]*km[i][j] for j in 1:nσ))) for i in 1:ne)
    B = transpose(sum(kron_aXb(Ut*∇pm[i], Ωpm[i]*V) for i in 1:nd))

    # A = sum(kron_aXb(∂p[i], absΩp[i]) for i in 1:nd) + sum(kron_aXb(ρp[i], coeff_a[i]*Ip + sum(coeff_c[i][j]*kp[i][j] for j in 1:nσ)) for i in 1:ne)
    # C = sum(kron_aXb(ρm[i], coeff_a[i]*Im + sum(coeff_c[i][j]*km[i][j] for j in 1:nσ)) for i in 1:ne)
    # B = transpose(sum(kron_aXb(∇pm[i], Ωpm[i]) for i in 1:nd))
    return EPMAfem.blockmatrix(A, B, C), (a=coeff_a, b=coeff_b, c=coeff_c)
end

B, coeffs = build_blockmatrix(problem)

for i in 1:ne
    system.coeffs.a[i] = rand()
    coeffs.a[i][] = system.coeffs.a[i]
    for j in 1:nσ
        system.coeffs.c[i][j] = rand()
        coeffs.c[i][j][] = system.coeffs.c[i][j]
    end
end


function lazy_kron!(Y, A, B, X, temp)
    X_r = reshape(X, 100, 100, 500)
    Y_r = reshape(Y, 100, 100, 500)
    for i in 1:500
        mul!(temp, @view(X_r[:, :, i]), transpose(A))
        mul!(@view(Y_r[:, :, i]), B, temp)
        # Y_r[:, :, i] .= B * X_r[:, :, i] * transpose(A)
    end
end

A = sprand(100, 100, 0.05)
B = sprand(100, 100, 0.05)

AkronB = kron(A, B)

X = rand(100*100, 500)
Y1 = zeros(100*100, 500)
Y2 = zeros(100*100, 500)

temp = zeros(100, 100)

mul!(Y1, AkronB, X)
lazy_kron!(Y2, A, B, X, temp)

using BenchmarkTools

@profview @benchmark mul!($Y1, $AkronB, $X)
@profview @benchmark lazy_kron!($Y2, $A, $B, $X, $temp)

Y1 ≈ Y2

maximum(abs.(Y .- AkronB*X))

A = sprand(100, 100, 0.02)
B = Diagonal(rand(100))

AL = lazy(A)
BL = lazy(B)

X = rand(10000, 30)
Y1 = rand(10000, 30)
Y2 = rand(10000, 30)

KL = kron(lazy(A), lazy(B))
KL2 = kron(lazy(Matrix(A)), lazy(Matrix(B)))

ws = EPMAfem.create_workspace(EPMAfem.required_workspace(EPMAfem.mul_with!, KL), zeros)
@benchmark EPMAfem.mul_with!(ws, Y1, transpose(KL), X, true, false)
@benchmark EPMAfem.mul_with!(ws, Y1, transpose(KL2), X, true, false)
Kd = transpose(kron(A, B))
@benchmark mul!(Y2, Kd, X)

Y2 ≈ Y1
