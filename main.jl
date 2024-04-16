using Gridap
using SparseArrays
using StaticArrays
using LinearAlgebra
using LinearSolve
using Plots
using KrylovKit
using Distributions
using Pardiso
using Enzyme
using HCubature

using StaRMAP
using PNSolver

include("spherical_harmonics.jl")
using .SphericalHarmonicsMatrices

include("even_odd_fem.jl")
using .EvenOddFEM

## for comparison
struct TestMaterial{E} <: PNMaterial{1, Float64}
    elms::E
end

import PNSolver.component_densities!
function component_densities!(ρ::AbstractVector, m::TestMaterial, x::AbstractVector)
    ρ .= 1.0
end

struct DummySP <: PNSolver.NeXLCore.BetheEnergyLoss end
import PNSolver:stopping_power_func
function stopping_power_func(::Type{DummySP}, element::PNSolver.NeXLCore.PeriodicTable.Element)
    return ϵ -> -s([ϵ]) + 0.0*ϵ
end

struct DummyTC <: PNSolver.NeXLCore.ElasticScatteringCrossSection end
import PNSolver:transport_coefficient_func
function transport_coefficient_func(TCA::Type{<:DummyTC}, element, Nmax)
    # A_e = convert_strip_mass(element.atomic_mass)
    # ρ = convert_strip_density(element.density)
    tc = zeros(Nmax+1)
    for l = 0:Nmax
        f(x) = scattering_kernel(x) * SphericalHarmonicsMatrices.Pl(x, l)
        # tc[l+1] = 2. * π * hquadrature(f, -1., 1.)[1]
        tc[l+1] = 2*π*hquadrature(f, -1., 1., maxevals=1000)[1] # the 2π probably depends on the definition of the differential scattering cross section (sometimes it might be included, sometimes it might be not..)
    end
    function tcoeff(ϵ)
        # integrate the differential cross section into legendre polynomials
        # ϵ_eV = ϵ # no need to convert as our default unit is eV.
        # tc = zeros(Nmax+1)
        # for l = 0:Nmax
        #     f(x) = δσδΩ(TCA, acos(x), element, ϵ_eV) * Pl(x, l)
        #     # tc[l+1] = 2. * π * hquadrature(f, -1., 1.)[1]
        #     tc[l+1] = hquadrature(f, -1., 1.)[1] # the 2π probably depends on the definition of the differential scattering cross section (sometimes it might be included, sometimes it might be not..)
        # end
        return σ(ϵ)*tc
    end
    return tcoeff
end

# temp = transport_coefficient_func(DummyTC, nothing, 5)

# function δσδΩ(::Type{DummyTC} , θ, element, ϵ)
#     return 0.5*scattering_kernel(cos(θ))*PNSolver.convert_strip_mass(PNSolver.n"Cu".atomic_mass)
# end

gspec = PNSolver.make_gridspec((100, 1, 1), (-4.0, 0.0), 0.0, 0.0)
beam = PNSolver.PNBeam{Float64}(MultivariateNormal([0.0, 0.0], [1.0, 1.0]), VonMisesFisher([1.0, 0.0, 0.0], 10.0), Normal(0.7, 0.09))
problem = PNSolver.ForwardPNProblem{11, Float64}(gspec, [PNSolver.n"Cu"], beam, 1.0, -1.0, 300, PNSolver.PhysicalAlgorithm{DummySP, DummyTC})

m = TestMaterial{typeof([PNSolver.n"Cu"])}([PNSolver.n"Cu"])
save = PNSolver.compute_and_save(problem, m)

@gif for i in 1:300
    plot(save[i][2][:, 1, 1])
    title!("$(i)")
end


# include("quick_and_dirty_1D_even_odd_fem.jl")
# using .EvenOddFiniteElements

# function get_model(::Val{1})
#     return CartesianDiscreteModel((-1.0, 1.0), (50))
# end
# function get_model(::Val{2})
#     return CartesianDiscreteModel((-1.0, 1.0, -1.0, 1.0), (26, 26))
# end

function assemble_bilinear(a, U, V)
    u = get_trial_fe_basis(U)
    v = get_fe_basis(V)
    matcontribs = a(u, v)
    data = Gridap.FESpaces.collect_cell_matrix(U, V, matcontribs)
    return assemble_matrix(SparseMatrixAssembler(U, V), data)
end

function assemble_linear(b, U, V)
    v = get_fe_basis(V)
    veccontribs = b(v)
    data = Gridap.FESpaces.collect_cell_vector(V, veccontribs)
    return assemble_vector(SparseMatrixAssembler(U, V), data)
end

# function get_rhs_x(::Val{1})
#     return x -> -exp(-50*(x[1]-(-0.0))^2)
# end

# function get_rhs_x(::Val{2})
#     return x -> -exp(-50*(x[1]-(-0))^2 - 50*(x[2]-0)^2)
# end

# function get_boundary_condition_x(::Val{1})
#     return x -> exp(-10.0*(x[1] - (-1.0))^2)
# end

# function get_G_xd(::Val{1})
#     g_x_func = get_boundary_condition_x(nd)
#     n = get_normal_vector(∂R)
#     d1 = VectorValue(1.0)
#     g_x_v1(v) = ∫(g_x_func*dot(n,d1)*v)*d∂R
#     G_x1 = assemble_space_rhs(g_x_v1, U_x, V_x)
#     return (G_x1, )
# end

# function get_G_xd(::Val{2})
#     g_x_func = get_boundary_condition_x(nd)
#     n = get_normal_vector(∂R)
#     d1 = VectorValue(1.0, 0.0)
#     g_x_v1(v) = ∫(g_x_func*dot(n,d1)*v)*d∂R
#     G_x1 = assemble_space_rhs(g_x_v1, U_x, V_x)

#     d2 = VectorValue(0.0, 1.0)
#     g_x_v2(v) = ∫(g_x_func*dot(n,d2)*v)*d∂R
#     G_x2 = assemble_space_rhs(g_x_v2, U_x, V_x)
#     return (G_x1, G_x2)
# end

function get_transport_matrices(N, nd::Val{1})
    Apm1 = assemble_transport_matrix(N, Val{3}(), :pm, nd)
    Amp1 = assemble_transport_matrix(N, Val{3}(), :mp, nd)

    return Apm1, Amp1
end

# function get_transport_matrices(N, nd::Val{2})
#     Apm1 = assemble_transport_matrix(N, Val{3}(), :pm, nd)
#     Amp1 = assemble_transport_matrix(N, Val{3}(), :mp, nd)

#     Apm2 = assemble_transport_matrix(N, Val{1}(), :pm, nd)
#     Amp2 = assemble_transport_matrix(N, Val{1}(), :mp4, nd)
#     return (Apm1 + Amp1, Amp2 + Apm2)
# end

function get_∂A(N, nd::Val{1})
    ∂A1 = assemble_boundary_matrix(N, Val{3}(), nd)
    # ∂A2 = assemble_boundary_matrix(N, Val{1}(), nd)
    return (∂A1, )
end

function get_∂A(N, nd::Val{2})
    ∂A1 = assemble_boundary_matrix(N, Val{3}(), nd)
    ∂A2 = assemble_boundary_matrix(N, Val{1}(), nd)
    return (∂A1, ∂A2)
end

function get_b_g_Ω(g_Ω, N, nd::Val{1})
    return (SphericalHarmonicsMatrices.assemble_boundary_source(N, g_Ω, Val{3}(), nd), )
end

function get_b_g_Ω(g_Ω, N, nd::Val{2})
    return (SphericalHarmonicsMatrices.assemble_boundary_source(N, g_Ω, Val{3}(), nd),
        SphericalHarmonicsMatrices.assemble_boundary_source(N, g_Ω, Val{1}(), nd))
end

function get_b_g_Ω(g_Ω, N, nd::Val{3})
    return (SphericalHarmonicsMatrices.assemble_boundary_source(N, g_Ω, Val{3}(), nd),
        SphericalHarmonicsMatrices.assemble_boundary_source(N, g_Ω, Val{1}(), nd),
        SphericalHarmonicsMatrices.assemble_boundary_source(N, g_Ω, Val{2}(), nd))
end

## basis evaluations
function eval_space(U_x, x)
    bp, bm = zeros(num_free_dofs(U_x[1])), zeros(num_free_dofs(U_x[2]))
    e_i_p = zeros(num_free_dofs(U_x[1]))
    e_i_m = zeros(num_free_dofs(U_x[2]))

    for i = 1:num_free_dofs(U_x[1])
        e_i_p[i] = 1.0
        funcp = FEFunction(U_x[1], e_i_p)
        bp[i] = funcp(Point(x))
        # bm[i] = funcm(Point(x))
        e_i_p[i] = 0.0
    end
    for i = 1:num_free_dofs(U_x[2])
        e_i_m[i] = 1.0
        funcm = FEFunction(U_x[2], e_i_m)
        bm[i] = funcm(Point(x))
        # bm[i] = funcm(Point(x))
        e_i_m[i] = 0.0
    end
    return bp, bm
end

function eval_energy(U_ϵ, ϵ)
    b = zeros(num_free_dofs(U_ϵ))
    e_i = zeros(num_free_dofs(U_ϵ))

    for i = 1:num_free_dofs(U_ϵ)
        e_i[i] = 1.0
        (func⁺, func⁻) = FEFunction(U_ϵ, e_i)
        if ϵ > 0
            b[i] = func⁺(Point(-ϵ)) - func⁻(Point(-ϵ))
        else
            b[i] = func⁺(Point(ϵ)) + func⁻(Point(ϵ))
        end
        e_i[i] = 0.0
    end
    return b
end

function eval_average(U_x, U_ϵ, N, u, x, ϵ)
    b_x = eval_space(U_x, x)
    b_ϵ = eval_energy(U_ϵ, ϵ)
    b_Ω = zeros(length(SphericalHarmonicsMatrices.get_moments(N, nd)))
    b_Ω[1] = 1.0
    return dot(u, b_x ⊗ b_ϵ ⊗ b_Ω)
end

# script
nd = Val{1}()

### space definitions
## space 
model_R = CartesianDiscreteModel((-1.0, 1.0), (100))
order_x = 1
refel_x = ReferenceFE(lagrangian, Float64, order_x)
# V_x = TestFESpace(model_R, refel_x, conformity=:H1)
V_x = MultiFieldFESpace([TestFESpace(model_R, ReferenceFE(lagrangian, Float64, order_x), conformity=:H1), TestFESpace(model_R, ReferenceFE(lagrangian, Float64, order_x-1), conformity=:L2)])
U_x = MultiFieldFESpace([TrialFESpace(V_x[1]), TrialFESpace(V_x[2])])

# U_x = TrialFESpace(V_x)
R = Triangulation(model_R)
dx = Measure(R, order_x+1)
∂R = BoundaryTriangulation(model_R)
d∂R = Measure(∂R, order_x+1)
n = get_normal_vector(∂R)

## direction
N = 5
n_dir_basis = length(SphericalHarmonicsMatrices.get_moments(N, nd))
n_dir_basis_p = length([m for m in SphericalHarmonicsMatrices.get_moments(N, nd) if SphericalHarmonicsMatrices.is_even(m...)])
n_dir_basis_m = length([m for m in SphericalHarmonicsMatrices.get_moments(N, nd) if SphericalHarmonicsMatrices.is_odd(m...)])
dir_idx_p = 1:n_dir_basis_p
dir_idx_m = n_dir_basis_p+1:n_dir_basis_p+n_dir_basis_m

## energy
model_E_2 = CartesianDiscreteModel((-1.0, 0.0), (30))
order_ϵ = 2
refel_ϵ = ReferenceFE(lagrangian, Float64, order_ϵ)
V_ϵ = MultiFieldFESpace([TestFESpace(model_E_2, refel_ϵ, conformity=:H1), TestFESpace(model_E_2, refel_ϵ, conformity=:H1, dirichlet_tags=[2])])
U_ϵ = MultiFieldFESpace([TrialFESpace(V_ϵ[1]), TrialFESpace(V_ϵ[2], VectorValue(0.0))])
E_2 = Triangulation(model_E_2)
dϵ = Measure(E_2, order_ϵ + 1)
∂E_2 = BoundaryTriangulation(model_E_2)
d∂E_2 = Measure(∂E_2, order_ϵ + 1)

### matrix assembly

q = (ϵ = (ϵ -> exp(-50.0*(ϵ[1] - (0.7))^2)),
    x = (x -> exp(-50*(x[1]-(0.0))^2)),
    Ω = (Ω -> pdf(VonMisesFisher([0.0, 0.0, -1.0], 10.0), [Ω...])))

g = (ϵ = (ϵ -> pdf(Normal(0.7, 0.09), ϵ[1])),
     x = (x -> isapprox(x[1], 1.0) ? pdf(MultivariateNormal([0.0, 0.0], [1.0, 1.0]), [0.0, 0.0]) : 0.0), 
     Ω = (Ω -> pdf(VonMisesFisher([0.0, 0.0, -1.0], 10.0), [Ω...])))

μ = (ϵ = (ϵ -> exp(-4*(ϵ[1]-0.0)^2)),
     x = (x -> (x[1] < 0.9 && x[1] > 0.8 || x[1] < 0.5 && x[1] > 0.4) ? 1.0 : 0.0),
     Ω = (Ω -> 1.0))

# plot(-1:0.01:1, x -> g.ϵ(Point(x)))
plot(-1:0.01:1, x -> μ.ϵ(Point(x)))

ρ(x) = 1.0

scattering_kernel_(μ) = exp(-4.0*(μ-(1))^2)
scattering_norm_factor = 2*π*hquadrature(x -> scattering_kernel_(x), -1.0, 1.0, rtol=1e-8, atol=1e-8, maxevals=100000)[1]
scattering_kernel(μ) = scattering_kernel_(μ) / scattering_norm_factor

s(ϵ) = -1.0
∂s_2(ϵ) = 0.5 * Enzyme.autodiff(Forward, s, Duplicated(ϵ[1], 1.0))[1]
γ(ϵ) = 0.5
σ(ϵ) = γ(ϵ)

## space
n_space_basis_p = num_free_dofs(U_x[1])
n_space_basis_m = num_free_dofs(U_x[2])
x_idx_p = 1:n_space_basis_p
x_idx_m = n_space_basis_p+1:n_space_basis_p + n_space_basis_m

Xpp = assemble_bilinear(
    ((up, um), (vp, vm)) -> ∫( ρ*(up*vp)) *dx,
    U_x, V_x)[x_idx_p, x_idx_p]

Xmm = assemble_bilinear(
    ((up, um), (vp, vm)) -> ∫( ρ*(um*vm)) *dx,
    U_x, V_x)[x_idx_m, x_idx_m]

dXpm = assemble_bilinear(
    ((up, um), (vp, vm)) -> ∫( dot(VectorValue(1.0), ∇(up))*vm) * dx,
    U_x, V_x)[x_idx_m, x_idx_p]

dXmp = assemble_bilinear(
    ((up, um), (vp, vm)) -> ∫( um * dot(VectorValue(1.0), ∇(vp))) * dx,
    U_x, V_x)[x_idx_p, x_idx_m]

∂Xpp = assemble_bilinear(
    ((up, um), (vp, vm)) -> ∫(abs(dot(n, VectorValue(1.0)))*up*vp)*d∂R,
    U_x, V_x)[x_idx_p, x_idx_p]

b_q_x = assemble_linear(
    ((vp, vm), ) -> ∫(q.x*vp + q.x*vm)*dx,
    U_x, V_x)
b_q_x_p = b_q_x[x_idx_p]
b_q_x_m = b_q_x[x_idx_m]

b_μ_x = assemble_linear(
    ((vp, vm), ) -> ∫(μ.x*vp + μ.x*vm)*dx,
    U_x, V_x)
b_μ_x_p = b_μ_x[x_idx_p]
b_μ_x_m = b_μ_x[x_idx_m]

b_g_x = assemble_linear(
    ((vp, vm), ) -> ∫(dot(n, VectorValue(1.0))*g.x*vp)*d∂R,
    U_x, V_x)
b_g_x_p = b_g_x[x_idx_p]
b_g_x_m = b_g_x[x_idx_m]

## direction
∂App = assemble_boundary_matrix(N, Val(3), :pp, nd)
∂App .= round.(∂App, digits=8)
dApm, dAmp = get_transport_matrices(N, nd)
AIpp = Diagonal(ones(n_dir_basis_p))
AImm = Diagonal(ones(n_dir_basis_m))
K = assemble_scattering_matrix(N, scattering_kernel, nd)
@assert isapprox(K[1, 1], 1.0)
# SphericalHarmonicsMatrices.assemble_gram_matrix(5, nd)
Kpp = sparse(K[dir_idx_p, dir_idx_p])
Kmm = sparse(K[dir_idx_m, dir_idx_m])

b_q_Ω = assemble_direction_rhs(N, q.Ω, nd)
b_q_Ω_p = b_q_Ω[dir_idx_p]
b_q_Ω_m = b_q_Ω[dir_idx_m]

b_μ_Ω = assemble_direction_rhs(N, μ.Ω, nd)
b_μ_Ω_p = b_μ_Ω[dir_idx_p]
b_μ_Ω_m = b_μ_Ω[dir_idx_m]

b_g_Ω = SphericalHarmonicsMatrices.assemble_boundary_source(N, g.Ω, [0.0, 0.0, 1.0], nd)
b_g_Ω_p = b_g_Ω[dir_idx_p]
b_g_Ω_m = b_g_Ω[dir_idx_m]
# # switch off transport
# ∂A[1] .= 0.0
# dApm[1] .= 0.0
# dAmp[1] .= 0.0


# begin
#     # for plotting
#     z(θ) = cos(θ)
#     x(θ) = sin(θ)
#     plot(z.(0:0.01:2π), x.(0:0.01:2π), map(θ -> q_Ω([x(θ), 0 , z(θ)]), 0:0.01:2π))
# end 

# begin
#     plot(-1:0.01:1, scattering_kernel)
#     using HCubature

#     hquadrature(scattering_kernel, -1, 1)[1]
#     plotly()
#     plot(z.(0:0.01:2π), x.(0:0.01:2π), scattering_kernel.(cos.(0:0.01:2π)))
# end

# g_Ω(Ω) = pdf(VonMisesFisher([0.0, 0.0, 1.0], 10.0), [Ω...])

# G_Ω = get_G_Ω(g_Ω, N, nd)

## energy
E = assemble_bilinear(
    (u, v) -> ∫(⁺(u*v))*dϵ,
    U_ϵ, V_ϵ
)
# E = EvenOddFiniteElements.assemble_E(E_interval, number_of_elements, basis_order)
δϵ⁻(ϵ) = isapprox(ϵ[1], -1.0)
dE = assemble_bilinear(
    (u, v) -> ∫(⁺(s*∂(u)*v) - ⁺(s*u*∂(v)))*dϵ - ∫(δϵ⁻* ⁺(s*u*v))*d∂E_2, # this defines the "initial" condition on the upper end of E
    U_ϵ, V_ϵ
)
dE = round.(dE, digits=8)

C = assemble_bilinear(
    (u, v) -> ∫(2.0 * ⁺((ϵ -> (γ(ϵ) + ∂s_2(ϵ)))* u * v))*dϵ,
    U_ϵ, V_ϵ
)
C = round.(C, digits=8)

S = assemble_bilinear(
    (u, v) -> ∫(2.0 * ⁺(σ * u * v))*dϵ,
    U_ϵ, V_ϵ
)
S = round.(S, digits=8)
b_q_ϵ = assemble_linear(
    v -> ∫(2.0 * ⁺(q.ϵ * v))*dϵ,
    U_ϵ, V_ϵ
)
b_μ_ϵ = assemble_linear(
    v -> ∫(2.0 * ⁺(μ.ϵ * v))*dϵ,
    U_ϵ, V_ϵ
)
b_g_ϵ = assemble_linear(
    v -> ∫(2.0 * ⁺(g.ϵ * v))*dϵ,
    U_ϵ, V_ϵ
)

### solve

# b_ϵ = EvenOddFiniteElements.assemble_source(q_ϵ, E_interval, number_of_elements, basis_order)

import Gridap.TensorValues.⊗
⊗(A, B) = kron(A, B)

n_p = n_space_basis_p*num_free_dofs(U_ϵ)*n_dir_basis_p
n_m = n_space_basis_m*num_free_dofs(U_ϵ)*n_dir_basis_m

A = vcat(
    hcat(Xpp⊗(dE⊗AIpp + C⊗AIpp - S⊗Kpp) + ∂Xpp⊗E⊗∂App, -dXmp⊗E⊗dAmp),
    hcat(dXpm⊗E⊗dApm, Xmm⊗(dE⊗AImm + C⊗AImm - S⊗Kmm))
)


# decom = svd(Matrix(A))
# plot(decom.S)

# A = X⊗dE⊗AI + dXpm[1] ⊗ E ⊗ dApm[1] - dXmp[1] ⊗ E ⊗ dAmp[1] + ∂X[1]⊗E⊗∂A[1] + X⊗(C⊗AI - S⊗K) 
# A = X⊗dE⊗AI + X⊗(C⊗AI) 

# b = b_q_x⊗b_q_ϵ⊗b_q_Ω - 2*b_g_x⊗b_g_ϵ⊗b_g_Ω[1]
b = - 2.0 * vcat(
    b_g_x_p⊗b_g_ϵ⊗b_g_Ω_p, 
    b_g_x_m⊗b_g_ϵ⊗b_g_Ω_m)

c = vcat(
    b_μ_x_p⊗b_μ_ϵ⊗b_μ_Ω_p,
    b_μ_x_m⊗b_μ_ϵ⊗b_μ_Ω_m,
)

# 

# u = A \ b
ps = MKLPardisoSolver()
@time u = Pardiso.solve(ps, A, b)
#v = Pardiso.solve(ps, sparse(transpose(A)), c)

#dot(u, c)
#dot(v, b)

#c = A \ (-b)

GC.gc()

x_coords = range(-1, 1, length=100)
y = zeros(length(x_coords))
y_v = zeros(length(x_coords))

e_x = [eval_space(U_x, x_) for x_ in x_coords]

gr()
e_Ω_p = zeros(n_dir_basis_p)
e_Ω_m = zeros(n_dir_basis_m)
e_Ω_p[1] = 1.0

# function extraction(ϵ)
#     b_ϵ_temp = eval_energy(U_ϵ, ϵ)
#     b_temp = -2.0 * vcat(
#         b_g_x_p ⊗ b_ϵ_temp ⊗ b_g_Ω_p,
#         b_g_x_m ⊗ b_ϵ_temp ⊗ b_g_Ω_m,
#     )
#     return dot(v, b_temp)
# end

# plot(-1:0.01:1, extraction, label=nothing)
# xlabel!("energy ϵ")
# ylabel!("intensity")

@gif for (i, ϵ) = enumerate(range(1, -1, length=100))
    e_ϵ = eval_energy(U_ϵ, ϵ)
    for (i, x_) in enumerate(x_coords)
        e_x_p, e_x_m = e_x[i]
        full_basis = vcat(e_x_p⊗e_ϵ⊗e_Ω_p, e_x_m⊗e_ϵ⊗e_Ω_m)
        y[i] = dot(u, full_basis)
        #y_v[i] = dot(v, full_basis)
    end

    plot(x_coords, y, xflip=true, label="epma-fem")
    # plot!(x_coords, y_v, label="adjoint")
    # plot!(x_coords, x -> μ.x(Point(x))*μ.ϵ(Point(ϵ)), label="extraction")
    # scatter!([0.0], [u_diffeq(ϵ)])
    title!("energy: $(round(ϵ, digits=2))")
    xlabel!("z")
    # scatter!([x[1] for x in R.grid.node_coords], zeros(length(R.grid.node_coords)))
    plot!(x_coords, zeros(length(x_coords)), ls=:dot, color=:black, label=nothing)
    plot!(range(-1, 1, 100), save[i*3][2][:, 1, 1], label="epma-starmap")
    ylims!(-0.05, 0.5)
end fps=3

# using OrdinaryDiffEq
# u_diffeq = solve_DiffEq(s, ϵ -> -q_ϵ(Point(ϵ)), γ, 0.0, (-1, 1))
# plot(-1:0.01:1, ϵ -> eval_average(U_x, U_ϵ, N, u, 0.0, ϵ))
# plot!(u_diffeq)