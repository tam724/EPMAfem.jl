using Gridap
using SparseArrays
using StaticArrays
using LinearAlgebra
using LinearSolve
using Plots
using KrylovKit
using Distributions
using Pardiso
using AlgebraicMultigrid
using Enzyme
using HCubature

# using StaRMAP
# using PNSolver

include("spherical_harmonics.jl")
using .SphericalHarmonicsMatrices

include("even_odd_fem.jl")
using .EvenOddFEM

# ## for comparison
# struct TestMaterial{E} <: PNMaterial{1, Float64}
#     elms::E
# end

# import PNSolver.component_densities!
# function component_densities!(ρ::AbstractVector, m::TestMaterial, x::AbstractVector)
#     ρ .= 1.0
# end

# struct DummySP <: PNSolver.NeXLCore.BetheEnergyLoss end
# import PNSolver:stopping_power_func
# function stopping_power_func(::Type{DummySP}, element::PNSolver.NeXLCore.PeriodicTable.Element)
#     return ϵ -> -s([ϵ]) + 0.0*ϵ
# end

# struct DummyTC <: PNSolver.NeXLCore.ElasticScatteringCrossSection end
# import PNSolver:transport_coefficient_func
# function transport_coefficient_func(TCA::Type{<:DummyTC}, element, Nmax)
#     # A_e = convert_strip_mass(element.atomic_mass)
#     # ρ = convert_strip_density(element.density)
#     tc = zeros(Nmax+1)
#     for l = 0:Nmax
#         f(x) = scattering_kernel(x) * SphericalHarmonicsMatrices.Pl(x, l)
#         # tc[l+1] = 2. * π * hquadrature(f, -1., 1.)[1]
#         tc[l+1] = 2*π*hquadrature(f, -1., 1., maxevals=1000)[1] # the 2π probably depends on the definition of the differential scattering cross section (sometimes it might be included, sometimes it might be not..)
#     end
#     function tcoeff(ϵ)
#         # integrate the differential cross section into legendre polynomials
#         # ϵ_eV = ϵ # no need to convert as our default unit is eV.
#         # tc = zeros(Nmax+1)
#         # for l = 0:Nmax
#         #     f(x) = δσδΩ(TCA, acos(x), element, ϵ_eV) * Pl(x, l)
#         #     # tc[l+1] = 2. * π * hquadrature(f, -1., 1.)[1]
#         #     tc[l+1] = hquadrature(f, -1., 1.)[1] # the 2π probably depends on the definition of the differential scattering cross section (sometimes it might be included, sometimes it might be not..)
#         # end
#         return σ(ϵ)*tc
#     end
#     return tcoeff
# end

# # temp = transport_coefficient_func(DummyTC, nothing, 5)

# # function δσδΩ(::Type{DummyTC} , θ, element, ϵ)
# #     return 0.5*scattering_kernel(cos(θ))*PNSolver.convert_strip_mass(PNSolver.n"Cu".atomic_mass)
# # end

# gspec = PNSolver.make_gridspec((100, 1, 1), (-4.0, 0.0), 0.0, 0.0)
# beam = PNSolver.PNBeam{Float64}(MultivariateNormal([0.0, 0.0], [1.0, 1.0]), VonMisesFisher([1.0, 0.0, 0.0], 10.0), Normal(0.7, 0.09))
# problem = PNSolver.ForwardPNProblem{11, Float64}(gspec, [PNSolver.n"Cu"], beam, 1.0, -1.0, 300, PNSolver.PhysicalAlgorithm{DummySP, DummyTC})

# m = TestMaterial{typeof([PNSolver.n"Cu"])}([PNSolver.n"Cu"])
# save = PNSolver.compute_and_save(problem, m)

# @gif for i in 1:300
#     plot(save[i][2][:, 1, 1])
#     title!("$(i)")
# end


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

    return (Apm1, ), (Amp1, )
end

function get_transport_matrices(N, nd::Val{2})
    Apm1 = assemble_transport_matrix(N, Val{3}(), :pm, nd)
    Amp1 = assemble_transport_matrix(N, Val{3}(), :mp, nd)

    Apm2 = assemble_transport_matrix(N, Val{1}(), :pm, nd)
    Amp2 = assemble_transport_matrix(N, Val{1}(), :mp, nd)

    return (Apm1, Apm2), (Amp1, Amp2)
end

# function get_transport_matrices(N, nd::Val{2})
#     Apm1 = assemble_transport_matrix(N, Val{3}(), :pm, nd)
#     Amp1 = assemble_transport_matrix(N, Val{3}(), :mp, nd)

#     Apm2 = assemble_transport_matrix(N, Val{1}(), :pm, nd)
#     Amp2 = assemble_transport_matrix(N, Val{1}(), :mp4, nd)
#     return (Apm1 + Amp1, Amp2 + Apm2)
# end

function get_∂A(N, nd::Val{1})
    ∂A1 = assemble_boundary_matrix(N, Val(3), nd)
    # ∂A2 = assemble_boundary_matrix(N, Val{1}(), nd)
    return (∂A1, )
end

function get_∂A(N, nd::Val{2})
    ∂A1 = assemble_boundary_matrix(N, Val(3), nd)
    ∂A2 = assemble_boundary_matrix(N, Val(1), nd)
    return (∂A1, ∂A2)
end

function get_b_g_Ω(g_Ω, N, nd::Val{1})
    return (SphericalHarmonicsMatrices.assemble_boundary_source(N, g_Ω, Val(3), nd), )
end

function get_b_g_Ω(g_Ω, N, nd::Val{2})
    return (SphericalHarmonicsMatrices.assemble_boundary_source(N, g_Ω, Val(3), nd),
        SphericalHarmonicsMatrices.assemble_boundary_source(N, g_Ω, Val(1), nd))
end

function get_b_g_Ω(g_Ω, N, nd::Val{3})
    return (SphericalHarmonicsMatrices.assemble_boundary_source(N, g_Ω, Val(3), nd),
        SphericalHarmonicsMatrices.assemble_boundary_source(N, g_Ω, Val(1), nd),
        SphericalHarmonicsMatrices.assemble_boundary_source(N, g_Ω, Val(2), nd))
end

## basis evaluations
function eval_space(U_x, x)
    bp, bm = spzeros(num_free_dofs(U_x[1])), spzeros(num_free_dofs(U_x[2]))
    e_i_p = spzeros(num_free_dofs(U_x[1]))
    e_i_m = spzeros(num_free_dofs(U_x[2]))

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
    b = spzeros(num_free_dofs(U_ϵ))
    e_i = spzeros(num_free_dofs(U_ϵ))

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
nd = Val(2)

### space definitions
## space 
model_R = CartesianDiscreteModel((-1.0, 1.0, -1.0, 1.0), (20, 20))
# model_R = CartesianDiscreteModel((-1.0, 1.0), (150))
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
N = 15
n_dir_basis = length(SphericalHarmonicsMatrices.get_moments(N, nd))
n_dir_basis_p = length([m for m in SphericalHarmonicsMatrices.get_moments(N, nd) if SphericalHarmonicsMatrices.is_even(m...)])
n_dir_basis_m = length([m for m in SphericalHarmonicsMatrices.get_moments(N, nd) if SphericalHarmonicsMatrices.is_odd(m...)])
dir_idx_p = 1:n_dir_basis_p
dir_idx_m = n_dir_basis_p+1:n_dir_basis_p+n_dir_basis_m

## energy
model_E_2 = CartesianDiscreteModel((-1.0, 0.0), (70))
order_ϵ = 1
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
     x = (x -> isapprox(x[1], 1.0) ? (pdf(MultivariateNormal([0.0, 0.0], [0.1, 0.1]), [(length(x)>1) ? x[2] : 0.0, (length(x)>2) ? x[3] : 0.0])) : 0.0), 
     Ω = (Ω -> pdf(VonMisesFisher([0.0, 0.0, -1.0], 10.0), [Ω...])))

function μ_x(nd)
    if nd == Val(1)
        return x -> (x[1] < 0.9 && x[1] > 0.8) ? 1.0 : 0.0
    elseif nd == Val(2)
        return x -> (x[2] > -0.2 && x[2] < 0.2) ? 1.0 : 0.0
    end
end

μ = (ϵ = (ϵ -> (ϵ[1]+0.8 > 0) ? sqrt(ϵ[1]+0.8) : 0.0),
     x = μ_x(nd),
     Ω = (Ω -> 1.0))

# plot(-1:0.01:1, x -> g.ϵ(Point(x)))
plot(-1:0.01:1, x -> μ.ϵ(Point(x)))
plot(-1:0.01:1, x -> μ.x(Point(0.0, x)))

ρ(x) = 1.0

scattering_kernel_(μ) = exp(-4.0*(μ-(1))^2)
scattering_norm_factor = 2*π*hquadrature(x -> scattering_kernel_(x), -1.0, 1.0, rtol=1e-8, atol=1e-8, maxevals=100000)[1]
scattering_kernel(μ) = scattering_kernel_(μ) / scattering_norm_factor

s(ϵ) = -1.0
∂s_2(ϵ) = 0.5 * Enzyme.autodiff(Forward, s, Duplicated(ϵ[1], 1.0))[1]
τ(ϵ) = 0.5
σ(ϵ) = τ(ϵ)

si(ϵ) = 0.5*(-s(ϵ)) # the minus is due to the definition in s(ϵ)
τi(ϵ) = τ(ϵ) - (-∂s_2(ϵ)) #also the minus here
σi(ϵ) = σ(ϵ)


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

∂x(u, ::Val{1}) = dot(VectorValue(1.0), ∇(u))
∂x(u, ::Val{2}) = dot(VectorValue(1.0, 0.0), ∇(u))
∂y(u, ::Val{2}) = dot(VectorValue(0.0, 1.0), ∇(u))

dXpm1 = assemble_bilinear(
    ((up, um), (vp, vm)) -> ∫( ∂x(up, nd)*vm) * dx,
    U_x, V_x)[x_idx_m, x_idx_p]

dXpm = (dXpm1, )

if nd == Val(2)
    dXpm2 = assemble_bilinear(
        ((up, um), (vp, vm)) -> ∫( ∂y(up, nd)*vm) * dx,
        U_x, V_x)[x_idx_m, x_idx_p]

    dXpm = (dXpm1, dXpm2)
end

dXmp1 = assemble_bilinear(
    ((up, um), (vp, vm)) -> ∫( um * ∂x(vp, nd)) * dx,
    U_x, V_x)[x_idx_p, x_idx_m]
dXmp = (dXmp1, )

if nd == Val(2)
    dXmp2 = assemble_bilinear(
        ((up, um), (vp, vm)) -> ∫( um * ∂y(vp, nd)) * dx,
        U_x, V_x)[x_idx_p, x_idx_m]

    dXmp = (dXmp1, dXmp2)
end

nx(n, ::Val{1}) = dot(n, VectorValue(1.0))
nx(n, ::Val{2}) = dot(n, VectorValue(1.0, 0.0))
ny(n, ::Val{2}) = dot(n, VectorValue(0.0, 1.0))

∂Xpp1 = assemble_bilinear(
    ((up, um), (vp, vm)) -> ∫(abs(nx(n, nd))*up*vp)*d∂R,
    U_x, V_x)[x_idx_p, x_idx_p]

∂Xpp = (∂Xpp1, )

if nd == Val(2)
    ∂Xpp2 = assemble_bilinear(
        ((up, um), (vp, vm)) -> ∫(abs(ny(n, nd))*up*vp)*d∂R,
        U_x, V_x)[x_idx_p, x_idx_p]

    ∂Xpp = (∂Xpp1, ∂Xpp2)
end

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
    ((vp, vm), ) -> ∫(nx(n, nd)*g.x*vp)*d∂R,
    U_x, V_x)
b_g_x_p = b_g_x[x_idx_p]
b_g_x_m = b_g_x[x_idx_m]

## direction
function get_boundary_matrices(N, ::Val{1})
    ∂App1 = assemble_boundary_matrix(N, Val(3), :pp, nd)
    return (round.(∂App1, digits=8), )
end

function get_boundary_matrices(N, ::Val{2})
    ∂App1 = assemble_boundary_matrix(N, Val(3), :pp, nd)
    ∂App2 = assemble_boundary_matrix(N, Val(1), :pp, nd)
    return (round.(∂App1, digits=8), round.(∂App2, digits=8))
end

∂App = get_boundary_matrices(N, nd)
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

# mat = b_g_x * b_g_Ω'
# plot(svd(mat).S)
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
    (u, v) -> ∫(2.0 * ⁺((ϵ -> (τ(ϵ) + ∂s_2(ϵ)))* u * v))*dϵ,
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
⊗ₓ((A, ), (B, )) = kron(A, B)
⊗ₓ((A1, A2), (B1, B2)) = kron(A1, B1) + kron(A2, B2)

n_p = n_space_basis_p*num_free_dofs(U_ϵ)*n_dir_basis_p
n_m = n_space_basis_m*num_free_dofs(U_ϵ)*n_dir_basis_m

n_pp = n_space_basis_p*n_dir_basis_p
n_mm = n_space_basis_m*n_dir_basis_m


# ## now everything matrix valued.. (explicit time stepping)

# Ψs = [(zeros(n_space_basis_p, n_dir_basis_p), zeros(n_space_basis_m, n_dir_basis_m))]
# F((Ψp, Ψm), ϵ) = (
#     -(∂Xpp[1]*Ψp*transpose(∂App[1]) + ∂Xpp[2]*Ψp*transpose(∂App[2]) - dXmp[1]*Ψm*transpose(dAmp[1]) - dXmp[2]*Ψm*transpose(dAmp[2]) + γ(ϵ).*Xpp*Ψp*transpose(AIpp) - σ(ϵ)*Xpp*Ψp*transpose(Kpp)) - 2.0 .* g.ϵ(ϵ) .*b_g_x_p * transpose(b_g_Ω_p),
#     -(dXpm[1]*Ψp*transpose(dApm[1]) + dXpm[2]*Ψp*transpose(dApm[2]) + γ(ϵ).*Xmm*Ψm*transpose(AImm) - σ(ϵ)*Xmm*Ψm*transpose(Kmm))  - 2.0 .* g.ϵ(ϵ) .* b_g_x_m * transpose(b_g_Ω_m)
# )
# ϵs = range(1.0, -1.0, length=500)
# ps = MKLPardisoSolver()
# for i in 1:length(ϵs)-1
#     @show i
#     ϵ = ϵs[i]
#     ϵ_ = ϵs[i+1]
#     Δϵ = ϵs[i+1] - ϵs[i]
#     dΨ = F(Ψs[i], ϵ)
#     Ψp_ = (s(ϵ) .* Xpp) * Ψs[i][1] .+ Δϵ .* dΨ[1]
#     Ψm_ = (s(ϵ) .* Xmm) * Ψs[i][2] .+ Δϵ .* dΨ[2]
#     Ψp = Pardiso.solve(ps, s(ϵ_) .* Xpp, Ψp_)
#     Ψm = Pardiso.solve(ps, s(ϵ_) .* Xmm, Ψm_)
#     #A_ = A(s(ϵ_)/0.5, γ(ϵ_), σ(ϵ_), Δϵ)
#     #b_ = b(s(ϵ)/0.5, g.ϵ(ϵ_), Δϵ, ψs[i])
#     #ψi1 = Pardiso.solve(ps, A_, b_)
#     push!(Ψs, (Ψp, Ψm))
# end


# x_coords = range(-1, 1, length=50)
# y_coords = range(-1, 1, length=50)
# sol = zeros(length(x_coords), length(y_coords))
# # y_alt = zeros(length(x_coords))
# # y_v = zeros(length(x_coords), length(y_coords))

# e_x = [eval_space(U_x, (x_, y_)) for x_ in x_coords, y_ in y_coords]

# gr()
# e_Ω_p = spzeros(n_dir_basis_p)
# e_Ω_m = spzeros(n_dir_basis_m)
# e_Ω_p[1] = 1.0


# @gif for k in 1:1000
#     @show k
#     # e_ϵ = eval_energy(U_ϵ, ϵ)
#     for (i, x_) in enumerate(x_coords)
#         for (j, y_) in enumerate(y_coords)
#             e_x_p, e_x_m = e_x[i, j]
#             #full_basis = vcat(e_x_p⊗e_Ω_p, e_x_m⊗e_Ω_m)
#             # full_basis_alt = e_ϵ ⊗ vcat(e_x_p ⊗ e_Ω_p, e_x_m ⊗ e_Ω_m)
#             sol[i, j] = transpose(e_x_p) * Ψs[k][1] * e_Ω_p
#             # y_alt[i] = dot(u_alt, full_basis_alt)
#             #y_v[i] = dot(v, full_basis)
#         end
#     end

#     heatmap(sol)
#     # plot!(x_coords, y_alt, xflip=true, label="epma-fem_alt")
#     # plot!(x_coords, y_v, label="adjoint")
#     # plot!(x_coords, x -> μ.x(Point(x))*μ.ϵ(Point(ϵ)), label="extraction")
#     # scatter!([0.0], [u_diffeq(ϵ)])
#     title!("energy: $(round(ϵs[k], digits=2))")
#     xlabel!("z")
#     # scatter!([x[1] for x in R.grid.node_coords], zeros(length(R.grid.node_coords)))
#     # plot!(x_coords, zeros(length(x_coords)), ls=:dot, color=:black, label=nothing)
#     # plot!(range(-1, 1, 100), save[i*3][2][:, 1, 1], label="epma-starmap")
#     # ylims!(-0.05, 0.5)
# end fps=90




## implicit energy stepping method

# function vec(A)
#     return @view(A[:])
# end

# function vec((Ap, Am)::Tuple)
#     return vcat(
#         vec(Ap),
#         vec(Am)
#     )
# end


function A_midpoint(ϵ0, ϵ)
    Δϵ = ϵ - ϵ0
    @assert Δϵ < 0.0 # should be solve backwards!
    Δϵ = abs(Δϵ)
    s = si(ϵ)
    s_2 = si((ϵ0 + ϵ) / 2.0)
    τ_2 = τi((ϵ0 + ϵ) / 2.0)
    σ_2 = σi((ϵ0 + ϵ) / 2.0)
    return vcat(
        hcat((s + s_2) .* Xpp⊗AIpp + (Δϵ / 2) .* (∂Xpp⊗ₓ ∂App .+ τ_2 .* Xpp⊗AIpp .- σ_2.*Xpp⊗Kpp), (Δϵ / 2) .* dXmp ⊗ₓ dAmp),
        hcat(-(Δϵ / 2) .* dXpm ⊗ₓ dApm, (s + s_2) .* Xmm ⊗ AImm + (Δϵ / 2) .* (τ_2 .* Xmm⊗AImm .- σ_2.*Xmm⊗Kmm))
    )
end

function Ax_midpoint(ϵ0, ϵ)
    Δϵ = ϵ - ϵ0
    s = si(ϵ)
    s_2 = si((ϵ0 + ϵ) / 2.0)
    τ_2 = τi((ϵ0 + ϵ) / 2.0)
    σ_2 = σi((ϵ0 + ϵ) / 2.0)
    return vcat(
        hcat((s + s_2) .* Xpp⊗AIpp + (Δϵ / 2) .* (∂Xpp⊗ₓ ∂App .+ τ_2 .* Xpp⊗AIpp .- σ_2.*Xpp⊗Kpp), (Δϵ / 2) .* dXmp ⊗ₓ dAmp),
        hcat(-(Δϵ / 2) .* dXpm ⊗ₓ dApm, (s + s_2) .* Xmm ⊗ AImm + (Δϵ / 2) .* (τ_2 .* Xmm⊗AImm .- σ_2.*Xmm⊗Kmm))
    )
end

function bx_midpoint(ϵ0, ϵ, λ0)
    Δϵ = ϵ - ϵ0
    s = si(ϵ)
    s_2 = si((ϵ0 + ϵ) / 2.0)
    τ_2 = τi((ϵ0 + ϵ) / 2.0)
    σ_2 = σi((ϵ0 + ϵ) / 2.0)
    A = vcat(
        hcat((s + s_2) .* Xpp⊗AIpp - (Δϵ / 2) .* (∂Xpp⊗ₓ ∂App .+ τ_2 .* Xpp⊗AIpp .- σ_2.*Xpp⊗Kpp), -(Δϵ / 2) .* dXmp ⊗ₓ dAmp),
        hcat((Δϵ / 2) .* dXpm ⊗ₓ dApm, (s + s_2) .* Xmm ⊗ AImm - (Δϵ / 2) .* (τ_2 .* Xmm⊗AImm .- σ_2.*Xmm⊗Kmm))
    )

    μ_2 = μ.ϵ((ϵ0 + ϵ) / 2.0)
    c = vcat(
        b_μ_x_p⊗b_μ_Ω_p, 
        b_μ_x_m⊗b_μ_Ω_m)

    return A*λ0 - Δϵ*μ_2*c
end

λs = [zeros(n_pp + n_mm)]
ϵs_x = range(-1.0, 1.0, length=100)
ps = MKLPardisoSolver()

for k in 2:length(ϵs_x)
    @show k
    A = Ax_midpoint(ϵs_x[k-1], ϵs_x[k])
    b = bx_midpoint(ϵs_x[k-1], ϵs_x[k], λs[k-1])
    λk = Pardiso.solve(ps, A, b)
    push!(λs, λk)
end


function A(s, τ, σ, Δϵ)
    return vcat(
        hcat((s + Δϵ*τ) .* Xpp ⊗ AIpp + Δϵ .* ∂Xpp[1] ⊗ ∂App[1] + Δϵ .* ∂Xpp[2] ⊗ ∂App[2] - (Δϵ * σ).* Xpp ⊗Kpp , -Δϵ .* dXmp[1] ⊗ dAmp[1] - Δϵ .* dXmp[2] ⊗ dAmp[2]),
        hcat(Δϵ .* dXpm[1]⊗dApm[1] + Δϵ .* dXpm[2]⊗dApm[2], (s + Δϵ*τ) .* Xmm ⊗ AImm - (Δϵ*σ).* Xmm ⊗Kmm)
    )
end

function Ax(s, τ, σ, Δϵ)
    return vcat(
        hcat((s + Δϵ*τ) .* Xpp ⊗ AIpp + Δϵ .* ∂Xpp[1] ⊗ ∂App[1] + Δϵ .* ∂Xpp[2] ⊗ ∂App[2] - (Δϵ * σ).* Xpp ⊗Kpp , Δϵ .* dXmp[1] ⊗ dAmp[1] + Δϵ .* dXmp[2] ⊗ dAmp[2]),
        hcat(-Δϵ .* dXpm[1]⊗dApm[1] - Δϵ .* dXpm[2]⊗dApm[2], (s + Δϵ*τ) .* Xmm ⊗ AImm - (Δϵ*σ).* Xmm ⊗Kmm)
    )
end

function A_U(s, τ, σ, Δϵ, (Up, Um))
    return vcat(
        hcat((s + Δϵ*τ) .* Xpp ⊗ (transpose(Up)*AIpp*Up) + Δϵ .* ∂Xpp[1] ⊗ (transpose(Up)*∂App[1]*Up) + Δϵ .* ∂Xpp[2] ⊗ (transpose(Up)*∂App[2]*Up) - (Δϵ * σ).* Xpp ⊗(transpose(Up)*Kpp*Up) , -Δϵ .* dXmp[1] ⊗ (transpose(Up)*dAmp[1]*Um) - Δϵ .* dXmp[2] ⊗ (transpose(Up)*dAmp[2]*Um)),
        hcat(Δϵ .* dXpm[1]⊗(transpose(Um)*dApm[1]*Up) + Δϵ .* dXpm[2]⊗(transpose(Um)*dApm[2]*Up), (s + Δϵ*τ) .* Xmm ⊗ (transpose(Um)*AImm*Um) - (Δϵ*σ).* Xmm ⊗(transpose(Um)*Kmm*Um))
    )
end

function A_V(s, τ, σ, Δϵ, (Vp, Vm))
    vcat(
        hcat((s + Δϵ*τ) .* (transpose(Vp)*Xpp*Vp) ⊗ AIpp + Δϵ .* (transpose(Vp)*∂Xpp[1]*Vp) ⊗ ∂App[1] + Δϵ .* (transpose(Vp)*∂Xpp[2]*Vp) ⊗ ∂App[2] - (Δϵ * σ).* (transpose(Vp)*Xpp*Vp) ⊗Kpp , -Δϵ .* (transpose(Vp)*dXmp[1]*Vm) ⊗ dAmp[1] - Δϵ .* (transpose(Vp)*dXmp[2]*Vm) ⊗ dAmp[2]),
        hcat(Δϵ .* (transpose(Vm)*dXpm[1]*Vp)⊗dApm[1] + Δϵ .* (transpose(Vm)*dXpm[2]*Vp)⊗dApm[2], (s + Δϵ*τ) .* (transpose(Vm)*Xmm*Vm) ⊗ AImm - (Δϵ*σ).* (transpose(Vm)*Xmm*Vm) ⊗Kmm)
    )
end

function A_UV(s, τ, σ, Δϵ, (Up, Um), (Vp, Vm))
    vcat(
        hcat((s + Δϵ*τ) .* (transpose(Vp)*Xpp*Vp) ⊗ (transpose(Up)*AIpp*Up) + Δϵ .* (transpose(Vp)*∂Xpp[1]*Vp) ⊗ (transpose(Up)*∂App[1]*Up) + Δϵ .* (transpose(Vp)*∂Xpp[2]*Vp) ⊗ (transpose(Up)*∂App[2]*Up) - (Δϵ * σ).* (transpose(Vp)*Xpp*Vp) ⊗(transpose(Up)*Kpp*Up) , -Δϵ .* (transpose(Vp)*dXmp[1]*Vm) ⊗ (transpose(Up)*dAmp[1]*Um) - Δϵ .* (transpose(Vp)*dXmp[2]*Vm) ⊗ (transpose(Up)*dAmp[2]*Um)),
        hcat(Δϵ .* (transpose(Vm)*dXpm[1]*Vp)⊗(transpose(Um)*dApm[1]*Up) + Δϵ .* (transpose(Vm)*dXpm[2]*Vp)⊗(transpose(Um)*dApm[2]*Up), (s + Δϵ*τ) .* (transpose(Vm)*Xmm*Vm) ⊗ (transpose(Um)*AImm*Um) - (Δϵ*σ).* (transpose(Vm)*Xmm*Vm) ⊗(transpose(Um)*Kmm*Um))
    )
end

function b(s, g_ϵ, Δϵ, ψN)
    b_ψΦ = vcat(
    hcat(Xpp ⊗ AIpp, spzeros(n_pp, n_mm)),
    hcat(spzeros(n_mm, n_pp), Xmm ⊗ AImm))

    b_nΩΦ = vcat(
        b_g_x_p⊗b_g_Ω_p, 
        b_g_x_m⊗b_g_Ω_m)
    return s .* (b_ψΦ * ψN) - 2*Δϵ*g_ϵ*b_nΩΦ
end

function c(s, μ_ϵ, Δϵ, ΨN)
    c_ψΦ = vcat(
        hcat(Xpp ⊗ AIpp, spzeros(n_pp, n_mm)),
        hcat(spzeros(n_mm, n_pp), Xmm ⊗ AImm))

    c_μψ = vcat(
        b_μ_x_p⊗b_μ_Ω_p, 
        b_μ_x_m⊗b_μ_Ω_m)
    return s .* (c_ψΦ * ΨN) - Δϵ*μ_ϵ*c_μψ
end

function b_V(s, g_ϵ, Δϵ, ψN, (Vp, Vm))
    b_ψΦ_V = vcat(
        hcat((transpose(Vp)*Xpp) ⊗ AIpp, spzeros(size(Vp, 2)*n_dir_basis_p, n_space_basis_m*n_dir_basis_m)),
        hcat(spzeros(size(Vm, 2)*n_dir_basis_m, n_space_basis_p*n_dir_basis_p), (transpose(Vm)*Xmm) ⊗ AImm))
    b_nΩΦ_V = vcat(
        (transpose(Vp)*b_g_x_p)⊗b_g_Ω_p, 
        (transpose(Vm)*b_g_x_m)⊗b_g_Ω_m)
    s .* (b_ψΦ_V * ψN) - 2*Δϵ*g_ϵ*b_nΩΦ_V
end

function b_U(s, g_ϵ, Δϵ, ψN, (Up, Um))
    b_ψΦ_U = vcat(
        hcat(Xpp ⊗ (transpose(Up)*AIpp), spzeros(n_space_basis_p*size(Up, 2), n_space_basis_m*n_dir_basis_m)),
        hcat(spzeros(n_space_basis_m*size(Um, 2), n_space_basis_p*n_dir_basis_p), Xmm ⊗ (transpose(Um)*AImm)))
    b_nΩΦ_U = vcat(
        b_g_x_p⊗(transpose(Up)*b_g_Ω_p), 
        b_g_x_m⊗(transpose(Um)*b_g_Ω_m))
    return s .* (b_ψΦ_U * ψN) - 2*Δϵ*g_ϵ*b_nΩΦ_U
end

#currently the expensive term..
function b_UV(s, g_ϵ, Δϵ, ψN, (Up, Um), (Vp, Vm))
    b_ψΦ_UV = vcat(
        hcat((transpose(Vp)*Xpp) ⊗ (transpose(Up)*AIpp), spzeros(size(Vp, 2)*size(Up, 2), n_space_basis_m*n_dir_basis_m)),
        hcat(spzeros(size(Vm, 2)*size(Um, 2), n_space_basis_p*n_dir_basis_p), (transpose(Vm)*Xmm) ⊗ (transpose(Um)*AImm)))
    b_nΩΦ_UV = vcat(
        (transpose(Vp)*b_g_x_p)⊗(transpose(Up)*b_g_Ω_p), 
        (transpose(Vm)*b_g_x_m)⊗(transpose(Um)*b_g_Ω_m))
    return s .* (b_ψΦ_UV * ψN) - 2*Δϵ*g_ϵ*b_nΩΦ_UV
end


# solver
using CatViews
Ψ0, (Ψ0p, Ψ0m) = splitview((n_dir_basis_p, n_space_basis_p), (n_dir_basis_m, n_space_basis_m))
Ψ0[:] .= 0.0
Ψs_full = [(Ψ0, (Ψ0p, Ψ0m))]

ϵs = range(1.0, -1.0, length=100)
ps = MKLPardisoSolver()
for i in 1:length(ϵs)-1
    @show i
    ϵ = ϵs[i]
    ϵ_ = ϵs[i+1]
    Δϵ = ϵs[i+1] - ϵs[i]
    A_ = A(s(ϵ_), τ(ϵ_), σ(ϵ_), Δϵ)
    b_ = b(s(ϵ), g.ϵ(ϵ_), Δϵ, Ψs_full[i][1])
    Ψ, (Ψp, Ψm) = splitview((n_dir_basis_p, n_space_basis_p), (n_dir_basis_m, n_space_basis_m))
    Pardiso.solve!(ps, Ψ, A_, b_)
    push!(Ψs_full, (Ψ, (Ψp, Ψm)))
end

## dynamical low rank solver
r = 30
Ψ0p_svd = svd(zeros(n_dir_basis_p, n_space_basis_p))
Ψ0m_svd = svd(rand(n_dir_basis_m, n_space_basis_m))

## continue here !!

# V is space basis
# U is dir basis


Ψs = [(p=(U=Ψ0p_svd.U[:, 1:r], S=diagm(Ψ0p_svd.S[1:r]), V=Ψ0p_svd.V[:, 1:r]), m=(U=Ψ0m_svd.U[:, 1:r], S=diagm(Ψ0m_svd.S[1:r]), V=Ψ0m_svd.V[:, 1:r]))]

ϵs = range(1.0, -1.0, length=100)
ps = MKLPardisoSolver()
@profview for i in 1:10 # length(ϵs)-1
    @show i
    ϵ = ϵs[i]
    ϵ_ = ϵs[i+1]
    Δϵ = ϵs[i+1] - ϵs[i]
    Ψp, Ψm = Ψs[i].p.U * Ψs[i].p.S * transpose(Ψs[i].p.V), Ψs[i].m.U * Ψs[i].m.S * transpose(Ψs[i].m.V)
    # K-step
    A_V_ = A_V(s(ϵ_), τ(ϵ_), σ(ϵ_), Δϵ, (Ψs[i].p.V, Ψs[i].m.V))
    b_V_ = b_V(s(ϵ), g.ϵ(ϵ_), Δϵ, vcat(Ψp[:], Ψm[:]), (Ψs[i].p.V, Ψs[i].m.V))
    K_ = Pardiso.solve(ps, A_V_, b_V_)
    # factorization
    Kp_, Km_ = reshape(K_[1:n_dir_basis_p*r], n_dir_basis_p, r), reshape(K_[n_dir_basis_p*r+1:end], n_dir_basis_m, r)
    K_svd = (p=svd(Kp_), m=svd(Km_))
    U_ = (p=K_svd.p.U, m=K_svd.m.U)
    MM = (p=transpose(U_.p)*Ψs[i].p.U, m=transpose(U_.m)*Ψs[i].m.U)
    # L-step
    A_U_ = A_U(s(ϵ_), τ(ϵ_), σ(ϵ_), Δϵ, (Ψs[i].p.U, Ψs[i].m.U))
    b_U_ = b_U(s(ϵ), g.ϵ(ϵ_), Δϵ, vcat(Ψp[:], Ψm[:]), (Ψs[i].p.U, Ψs[i].m.U))
    L_ = Pardiso.solve(ps, A_U_, b_U_)
    # factorization
    Lp_, Lm_ = reshape(L_[1:r*n_space_basis_p], r, n_space_basis_p), reshape(L_[r*n_space_basis_p+1:end], r, n_space_basis_m)
    L_svd = (p=svd(Lp_), m=svd(Lm_))
    V_ = (p=L_svd.p.V, m=L_svd.m.V)
    NN = (p=transpose(V_.p)*Ψs[i].p.V, m=transpose(V_.m)*Ψs[i].m.V)
    # S-step
    # Ψp_S, Ψm_S = MM.p * Ψs[i].p.S * transpose(NN.p), MM.m * Ψs[i].m.S * transpose(NN.m)
    A_UV_ = A_UV(s(ϵ_), τ(ϵ_), σ(ϵ_), Δϵ, U_, V_)
    b_UV_ = b_UV(s(ϵ), g.ϵ(ϵ_), Δϵ, vcat(Ψp[:], Ψm[:]), U_, V_)
    S_ = Pardiso.solve(ps, sparse(A_UV_), b_UV_)
    Sp_, Sm_ = reshape(S_[1:r*r], r, r), reshape(S_[r*r+1:end], r, r)
    
    push!(Ψs, (p=(U=U_.p, S=Sp_, V=V_.p), m=(U=U_.m, S=Sm_, V=V_.m)))
end

x_coords = range(-1, 1, length=50)
y_coords = range(-1, 1, length=50)
sol_full = zeros(length(x_coords), length(y_coords))
sol = zeros(length(x_coords), length(y_coords))

e_x = [eval_space(U_x, (x_, y_)) for x_ in x_coords, y_ in y_coords]

gr()
e_Ω_p = spzeros(n_dir_basis_p)
e_Ω_m = spzeros(n_dir_basis_m)
e_Ω_p[1] = 1.0

@gif for k in 1:100
    @show k
    # e_ϵ = eval_energy(U_ϵ, ϵ)
    for (i, x_) in enumerate(x_coords)
        for (j, y_) in enumerate(y_coords)
            e_x_p, e_x_m = e_x[i, j]
            full_basis = vcat(e_x_p⊗e_Ω_p, e_x_m⊗e_Ω_m)
            # full_basis_alt = e_ϵ ⊗ vcat(e_x_p ⊗ e_Ω_p, e_x_m ⊗ e_Ω_m)
            #Ψkp = Ψs[k].p.U*Ψs[k].p.S*transpose(Ψs[k].p.V)
            # sol[i, j] = transpose(e_Ω_p) * Ψkp * e_x_p
            sol[i, j] = dot(λs[k], full_basis)
            #sol_full[i, j] = transpose(e_Ω_p) * Ψs_full[k][2][1] * e_x_p
            # y_alt[i] = dot(u_alt, full_basis_alt)
            #y_v[i] = dot(v, full_basis)
        end
    end

    heatmap(sol)
    #p2 = heatmap(sol_full)
    #plot(p1, p2)
    # plot!(x_coords, y_alt, xflip=true, label="epma-fem_alt")
    # plot!(x_coords, y_v, label="adjoint")
    # plot!(x_coords, x -> μ.x(Point(x))*μ.ϵ(Point(ϵ)), label="extraction")
    # scatter!([0.0], [u_diffeq(ϵ)])
    title!("energy: $(round(ϵs_x[k], digits=2))")
    # scatter!([x[1] for x in R.grid.node_coords], zeros(length(R.grid.node_coords)))
    # plot!(x_coords, zeros(length(x_coords)), ls=:dot, color=:black, label=nothing)
    # plot!(range(-1, 1, 100), save[i*3][2][:, 1, 1], label="epma-starmap")
    # ylims!(-0.05, 0.5)
end fps=3

svd(ψs[1])

x_coords = range(-1, 1, length=50)
sol = zeros(length(x_coords))

e_x = [eval_space(U_x, (x_)) for x_ in x_coords]

gr()
e_Ω_p = spzeros(n_dir_basis_p)
e_Ω_m = spzeros(n_dir_basis_m)
e_Ω_p[1] = 1.0

@gif for j in 1:100
    @show j
    for (i, x_) in enumerate(x_coords)
        e_x_p, e_x_m = e_x[i]
        full_basis = vcat(e_x_p⊗e_Ω_p, e_x_m⊗e_Ω_m)
        # full_basis_alt = e_ϵ ⊗ vcat(e_x_p ⊗ e_Ω_p, e_x_m ⊗ e_Ω_m)
        sol[i] = dot(ψs[j], full_basis)
    end

    # plot(x_csol)
    plot(x_coords, sol, xflip=true, label="epma-fem")
    # plot!(x_coords, sol_trunc, xflip=true, label="epma-fem_trunc")
    # plot!(x_coords, y_v, label="adjoint")
    # plot!(x_coords, x -> μ.x(Point(x))*μ.ϵ(Point(ϵ)), label="extraction")
    # scatter!([0.0], [u_diffeq(ϵ)])
    title!("energy: $(round(ϵs[j], digits=2))")
    xlabel!("z")
    # scatter!([x[1] for x in R.grid.node_coords], zeros(length(R.grid.node_coords)))
    # plot!(x_coords, zeros(length(x_coords)), ls=:dot, color=:black, label=nothing)
    # plot!(range(-1, 1, 100), save[i*3][2][:, 1, 1], label="epma-starmap")
    #ylims!(-0.1, 2.0)
end fps=6


# XW = vcat(
#     hcat(∂Xpp[1]⊗∂App[1], -dXmp[1]⊗dAmp[1]),
#     hcat(dXpm[1]⊗dApm[1], spzeros(n_space_basis_m*n_dir_basis_m, n_space_basis_m*n_dir_basis_m))
# )

# XW_svd = svd(Matrix(XW))
# plot(XW_svd.S)

# V_trunc = E ⊗ XW_svd.V[:, 1:300]
# U_trunc = E ⊗ XW_svd.U[:, 1:300]

A = vcat(
    hcat((dE + C)⊗Xpp⊗AIpp - S⊗Xpp⊗Kpp + E⊗∂Xpp[1]⊗∂App[1], -E⊗dXmp[1]⊗dAmp[1]),
    hcat(E⊗dXpm[1]⊗dApm[1], (dE + C)⊗Xmm⊗AImm - S⊗Xmm⊗Kmm)
)

# A*b

# TEMP = A*V_trunc
# TEMP = Matrix(TEMP)
# U_trunc_full = Matrix(transpose(U_trunc))

# A_trunc = U_trunc_full*TEMP
# A_trunc = sparse(round.(A_trunc, digits=5))

b = - 2.0 * vcat(
    b_g_ϵ⊗b_g_x_p⊗b_g_Ω_p, 
    b_g_ϵ⊗b_g_x_m⊗b_g_Ω_m)

c = vcat(
    b_μ_ϵ⊗b_μ_x_p⊗b_μ_Ω_p,
    b_μ_ϵ⊗b_μ_x_m⊗b_μ_Ω_m,
)

# b_trunc = transpose(U_trunc)*b

ps = MKLPardisoSolver()
v = Pardiso.solve(ps, sparse(transpose(A)), c)

AT = sparse(transpose(A))

# u_trunc = Pardiso.solve(ps, A_trunc, b_trunc)
# u_trunc_full = V_trunc*u_trunc
# u

x_coords = range(-1, 1, length=50)
sol = zeros(length(x_coords))
sol_trunc = zeros(length(x_coords))
# y_alt = zeros(length(x_coords))

e_x = [eval_space(U_x, (x_)) for x_ in x_coords]

gr()
e_Ω_p = spzeros(n_dir_basis_p)
e_Ω_m = spzeros(n_dir_basis_m)
e_Ω_p[1] = 1.0

# testtemp = E ⊗ XW_svd.V[:, 1]
function gram_schmidt!(w)
    w_i = zeros(size(w, 1))
    for i = 1:size(w, 2)
        @show i
        w_i .= @view(w[:, i])
        for k = 1:i-1
            @view(w[:, i]) .-= dot(@view(w[:, k]), w_i).*@view(w[:, k])
        end
        @view(w[:, i]) ./= norm(@view(w[:, i]))
    end
    return w
end

N_trunc = 200
w = zeros(size(AT, 1), N_trunc)
w[:, 1] .= c ./ norm(c)

for i in 2:N_trunc
    w[:, i] .= AT * (AT * @view(w[:, i-1]))
    w[:, i] ./= norm(@view(w[:, i]))
end

Q = gram_schmidt!(w)

A_low = transpose(Q)*AT*Q
b_low = transpose(Q)*c

x_low = A_low \ b_low
x = Q*x_low
@gif for (j, ϵ) = enumerate(range(1, -1, length=200))
    @show j
    e_ϵ = eval_energy(U_ϵ, ϵ)
    for (i, x_) in enumerate(x_coords)
        e_x_p, e_x_m = e_x[i]
        full_basis = vcat(e_ϵ⊗e_x_p⊗e_Ω_p, e_ϵ⊗e_x_m⊗e_Ω_m)
        sol_trunc[i] = dot(x, full_basis)
        # full_basis_alt = e_ϵ ⊗ vcat(e_x_p ⊗ e_Ω_p, e_x_m ⊗ e_Ω_m)
        sol[i] = dot(v, full_basis)
    end

    # plot(x_csol)
    plot(x_coords, sol, xflip=true, label="epma-fem")
    plot!(x_coords, sol_trunc, xflip=true, label="epma-fem_trunc")
    # plot!(x_coords, y_v, label="adjoint")
    # plot!(x_coords, x -> μ.x(Point(x))*μ.ϵ(Point(ϵ)), label="extraction")
    # scatter!([0.0], [u_diffeq(ϵ)])
    title!("energy: $(round(ϵ, digits=2))")
    xlabel!("z")
    # scatter!([x[1] for x in R.grid.node_coords], zeros(length(R.grid.node_coords)))
    # plot!(x_coords, zeros(length(x_coords)), ls=:dot, color=:black, label=nothing)
    # plot!(range(-1, 1, 100), save[i*3][2][:, 1, 1], label="epma-starmap")
    #ylims!(-0.1, 2.0)
end fps=6

## old stuff

A = vcat(
    hcat(Xpp⊗(dE⊗AIpp + C⊗AIpp - S⊗Kpp) + ∂Xpp[1]⊗E⊗∂App[1], -dXmp[1]⊗E⊗dAmp[1]),
    hcat(dXpm[1]⊗E⊗dApm[1], Xmm⊗(dE⊗AImm + C⊗AImm - S⊗Kmm))
)

if nd == Val(2)
    A += vcat(
        hcat(∂Xpp[2]⊗E⊗∂App[2], -dXmp[2]⊗E⊗dAmp[2]),
        hcat(dXpm[2]⊗E⊗dApm[2], spzeros(n_m, n_m))
    )
end

b = - 2.0 * vcat(
    b_g_x_p⊗b_g_ϵ⊗b_g_Ω_p, 
    b_g_x_m⊗b_g_ϵ⊗b_g_Ω_m)

# 

# u = A \ b
ps = MKLPardisoSolver()
u = Pardiso.solve(ps, A, b)

# Pardiso.set_matrixtype!(ps, Pardiso.REAL_NONSYM)
# Pardiso.set_msglvl!(ps, 1)
# @time Pardiso.pardiso(ps, u, A, b)

# v = zeros(size(b))
v = Pardiso.solve(ps, sparse(transpose(A)), c)


# LinearSolve.solve(prob, KrylovJL_GMRES())
# @time u_alt = Pardiso.solve(ps, A_alt, b_alt)
#v = Pardiso.solve(ps, sparse(transpose(A)), c)

#dot(u, c)
#dot(v, b)

#c = A \ (-b)

GC.gc()

### plotting for 1D

x_coords = range(-1, 1, length=50)
sol = zeros(length(x_coords))
# y_alt = zeros(length(x_coords))

e_x = [eval_space(U_x, (x_)) for x_ in x_coords]

gr()
e_Ω_p = spzeros(n_dir_basis_p)
e_Ω_m = spzeros(n_dir_basis_m)
e_Ω_p[1] = 1.0

@gif for (j, ϵ) = enumerate(range(1, -1, length=100))
    e_ϵ = eval_energy(U_ϵ, ϵ)
    for (i, x_) in enumerate(x_coords)
        e_x_p, e_x_m = e_x[i]
        full_basis = vcat(e_x_p⊗e_ϵ⊗e_Ω_p, e_x_m⊗e_ϵ⊗e_Ω_m)
        full_basis_alt = e_ϵ ⊗ vcat(e_x_p ⊗ e_Ω_p, e_x_m ⊗ e_Ω_m)
        sol[i] = dot(u, full_basis)
    end

    # plot(x_csol)
    plot(x_coords, sol, xflip=true, label="epma-fem")
    # plot!(x_coords, y_v, label="adjoint")
    # plot!(x_coords, x -> μ.x(Point(x))*μ.ϵ(Point(ϵ)), label="extraction")
    # scatter!([0.0], [u_diffeq(ϵ)])
    title!("energy: $(round(ϵ, digits=2))")
    xlabel!("z")
    # scatter!([x[1] for x in R.grid.node_coords], zeros(length(R.grid.node_coords)))
    # plot!(x_coords, zeros(length(x_coords)), ls=:dot, color=:black, label=nothing)
    # plot!(range(-1, 1, 100), save[i*3][2][:, 1, 1], label="epma-starmap")
    # ylims!(-0.05, 0.5)
end fps=3

### plotting for 2D

x_coords = range(-1, 1, length=50)
y_coords = range(-1, 1, length=50)
sol = zeros(length(x_coords), length(y_coords))
# y_alt = zeros(length(x_coords))
y_v = zeros(length(x_coords), length(y_coords))

e_x = [eval_space(U_x, (x_, y_)) for x_ in x_coords, y_ in y_coords]

gr()
e_Ω_p = spzeros(n_dir_basis_p)
e_Ω_m = spzeros(n_dir_basis_m)
e_Ω_p[1] = 1.0

function extraction(b_ϵ_temp, (b_x_temp_p, b_x_temp_m))
    #b_ϵ_temp = eval_energy(U_ϵ, ϵ)
    #b_x_temp_p, b_x_temp_m = eval_space(U_x, (1.0, y))
    b_temp = -2.0 * vcat(
        b_x_temp_p ⊗ b_ϵ_temp ⊗ b_g_Ω_p,
        b_x_temp_m ⊗ b_ϵ_temp ⊗ b_g_Ω_m,
    )
    return dot(v, b_temp)
end

function extraction_fixed_e(b_ϵ_temp, y)
    b_x_temp_p, b_x_temp_m = eval_space(U_x, (1.0, y))
    b_temp = -2.0 * vcat(
        b_x_temp_p ⊗ b_ϵ_temp ⊗ b_g_Ω_p,
        b_x_temp_m ⊗ b_ϵ_temp ⊗ b_g_Ω_m,
    )
    return dot(v, b_temp)
end

b_ϵ_temps = [eval_energy(U_ϵ, ϵ) for ϵ ∈ -1:0.05:1]
b_x_temps = [eval_space(U_x, (1.0, y)) for y in -1:0.05:1]
# plot(-1:0.05:1, y -> extraction_fixed_e(eval_energy(U_ϵ, -0.5), y))
# plot!(-1:0.05:1, y -> extraction_fixed_e(eval_energy(U_ϵ, -0.4), y))
# plot!(-1:0.05:1, y -> extraction_fixed_e(eval_energy(U_ϵ, -0.3), y))
# plot!(-1:0.05:1, y -> extraction_fixed_e(eval_energy(U_ϵ, -0.2), y))

plotly()

permutedims([(), (), ()])

heatmap(-1:0.05:1, -1:0.05:1, extraction.(b_ϵ_temps, permutedims(b_x_temps)))
vline!([-0.2, 0.2])
xlabel!("y")
ylabel!("energy")

plot()
for (i, b_ϵ_temp) in enumerate(b_ϵ_temps)
    plot!(extraction.(Ref(b_ϵ_temp), b_x_temps), label="energy = $((-1:0.05:1)[i])")
end
plot!()

# plot(-1:0.01:1, extraction, label=nothing)
# xlabel!("energy ϵ")
# ylabel!("intensity")

gr()

@gif for (j, ϵ) = enumerate(range(1, -1, length=100))
    e_ϵ = eval_energy(U_ϵ, ϵ)
    for (i, x_) in enumerate(x_coords)
        for (j, y_) in enumerate(y_coords)
            e_x_p, e_x_m = e_x[i, j]
            full_basis = vcat(e_x_p⊗e_ϵ⊗e_Ω_p, e_x_m⊗e_ϵ⊗e_Ω_m)
            full_basis_alt = e_ϵ ⊗ vcat(e_x_p ⊗ e_Ω_p, e_x_m ⊗ e_Ω_m)
            sol[i, j] = dot(v, full_basis)
            # y_alt[i] = dot(u_alt, full_basis_alt)
            #y_v[i] = dot(v, full_basis)
        end
    end

    heatmap(sol)
    # plot!(x_coords, y_alt, xflip=true, label="epma-fem_alt")
    # plot!(x_coords, y_v, label="adjoint")
    # plot!(x_coords, x -> μ.x(Point(x))*μ.ϵ(Point(ϵ)), label="extraction")
    # scatter!([0.0], [u_diffeq(ϵ)])
    title!("energy: $(round(ϵ, digits=2))")
    xlabel!("z")
    # scatter!([x[1] for x in R.grid.node_coords], zeros(length(R.grid.node_coords)))
    # plot!(x_coords, zeros(length(x_coords)), ls=:dot, color=:black, label=nothing)
    # plot!(range(-1, 1, 100), save[i*3][2][:, 1, 1], label="epma-starmap")
    # ylims!(-0.05, 0.5)
end fps=3