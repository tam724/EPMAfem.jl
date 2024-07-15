using Gridap
using SparseArrays
using HCubature
using LinearAlgebra
using Enzyme
using Distributions
using Plots
using IterativeSolvers
using Zygote
using Lux
using Optim, Lux, Random, Optimisers
using GridapGmsh


include("spherical_harmonics.jl")
using .SphericalHarmonicsMatrices
include("epma-fem.jl")


#some fixed physics:
scattering_kernel_(μ) = 0.0 #exp(-4.0*(μ-(1))^2)
scattering_norm_factor = 2*π*hquadrature(x -> scattering_kernel_(x), -1.0, 1.0, rtol=1e-8, atol=1e-8, maxevals=100000)[1]
scattering_kernel(μ) = 0.0 # scattering_kernel_(μ) / scattering_norm_factor

ss(ϵ) = [1.0, 0.0]
∂ss(ϵ) = Enzyme.autodiff(Forward, ss, Duplicated(ϵ, 1.0))[1]
σ(ϵ) = [0.0, 0.0]
ττ(ϵ) = σ(ϵ)

s(ϵ) = 0.5 .* ss(ϵ)
τ(ϵ) = ττ(ϵ) .- 0.5 .* ∂ss(ϵ)

physics = (s=s, τ=τ, σ=σ)

model = CartesianDiscreteModel((-1.5, 1.5, -1.5, 1.5, -1.5, 1.5), (100, 100, 100))
model = DiscreteModelFromFile("square.msh")

M = material_space(model)

function ρ(x)
    [1.0, 0.0]
end

mass_concentrations = [interpolate(x -> ρ(x)[1], M), interpolate(x -> ρ(x)[2], M)]
# build solver:

solver = build_solver(model, 21)
nd(solver)
n_basis = number_of_basis_functions(solver)
X = assemble_space_matrices(solver, mass_concentrations)
Ω = assemble_direction_matrices(solver, scattering_kernel)

function sparsity(S)
    return length(S.nzval) / (size(S, 1)*size(S, 2))
end

sparsity.(X.Xpp)
sparsity.(X.Xmm)
sparsity.(X.dXmp)
sparsity.(X.dXpm)
sparsity.(X.∂Xpp)

sparsity.(Ω.dAmp)
sparsity.(Ω.dApm)
sparsity.(Ω.∂App)

∂App_dense = Matrix.(Ω.∂App)
using CUDA


test = rand(n_basis.x.p, n_basis.Ω.p)
temp = zeros(n_basis.x.p, n_basis.Ω.p)

test2 = rand(n_basis.Ω.p, n_basis.x.p)
temp2 = zeros(n_basis.Ω.p, n_basis.x.p)
using BenchmarkTools

using MKLSparse
@benchmark mul!(temp, X.Xpp[1], test)
@benchmark mul!(temp2, test2,  X.Xpp[1])

@benchmark mul!(temp, test, ∂App_dense[2])
@benchmark mul!(temp2, ∂App_dense[2], test2)

# reduced number of experiments
g = (ϵ = [(ϵ -> 0.0)],
    x = [(x -> 0.0)], 
    Ω = [(Ω -> 0.0)])

gh = semidiscretize_boundary(solver, g)

ghϵ(ϵ) = 2*gh.ϵ[1](ϵ)*vcat(gh.x[1].p⊗gh.Ω[1].p, gh.x[1].m⊗gh.Ω[1].m)

function initial_condition(solver)
    n_basis = number_of_basis_functions(solver)
    δ = DiracDelta(model, Point(0, 0))
    testfunc(v, args) = δ(v)
    u0xp = assemble_linear(testfunc, (nothing, ), solver.U[1], solver.V[1])
    u0Ωp = zeros(n_basis.Ω.p)
    u0Ωp[1] = 1.0
    u0xm = zeros(n_basis.x.m)
    u0Ωm = zeros(n_basis.Ω.m)
    return vcat(u0xp⊗u0Ωp, u0xm⊗u0Ωm)
end

function solve_forward_initial(solver, X, Ω, physics, (ϵ0, ϵ1), N, gh)
    n_basis = number_of_basis_functions(solver)
    ϵs = range(ϵ1, ϵ0, length=N)
    ψs = [initial_condition(solver)]
    @show length(ψs[1])
    for k in 2:5#length(ϵs)
        @show k
        A, b = Ab_midpoint(ϵs[k-1], ϵs[k], ψs[k-1], physics, X, Ω, gh)
        ψk = copy(ψs[k-1])
        ψk, log = IterativeSolvers.bicgstabl!(ψk, A, b, 2, log=true)
        @show log
        push!(ψs, ψk)
    end

    return reverse(ϵs), reverse(ψs)
end

using BenchmarkTools
ψ0 = initial_condition(solver)
ψ0p = ψ0[1:n_basis.x.p*n_basis.Ω.p]

temp = reshape(@view(ψ0[1:n_basis.x.p*n_basis.Ω.p]), (n_basis.x.p, n_basis.Ω.p))

temp * Ω.dAmp[1]

function test1(dX, dA, test, n)
    temp = dX ⊗ₓ dA
    @show size(temp)
    res = temp * test
    for _ in 1:n
        res .= temp * res
    end
    return temp * test
end

temp = rand(size(Ω.dAmp[2], 2), size(X.dXmp[1], 2))
@benchmark res1 = test1(X.dXmp, Ω.dAmp, vec(temp), 10)
@benchmark res2 = test2(X.dXmp, Ω.dAmp, temp)

maximum(abs.(res1 .- res2))

function test2(dX, dA, test)
    temp = dA[1] * test * transpose(dX[1]) .+ dA[2] * test * transpose(dX[2])
    return vec(temp)
end

test2(d)

@time A, b = Ab_midpoint(1.0, 0.9, ψ0, physics, X, Ω, ghϵ)
A, b = Abx_midpoint(0.9, 1.0, ψ0, physics, X, Ω, ghϵ)

@profview ψs = solve_forward_initial(solver, X, Ω, physics, (0.0, 1.0), 321, ghϵ)


X.dXmp[1]
Ω.dAmp[1]

z_coords = range(-1.5, 1.5, length=321)
x_coords = range(-1.5, 1.5, length=321)

e_x = eval_space(solver.U, Point.(z_coords, x_coords'))
sol = zeros(length(z_coords), length(x_coords))
n_basis = number_of_basis_functions(solver)
e_Ω_p = spzeros(n_basis.Ω.p)
e_Ω_m = spzeros(n_basis.Ω.m)
#e_Ω_p = gh.Ω[1].p
e_Ω_p[1] = 1.0

ψs[2][1][1:n_basis.x.p*n_basis.Ω.p][1:n_basis.Ω.p:end]

@gif for k in 100:-1:1
    @show k
    # for (i, x_) in enumerate(z_coords)
    #     for (j, y_) in enumerate(x_coords)
    #         e_x_p, e_x_m = e_x[i, j]
    #         full_basis = vcat(e_x_p⊗e_Ω_p, e_x_m⊗e_Ω_m)
    #         sol[i, j] = dot(ψs[2][k], full_basis)
    #     end
    # end
    # heatmap(x_coords, z_coords, reverse(sol, dims=2), linewidth=0.0, levels=10, aspect_ratio=:equal)

    temp_fe_func = FEFunction(solver.U[1], ψs[2][k][1:n_basis.x.p*n_basis.Ω.p][1:n_basis.Ω.p:end])
    points = Point.(z_coords, x_coords')
    sol = reshape(evaluate(temp_fe_func, points[:]), size(points))
    
    #heatmap(reshape(ψs[2][k][1:n_basis.x.p*n_basis.Ω.p][1:n_basis.Ω.p:end], (322, 322)))

    heatmap(sol)
    #contourf(x_coords, z_coords, reverse(sol, dims=2), linewidth=0.0, levels=10, aspect_ratio=:equal)
    #plot(x_coords, -sol[1, :])
    # title!("energy: $(round(ϵs[k], digits=2))")
end fps=3


sol_cartesian = deepcopy(sol)

temp_fe_func = FEFunction(solver.U[1], ψs[2][1][1:n_basis.x.p*n_basis.Ω.p][1:n_basis.Ω.p:end])
points = Point.(z_coords, x_coords')
sol = reshape(evaluate(temp_fe_func, points[:]), size(points))

gr()
heatmap(z_coords, x_coords, sol, aspect_ratio=:equal, cmap=:jet)


for (i, x_) in enumerate(z_coords)
    for (j, y_) in enumerate(x_coords)
        e_x_p, e_x_m = e_x[i, j]
        full_basis = vcat(e_x_p⊗e_Ω_p, e_x_m⊗e_Ω_m)
        sol[i, j] = dot(ψs[2][1], full_basis)
    end
end
heatmap(x_coords, z_coords, reverse(sol, dims=2), aspect_ratio=:equal)

mean_fluence = FEFunction(solver.U[1], ψs[2][1][1:n_basis.x.p*n_basis.Ω.p][1:n_basis.Ω.p:end])

heatmap(reshape(ψs[2][1][1:n_basis.x.p*n_basis.Ω.p][1:n_basis.Ω.p:end], (322, 322)), cmap=:jet, aspect_ratio=:equal)

using GridapMakie, GLMakie
Makie.inline!(true)

GLMakie.plot(solver.model.R, mean_fluence)

solver.model.R.grid.node_coords

mean_fluence