using Revise

using Gridap
using Serialization
using SparseArrays
using HCubature
using LinearAlgebra
using Enzyme
using Distributions
using Plots

using IterativeSolvers
using Krylov
using CUDA
using Zygote
using Lux
using Optim, Lux, Random, Optimisers
using BenchmarkTools
using StaticArrays

include("spherical_harmonics.jl")
using .SphericalHarmonicsMatrices
include("kroneckerblockmatrix.jl")
include("pnsystemmatrix.jl")
include("epma-fem.jl")

#some fixed physics:
scattering_kernel_(μ) = exp(-4.0*(μ-(1))^2)
scattering_norm_factor = 2*π*hquadrature(x -> scattering_kernel_(x), -1.0, 1.0, rtol=1e-8, atol=1e-8, maxevals=100000)[1]
scattering_kernel(μ) = scattering_kernel_(μ) / scattering_norm_factor

ss(ϵ) = [1.0 + 0.1*exp(-ϵ), 1.0 + 0.2*exp(-ϵ)]
∂ss(ϵ) = Enzyme.autodiff(Forward, ss, Duplicated(ϵ, 1.0))[1]
σ(ϵ) = [0.5 + 0.1*exp(-3*ϵ), 0.5 + 0.1*exp(-4*ϵ)]
ττ(ϵ) = σ(ϵ)

s(ϵ) = 0.5 .* ss(ϵ)
τ(ϵ) = ττ(ϵ) .- 0.5 .* ∂ss(ϵ)

physics = (s=s, τ=τ, σ=σ)

n_z = 50
model = CartesianDiscreteModel((0.0, 1.0, -1.0, 1.0), (n_z, 2*n_z))

M = material_space(model)

p_ellipse = (μ1=0.9, μ2=0.0, r=-π/2.5, a=0.35, b=0.15)

is_in_ellipse(x, p) = ((x[1] - p.μ1)*cos(p.r) + (x[2] - p.μ2)*sin(p.r))^2/p.a^2 + ((x[1] - p.μ1)*sin(p.r) - (x[2] - p.μ2)*cos(p.r))^2/p.b^2 < 1.0

function ellipse_points(p, thick=false)
    z = [p.μ1 + p.a*cos(θ)*cos(p.r) - p.b*sin(θ)*sin(p.r) for θ ∈ 0:0.01:2π]
    x = [p.μ2 + p.a*cos(θ)*sin(p.r) + p.b*sin(θ)*cos(p.r) for θ ∈ 0:0.01:2π]
    limit = thick ? 0.99 : 1.0
    for i in 1:length(z)
        if z[i] > limit
            z[i] = NaN
            x[i] = NaN
        end
    end
    return x, z
end

function ρ(x)
    return [1.0, 0.0]

    # if is_in_ellipse(x, p_ellipse)
    #     return [0.8, 0.0]
    # else
    #     return [0.0, 1.2]
    # end
end

# function ρ(x)
#     if (x[2] > -0.15 && x[2] < 0.15) && (x[1] > 0.85)
#         return [0.8, 0.0]
#     else
#         return [0.0, 1.2]
#     end
# end

mass_concentrations = [interpolate(x -> ρ(x)[1], M), interpolate(x -> ρ(x)[2], M)]
true_ρ_vals = [mass_concentrations[1].free_values mass_concentrations[2].free_values]

# build solver:

contourf(-1:0.01:1, 0:0.01:1, (x, z) -> mass_concentrations[2](Point(z, x)))
plot!(ellipse_points(p_ellipse)..., color=:gray, linestyle=:dash, label=nothing)

solver = build_solver(model, 21, 2)
# X = assemble_space_matrices(solver, mass_concentrations)
# Ω = assemble_direction_matrices(solver, scattering_kernel)

g = (ϵ = [(ϵ -> pdf(Normal(ϵpos, 0.04), ϵ[1])) for ϵpos ∈ [0.85, 0.75, 0.65, 0.55, 0.45, 0.35]],
    x = [(x -> isapprox(x[1], 1.0) ? (pdf(MultivariateNormal([xpos, 0.0], [0.05, 0.05]), [(length(x)>1) ? x[2] : 0.0, (length(x)>2) ? x[3] : 0.0])) : 0.0) for xpos ∈ range(-0.5, 0.5, length=40)], 
    Ω = [(Ω -> pdf(VonMisesFisher(normalize(Ωpos), 10.0), [Ω...])) for Ωpos ∈ [[-0.5, 0.0, -1.0], [0.0, 0.0, -1.0], [0.5, 0.0, -1.0]]])

# reduced number of experiments
# g = (ϵ = [(ϵ -> pdf(Normal(ϵpos, 0.04), ϵ[1])) for ϵpos ∈ [0.85]],
#     x = [(x -> isapprox(x[1], 1.0) ? (pdf(MultivariateNormal([xpos, 0.0], [0.05, 0.05]), [(length(x)>1) ? x[2] : 0.0, (length(x)>2) ? x[3] : 0.0])) : 0.0) for xpos ∈ range(-0.5, 0.5, length=5)], 
#     Ω = [(Ω -> pdf(VonMisesFisher(normalize(Ωpos), 10.0), [Ω...])) for Ωpos ∈ [[0.0, 0.0, -1.0]]])

plot()
for gϵ ∈ g.ϵ
    plot!(range(0, 1, 100), gϵ)
end
plot!()

plot()
for gx ∈ g.x
    plot!(range(-1, 1, nhx), x -> gx(Point(1.0, x)))
end
plot!()

# function μ_x(nd)
#     if nd == Val(1)
#         return x -> (x[1] < 0.9 && x[1] > 0.8) ? 1.0 : 0.0
#     elseif nd == Val(2)
#         return x -> (x[2] > -0.2 && x[2] < 0.2) ? 1.0 : 0.0
#     end
# end

μ = (ϵ = [(ϵ -> (ϵ[1]-0.1 > 0) ? sqrt(ϵ[1]-0.1) : 0.0), (ϵ -> (ϵ[1]-0.2 > 0) ? sqrt(ϵ[1]-0.2) : 0.0)],
    x = [(ρ -> ρ), (ρ -> ρ)],
    Ω = [(Ω -> 1.0)])

plot()
for μϵ ∈ μ.ϵ
    plot!(range(0, 1, 100), μϵ)
end
plot!()

gh = semidiscretize_boundary(solver, g)
μh = semidiscretize_source(solver, μ, mass_concentrations)

# plotly()
# mm = measure_forward(solver, X, Ω, physics, (0, 1), 100, gh, μh)
# mmx = measure_adjoint(solver, X, Ω, physics, (0, 1), 100, gh, μh)

# plot(mm[:, 1, 1, 2])
# plot!(mmx[:, 1, 1, 2])

# ϵs, ψs = solve_forward(solver, X, Ω, physics, (0, 1), 100, ϵ -> 2*gh.ϵ[1](ϵ)*vcat(gh.x[1].p⊗gh.Ω[2].p, gh.x[1].m⊗gh.Ω[2].m))

# param = (mass_concentrations=mass_concentrations, solver=solver, X=X, Ω=Ω, physics=physics, ϵ=(0.0, 1.0), N=100, gh=gh, μh=μh, proj=projection_matrix(solver), gram=gram_matrix(solver))

function pn_matrix(solver)
    U = solver.U    
    V = solver.V
    model = solver.model

    n_basis = number_of_basis_functions(solver)
    Kpp, Kmm = assemble_scattering_matrices(solver.PN, scattering_kernel, nd(solver))

    NE = solver.n_elem
    ND = number_of_dimensions(solver.model.model)
    Ni = 1

    # TODO: for speed: we know that ρm is Diagonal

    return PNMatrix(
        ((n_basis.x.p, n_basis.x.m), (n_basis.Ω.p, n_basis.Ω.m)),
        SVector{NE}([assemble_bilinear(∫uv, (model, ), U[1], V[1]) for _ in 1:solver.n_elem]),
        SVector{NE}([Diagonal(Vector(diag(assemble_bilinear(∫uv, (model, ), U[2], V[2])))) for _ in 1:solver.n_elem]),

        SVector{ND}([assemble_bilinear(a, (model, ), U[1], V[1]) for a ∈ ∫absn_uv(model.model)]),
        SVector{ND}([assemble_bilinear(a, (model, ), U[1], V[2]) for a ∈ ∫∂u_v(model.model)]),

        @MVector[1.0, 1.0],

        Diagonal(ones(n_basis.Ω.p)),
        Diagonal(ones(n_basis.Ω.m)),

        @SVector[@MVector[1.0], @MVector[1.0]],
        @SVector[@SVector[Kpp], @SVector[Kpp]],
        @SVector[@SVector[Kmm], @SVector[Kmm]],

        @MVector[1.0],
        SVector{ND}(Matrix.([assemble_boundary_matrix(solver.PN, dir, :pp, nd(model.model)) for dir ∈ space_directions(model.model)])),
        SVector{ND}(Matrix.([assemble_transport_matrix(solver.PN, dir, :pm, nd(model.model)) for dir ∈ space_directions(model.model)])),

        zeros(max(n_basis.x.p, n_basis.x.m)*max(n_basis.Ω.p, n_basis.Ω.m)),
        zeros(max(n_basis.Ω.p, n_basis.Ω.m))
    )
end

function pn_problem(solver)
    n_basis = number_of_basis_functions(solver)
    return PNProblem(pn_matrix(solver), zeros(n_basis.x.p*n_basis.Ω.p + n_basis.x.m*n_basis.Ω.m), zeros(n_basis.x.p*n_basis.Ω.p + n_basis.x.m*n_basis.Ω.m))
end

mat = pn_matrix(solver)

pn_prob = pn_problem(solver)

# size(mat)

# A_schur = PNSchurComplement(mat, zeros(mat.size[1][2]*mat.size[2][2]), zeros(max(mat.size[1][1]*mat.size[2][1], mat.size[1][2]*mat.size[2][2])))

# C = zeros(mat.size[1][1]*mat.size[2][1])
# B = rand(mat.size[1][1]*mat.size[2][1])

# using TimerOutputs
# const tmr = TimerOutput();

# _update_Dinv(A_schur)

function solve_forward(pn_prob, physics, (ϵ0, ϵ1), N, gh)
    ϵs = range(ϵ0, ϵ1, length=N)
    Δϵ = ϵs[2] - ϵs[1]
    ψ = zero(pn_prob.b)
    ψs_rev = [Vector(ψ)]
    mat = pn_prob.A

    solver = Krylov.GmresSolver(mat, pn_prob.b)
    copyto!(pn_prob.btmp, Vector(gh[2]))

    for k in 1:N-1
        i = N - k
        ϵi = ϵs[i]
        ϵip1 = ϵs[i+1]
        ϵ2 = 0.5*(ϵi + ϵip1)
        @show k
        # compute rhs
        # rhs .= -gh(ϵ2)
        _update_b(pn_prob, -gh[1](ϵ2))
        mat.α .= -1.0 / Δϵ .* physics.s(ϵip1) - 1.0 / Δϵ .* physics.s(ϵ2) + 1.0/2.0 .* physics.τ(ϵ2)
        mat.γ[1] .= [-physics.σ(ϵ2)[1]]
        mat.γ[2] .= [-physics.σ(ϵ2)[2]]
        mat.β .= [0.5]
        mul!(pn_prob.b, mat, ψ, -1.0, 1.0)
        # run iterative solver

        mat.α .= 1.0 / Δϵ .* physics.s(ϵi) + 1.0 / Δϵ .* physics.s(ϵ2) + 1.0/2.0 .* physics.τ(ϵ2)
        mat.γ[1] .= [-physics.σ(ϵ2)[1]]
        mat.γ[2] .= [-physics.σ(ϵ2)[2]]
        mat.β .= [0.5]
        #A, b = Ab_midpoint(ϵs[k-1], ϵs[k], ψs[k-1], physics, X, Ω, gh)

        #ψk = copy(last(ψs_rev))
        # ψk, log = IterativeSolvers.bicgstabl!(ψk, mat, rhs, 2, log=true, abstol=1e-3, reltol=1e-3)
        # @show log
        Krylov.gmres!(solver, mat, pn_prob.b, M=I*Δϵ, restart=true)
        copyto!(ψ, solver.x)
        # @show solver.stats
        push!(ψs_rev, Vector(solver.x))
    end

    return ϵs, reverse(ψs_rev)
end

# function solve_forward(solver, mat, physics, (ϵ0, ϵ1), N, gh)
#     n_basis = number_of_basis_functions(solver)
#     ϵs = range(ϵ0, ϵ1, length=N)
#     Δϵ = ϵs[2] - ϵs[1]
#     ψs_rev = [zeros(n_basis.x.p*n_basis.Ω.p + n_basis.x.m*n_basis.Ω.m)]
#     rhs = zeros(n_basis.x.p*n_basis.Ω.p + n_basis.x.m*n_basis.Ω.m)
#     solver = Krylov.GmresSolver(mat, rhs)

#     for k in 1:N-1
#         i = N - k
#         ϵi = ϵs[i]
#         ϵip1 = ϵs[i+1]
#         ϵ2 = 0.5*(ϵi + ϵip1)
#         @show k
#         # compute rhs
#         rhs .= -gh(ϵ2)
#         mat.α .= -1.0 / Δϵ .* physics.s(ϵip1) - 1.0 / Δϵ .* physics.s(ϵ2) + 1.0/2.0 .* physics.τ(ϵ2)
#         mat.γ[1] .= [-physics.σ(ϵ2)[1]]
#         mat.γ[2] .= [-physics.σ(ϵ2)[2]]
#         mat.β .= [0.5]
#         mul!(rhs, mat, last(ψs_rev), -1.0, 1.0)
#         # run iterative solver

#         mat.α .= 1.0 / Δϵ .* physics.s(ϵi) + 1.0 / Δϵ .* physics.s(ϵ2) + 1.0/2.0 .* physics.τ(ϵ2)
#         mat.γ[1] .= [-physics.σ(ϵ2)[1]]
#         mat.γ[2] .= [-physics.σ(ϵ2)[2]]
#         mat.β .= [0.5]
#         #A, b = Ab_midpoint(ϵs[k-1], ϵs[k], ψs[k-1], physics, X, Ω, gh)

#         #ψk = copy(last(ψs_rev))
#         # ψk, log = IterativeSolvers.bicgstabl!(ψk, mat, rhs, 2, log=true, abstol=1e-3, reltol=1e-3)
#         # @show log
#         Krylov.gmres!(solver, mat, rhs, M=I*Δϵ)
#         # @show solver.stats
#         push!(ψs_rev, copy(solver.x))
#     end

#     return ϵs, reverse(ψs_rev)
# end

function solve_forward_schur(schur_pn_prob, physics, (ϵ0, ϵ1), N, gh)
    ϵs = range(ϵ0, ϵ1, length=N)
    Δϵ = ϵs[2] - ϵs[1]
    #ψ_schur = zero(schur_pn_prob.b)
    fill!(schur_pn_prob.full_solution, 0.0)

    ψs_rev = [Vector(schur_pn_prob.full_solution)]
    fill!(ψs_rev[1], 0.0)
    
    # mat_schur = PNSchurComplement(mat, zeros(mat.size[1][2]*mat.size[2][2]), zeros(mat.size[1][2]*mat.size[2][2]))
    # rhs_schur = zeros(n_basis.x.p*n_basis.Ω.p)
    solver_schur = Krylov.GmresSolver(schur_pn_prob.A, schur_pn_prob.b)
    # solver = Krylov.GmresSolver(mat, rhs)
    # full_solution_schur = zeros(n_basis.x.p*n_basis.Ω.p + n_basis.x.m*n_basis.Ω.m)
    copyto!(schur_pn_prob.prob.btmp, Vector(gh[2]))

    mat = schur_pn_prob.prob.A

    #integral = zero(schur_pn_prob.full_solution)

    for k in 1:N-1
        i = N - k
        ϵi = ϵs[i]
        ϵip1 = ϵs[i+1]
        ϵ2 = 0.5*(ϵi + ϵip1)
        @show k
        # compute rhs
        _update_b(schur_pn_prob.prob, -gh[1](ϵ2))

        mat.α .= -1.0 / Δϵ .* physics.s(ϵip1) - 1.0 / Δϵ .* physics.s(ϵ2) + 1.0/2.0 .* physics.τ(ϵ2)
        mat.γ[1] .= [-physics.σ(ϵ2)[1]]
        mat.γ[2] .= [-physics.σ(ϵ2)[2]]
        mat.β .= [0.5]
        mul!(schur_pn_prob.prob.b, mat, schur_pn_prob.full_solution, -1.0, 1.0)
        # run iterative solver

        mat.α .= 1.0 / Δϵ .* physics.s(ϵi) + 1.0 / Δϵ .* physics.s(ϵ2) + 1.0/2.0 .* physics.τ(ϵ2)
        mat.γ[1] .= [-physics.σ(ϵ2)[1]]
        mat.γ[2] .= [-physics.σ(ϵ2)[2]]
        mat.β .= [0.5]
        #A, b = Ab_midpoint(ϵs[k-1], ϵs[k], ψs[k-1], physics, X, Ω, gh)

        #ψk = copy(last(ψs_rev))
        # ψk, log = IterativeSolvers.bicgstabl!(ψk, mat, rhs, 2, log=true, abstol=1e-3, reltol=1e-3)
        # @show log
        _update_D(schur_pn_prob.A)
        _compute_schur_rhs(schur_pn_prob.b, schur_pn_prob.A, schur_pn_prob.prob.b)
        Krylov.gmres!(solver_schur, schur_pn_prob.A, schur_pn_prob.b, M=I*Δϵ)
        _compute_full_solution_schur(schur_pn_prob.full_solution, schur_pn_prob.A, schur_pn_prob.prob.b, solver_schur.x)

        
        # @show solver_schur.stats
        
        #Krylov.gmres!(solver, mat, rhs, M=I*Δϵ)

        #@show solver.x[1:10]
        push!(ψs_rev, Vector(schur_pn_prob.full_solution))
        #integral .+= Δϵ .*schur_pn_prob.full_solution
    end


    #return ϵs, Vector(integral)
    return ϵs, reverse(ψs_rev)
end


# function solve_forward_schur(solver, mat, physics, (ϵ0, ϵ1), N, gh)
#     n_basis = number_of_basis_functions(solver)
#     ϵs = range(ϵ0, ϵ1, length=N)
#     Δϵ = ϵs[2] - ϵs[1]
#     ψs_rev = [zeros(n_basis.x.p*n_basis.Ω.p + n_basis.x.m*n_basis.Ω.m)]
#     rhs = zeros(n_basis.x.p*n_basis.Ω.p + n_basis.x.m*n_basis.Ω.m)
    
#     mat_schur = PNSchurComplement(mat, zeros(mat.size[1][2]*mat.size[2][2]), zeros(mat.size[1][2]*mat.size[2][2]))
#     rhs_schur = zeros(n_basis.x.p*n_basis.Ω.p)
#     solver_schur = Krylov.GmresSolver(mat_schur, rhs_schur)
#     solver = Krylov.GmresSolver(mat, rhs)
#     full_solution_schur = zeros(n_basis.x.p*n_basis.Ω.p + n_basis.x.m*n_basis.Ω.m)

#     for k in 1:N-1
#         i = N - k
#         ϵi = ϵs[i]
#         ϵip1 = ϵs[i+1]
#         ϵ2 = 0.5*(ϵi + ϵip1)
#         @show k
#         # compute rhs
#         rhs .= -gh(ϵ2)
#         mat.α .= -1.0 / Δϵ .* physics.s(ϵip1) - 1.0 / Δϵ .* physics.s(ϵ2) + 1.0/2.0 .* physics.τ(ϵ2)
#         mat.γ[1] .= [-physics.σ(ϵ2)[1]]
#         mat.γ[2] .= [-physics.σ(ϵ2)[2]]
#         mat.β .= [0.5]
#         mul!(rhs, mat, last(ψs_rev), -1.0, 1.0)
#         # run iterative solver

#         mat.α .= 1.0 / Δϵ .* physics.s(ϵi) + 1.0 / Δϵ .* physics.s(ϵ2) + 1.0/2.0 .* physics.τ(ϵ2)
#         mat.γ[1] .= [-physics.σ(ϵ2)[1]]
#         mat.γ[2] .= [-physics.σ(ϵ2)[2]]
#         mat.β .= [0.5]
#         #A, b = Ab_midpoint(ϵs[k-1], ϵs[k], ψs[k-1], physics, X, Ω, gh)

#         #ψk = copy(last(ψs_rev))
#         # ψk, log = IterativeSolvers.bicgstabl!(ψk, mat, rhs, 2, log=true, abstol=1e-3, reltol=1e-3)
#         # @show log
#         _update_D(mat_schur)
#         _compute_schur_rhs(rhs_schur, mat_schur, rhs)
#         Krylov.gmres!(solver_schur, mat_schur, rhs_schur, M=I*Δϵ)
#         _compute_full_solution_schur(full_solution_schur, mat_schur, rhs, solver_schur.x)

        
#         # @show solver_schur.stats
        
#         #Krylov.gmres!(solver, mat, rhs, M=I*Δϵ)

#         #@show solver.x[1:10]
#         push!(ψs_rev, copy(full_solution_schur))
#     end

#     return ϵs, reverse(ψs_rev)
# end

# function solve_forward_cu(solver, mat, physics, (ϵ0, ϵ1), N, gh)
#     n_basis = number_of_basis_functions(solver)
#     ϵs = range(ϵ0, ϵ1, length=N)
#     Δϵ = ϵs[2] - ϵs[1]
#     ψs_rev = [zeros(n_basis.x.p*n_basis.Ω.p + n_basis.x.m*n_basis.Ω.m)]
#     # rhs = zeros(n_basis.x.p*n_basis.Ω.p + n_basis.x.m*n_basis.Ω.m)
#     rhs_cu_temp = cu(Vector(gh[2])) # zeros(n_basis.x.p*n_basis.Ω.p + n_basis.x.m*n_basis.Ω.m)
#     rhs_cu = cu(Vector(gh[2])) # zeros(n_basis.x.p*n_basis.Ω.p + n_basis.x.m*n_basis.Ω.m)
#     mat_cu = cuda(mat)
#     # solver = Krylov.BicgstabSolver(mat_cu, rhs_cu)
#     solver = Krylov.GmresSolver(mat_cu, rhs_cu)
#     #solver = Krylov.MinresSolver(mat_cu, rhs_cu)
#     copyto!(solver.x, ψs_rev[1])

#     integral = cu(zeros(n_basis.x.p*n_basis.Ω.p + n_basis.x.m*n_basis.Ω.m))
#     for k in 1:N-1
#         i = N - k
#         ϵi = ϵs[i]
#         ϵip1 = ϵs[i+1]
#         ϵ2 = 0.5*(ϵi + ϵip1)
#         @show k
#         # compute rhs
#         #rhs .= -gh(ϵ2)
#         # copyto!(rhs_cu, rhs)
#         # rhs_cu = cu(rhs)
#         mul!(rhs_cu, I*(-gh[1](ϵ2)), rhs_cu_temp)
#         mat.α .= -1.0 / Δϵ .* physics.s(ϵip1) - 1.0 / Δϵ .* physics.s(ϵ2) + 1.0/2.0 .* physics.τ(ϵ2)
#         mat_cu.α .= -1.0 / Δϵ .* physics.s(ϵip1) - 1.0 / Δϵ .* physics.s(ϵ2) + 1.0/2.0 .* physics.τ(ϵ2)
#         mat.γ[1] .= [-physics.σ(ϵ2)[1]]
#         mat_cu.γ[1] .= [-physics.σ(ϵ2)[1]]
#         mat.γ[2] .= [-physics.σ(ϵ2)[2]]
#         mat_cu.γ[2] .= [-physics.σ(ϵ2)[2]]
#         mat.β .= [0.5] # [0.5, 0.5, -0.5] # [0.5, 0.5, -0.5]
#         mat_cu.β .= [0.5] # [0.5, 0.5, -0.5] # [0.5, 0.5, -0.5]
#         mul!(rhs_cu, mat_cu, solver.x, -1.0, 1.0)

#         # run iterative solver
#         mat.α .= 1.0 / Δϵ .* physics.s(ϵi) + 1.0 / Δϵ .* physics.s(ϵ2) + 1.0/2.0 .* physics.τ(ϵ2)
#         mat_cu.α .= 1.0 / Δϵ .* physics.s(ϵi) + 1.0 / Δϵ .* physics.s(ϵ2) + 1.0/2.0 .* physics.τ(ϵ2)
#         mat.γ[1] .= [-physics.σ(ϵ2)[1]]
#         mat_cu.γ[1] .= [-physics.σ(ϵ2)[1]]
#         mat.γ[2] .= [-physics.σ(ϵ2)[2]]
#         mat_cu.γ[2] .= [-physics.σ(ϵ2)[2]]
#         mat.β .= [0.5] # [0.5, 0.5, -0.5] # [0.5, 0.5, -0.5]
#         mat_cu.β .= [0.5] # [0.5, 0.5, -0.5] # [0.5, 0.5, -0.5]
#         #A, b = Ab_midpoint(ϵs[k-1], ϵs[k], ψs[k-1], physics, X, Ω, gh)

#         # ψk, log = IterativeSolvers.bicgstabl!(ψk, mat, rhs, 2, log=true, abstol=1e-3, reltol=1e-3)
#         # @show log
#         #copy!(rhs_cu, rhs)
#         # (ψi_cpu, stats1) = Krylov.bicgstab(mat, rhs, M=I*Δϵ)
#         Krylov.gmres!(solver, mat_cu, rhs_cu, M=I*Float32(Δϵ), itmax=1000, restart=true)
#         #Krylov.minres!(solver, mat_cu, rhs_cu, M=I*Float32(Δϵ))
#         @show solver.stats
#         # @show ψi_cpu 
#         # @show ψi_cu
#         # @show stats1
#         # @show stats2
#         # return
#         # @show stats
#         # @show ψi
#         # @show ψi_cu
#         #copy!(ψi, ψi_cu)
#         #return 
#         #@show stats
#         #push!(ψs_rev, Vector(solver.x))
#         integral .+= Δϵ .*solver.x
#     end

#     return Vector(integral) #reverse(ψs_rev)
# end


# mat_cu = cuda(mat)

# function SchurSolver(A, b)
#     solver = (
#         internal_solver = Krylov.GmresSolver(A.size[1][1]*A.size[2][1], A.size[1][1]*A.size[2][1], 20, typeof(b)),
#         D_inv = Diagonal(zeros(A.size[1][2]*A.size[2][2]))
#     )
# end

# mat
# mat_cu = cuda(mat)

# b = zeros(n_basis.x.p*n_basis.Ω.p + n_basis.x.m*n_basis.Ω.m)
# b = 2*gh.ϵ[1](0.85)*vcat(gh.Ω[2].p⊗gh.x[20].p, gh.Ω[2].m⊗gh.x[20].m)
# b_cu = cu(Vector(b))

# x, stats = Krylov.bicgstab(mat, b)
# x_cu, stats = Krylov.bicgstab(mat_cu, b_cu)

# plot(x)
# plot(Vector(x_cu))

using MKLSparse
ϵs, ψs = solve_forward(pn_prob, physics, (0.0, 1.0), 100, (ϵ -> 2*gh.ϵ[1](ϵ), vcat(gh.Ω[2].p⊗gh.x[20].p, gh.Ω[2].m⊗gh.x[20].m)))

schur_pn_prob = schur_complement(pn_prob)
ϵs, ψs_schur = solve_forward_schur(schur_pn_prob, physics, (0.0, 1.0), 100, (ϵ -> 2*gh.ϵ[1](ϵ), vcat(gh.Ω[2].p⊗gh.x[20].p, gh.Ω[2].m⊗gh.x[20].m)))


## double check schur solver, the plots dont look the same.. (on gpu!)

pn_prob_cu = cuda(pn_prob)
ϵs, ψs_cu = solve_forward(pn_prob_cu, physics, (0.0, 1.0), 100, (ϵ -> 2*gh.ϵ[1](ϵ), vcat(gh.Ω[2].p⊗gh.x[20].p, gh.Ω[2].m⊗gh.x[20].m)))


schur_pn_prob_cu = schur_complement(cuda(pn_prob))
ϵs, ψs_schur_cu = solve_forward_schur(schur_pn_prob_cu, physics, (0.0, 1.0), 100, (ϵ -> 2*gh.ϵ[1](ϵ), vcat(gh.Ω[2].p⊗gh.x[20].p, gh.Ω[2].m⊗gh.x[20].m)))



# ϵs, ψs_schur = solve_forward_schur(solver, mat, physics, (0.0, 1.0), 100, ϵ -> 2*gh.ϵ[1](ϵ)*vcat(gh.Ω[2].p⊗gh.x[20].p, gh.Ω[2].m⊗gh.x[20].m))

# ψs = solve_forward_cu(solver, mat, physics, (0.0, 1.0), 100, (ϵ -> 2*gh.ϵ[1](ϵ), vcat(gh.Ω[2].p⊗gh.x[20].p, gh.Ω[2].m⊗gh.x[20].m)))
# # @profview ϵs, ψs = solve_forward(solver, mat, physics, (0.0, 1.0), 100, ϵ -> 2*gh.ϵ[1](ϵ)*vcat(gh.Ω[2].p⊗gh.x[20].p, gh.Ω[2].m⊗gh.x[20].m))

n_basis = number_of_basis_functions(solver)

plotly()
integral = zeros((n_z+1, 2*n_z+1, 2*n_z+1))
for i in 1:100
    integral .+= reshape(ψs_schur[i][1:n_basis.x.p], (n_z+1, 2*n_z+1, 2*n_z+1))
end
temp = reshape(ψs_schur[end][1:n_basis.x.p], (n_z+1, 2*n_z+1, 2*n_z+1))
heatmap()

using GLMakie
temp = reshape(ψs_schur[end-40][1:n_basis.x.p], (n_z+1, 2*n_z+1, 2*n_z+1))
GLMakie.contour(integral, alpha=0.1)

gr()
@gif for (ψ_cu, ψ, ψ_schur_cu, ψ_schur) in zip(reverse(ψs_cu), reverse(ψs), reverse(ψs_schur_cu), reverse(ψs_schur))
    # temp = reshape(@view(ψ[1:n_basis.x.p]), (n_z+1, 2*n_z+1))
    z_coords = range(0.0, 1.0, length=n_z+1)
    x_coords = range(-1.0, 1.0, length=2*n_z+1)
    #vals = reshape(temp.(Point.(z_coords, x_coords')[:]), length(z_coords), length(x_coords))
    p1 = heatmap(x_coords, z_coords, reshape(@view(ψ_cu[1:n_basis.x.p]), (n_z+1, 2*n_z+1)))
    p2 = heatmap(x_coords, z_coords, reshape(@view(ψ_schur_cu[1:n_basis.x.p]), (n_z+1, 2*n_z+1)))
    p3 = heatmap(x_coords, z_coords, reshape(@view(ψ[1:n_basis.x.p]), (n_z+1, 2*n_z+1)))
    p4 = heatmap(x_coords, z_coords, reshape(@view(ψ_schur[1:n_basis.x.p]), (n_z+1, 2*n_z+1)))

    plot(p1, p2, p3, p4, layout=(2, 2))
    # temp = Vector(ψ_cu[1:n_basis.x.p])
    # temp = reshape(temp, (n_z+1, 2*n_z+1))
    # z_coords = range(0.0, 1.0, length=n_z+1)
    # x_coords = range(-1.0, 1.0, length=2*n_z+1)
    # #vals = reshape(temp.(Point.(z_coords, x_coords')[:]), length(z_coords), length(x_coords))
    # p2 = heatmap(x_coords, z_coords, temp)
    # plot(p1, p2)
end fps=10


# @gif for ψ in reverse(ψs)
#     temp = FEFunction(solver.U[1], ψ[1:n_basis.x.p])
#     N = 50
#     z_coords = range(0.0, 1.0, length=N)
#     x_coords = range(-1.0, 1.0, length=2*N)
#     vals = reshape(temp.(Point.(z_coords, x_coords')[:]), length(z_coords), length(x_coords))
#     contourf(x_coords, z_coords, vals)
# end fps=10


size(mat, 1)

n_basis = number_of_basis_functions(solver)
ψ0 = rand(n_basis.x.p*n_basis.Ω.p + n_basis.x.m*n_basis.Ω.m)
ψtemp = zeros(n_basis.x.p*n_basis.Ω.p + n_basis.x.m*n_basis.Ω.m)

ψ0_cu = cu(ψ0)
ψtemp_cu = cu(ψtemp)

mat_cu = cuda(mat)

using MKLSparse
plot(ψtemp)

@profview mul!(ψtemp, mat, ψ0, 1.0, 1.0)
@benchmark mul!($(ψtemp), $(mat), $(ψ0), $(1.0), $(1.0))

mul!(ψtemp_cu, mat_cu, ψ0_cu, 1.0, 1.0)
@benchmark mul!($(ψtemp_cu), $(mat_cu), $(ψ0_cu), 1.0, 1.0)


function pn_transport_matrix(solver)
    U = solver.U    
    V = solver.V
    model = solver.model
    dims = number_of_basis_functions(solver)

    dXpm = Tuple(assemble_bilinear(a, (model, ), U[1], V[2]) for a ∈ ∫∂u_v(model.model))
#   dXmp = (assemble_bilinear(a, (model, ), U[2], V[1]) for a ∈ ∫u_∂v(model.model))
    ∂Xpp = Tuple(assemble_bilinear(a, (model, ), U[1], V[1]) for a ∈ ∫absn_uv(model.model))

    dApm = Matrix.(Tuple(assemble_transport_matrix(solver.PN, dir, :pm, nd(model.model)) for dir ∈ space_directions(model.model)))
#   dAmp = (assemble_transport_matrix(solver.PN, dir, :mp, nd(model.model)) for dir ∈ space_directions(model.model))
    ∂App = Matrix.(Tuple(assemble_boundary_matrix(solver.PN, dir, :pp, nd(model.model)) for dir ∈ space_directions(model.model)))

    # ZXmm = Tuple(spzeros(dims.x.m, dims.x.m) for _ in 1:number_of_dimensions(model.model))
    ZXmm = Tuple(UniformScaling(0.0) for _ in 1:number_of_dimensions(model.model))
    ZAmm = Tuple(UniformScaling(0.0) for _ in 1:number_of_dimensions(model.model))
    
    return KroneckerBlockMat((∂Xpp, transpose.(dXpm), dXpm, ZXmm), (∂App, transpose.(dApm), dApm, ZAmm))
end

X.Xpp[2]
X2 = Diagonal(Vector(diag(X.Xpp[2])))
Xpp2 = X.Xpp[2] .- X2

b = rand(1030301)
y = zeros(1030301)

@benchmark mul!(y, $(X.Xmm[2]), b)
@benchmark mul!(y, X2, b)

function pn_scattering_matrix(solver)
    Xpp = [assemble_bilinear(∫uv, (model, ), U[1], V[1]) for _ in 1:solver.n_elem]
    Zpm = Tuple(UniformScaling(0.0) for _ in 1:solver.n_elem)
    Xmm = [assemble_bilinear(∫uv, (model, ), U[2], V[2]) for _ in 1:solver.n_elem]
    Zmp = Tuple(UniformScaling(0.0) for _ in 1:solver.n_elem)

    Kpp, Kmm = assemble_scattering_matrices(N, scattering_kernel, num_dim)
    AIpp = UniformScaling(1.0)
    AImm = UniformScaling(1.0)

    return KroneckerBlockMat((Xpp, ))

Ω∇ = pn_transport_matrix(solver)

function reorder(ψ0, n_basis_funcs)
    nxp, nxm, nΩp, nΩm = n_basis_funcs.x.p, n_basis_funcs.x.m, n_basis_funcs.Ω.p, n_basis_funcs.Ω.m
    np = nxp*nΩp
    nm = nxm*nΩm
    Xp = reshape(@view(ψ0[1:np]), (nxp, nΩp))
    Xm = reshape(@view(ψ0[np+1:np+nm]), (nxm, nΩm))
    return [transpose(Xp)[:]
        transpose(Xm)[:]]
end

A = cu(rand(10, 10))
C = cu(rand(10, 10))

B = cu(zeros(10, 10))

ZZ = UniformScaling(0.0)

@time mul!(C, A, B);


function reorder2(ψ0, n_basis_funcs)
    nxp, nxm, nΩp, nΩm = n_basis_funcs.x.p, n_basis_funcs.x.m, n_basis_funcs.Ω.p, n_basis_funcs.Ω.m
    np = nxp*nΩp
    nm = nxm*nΩm
    Xp = reshape(@view(ψ0[1:np]), (nΩp, nxp))
    Xm = reshape(@view(ψ0[np+1:np+nm]), (nΩm, nxm))
    return [transpose(Xp)[:]
        transpose(Xm)[:]]
end

n_basis = number_of_basis_functions(solver)
ψ0 = rand(n_basis.x.p*n_basis.Ω.p + n_basis.x.m*n_basis.Ω.m)
ψtemp = zeros(size(ψ0))

@benchmark mul!(ψtemp, Ω∇, ψ0, 1.0, 1.0)

A_old, b_old = Abx_midpoint(0.4, 0.5, ψ0, physics, X, Ω, ϵ -> zeros(length(ψ0)))

@time mul!(ψtemp2, A_old, ψ0_alt)

maximum(abs.(reorder2(ψtemp2, n_basis) .- ψtemp))

Krylov.bicgstab(A, ψ0)

using Base: size
function size(A::ImplMidPointPNSystemMatrix, i)
    nxp, nxm, nΩp, nΩm = A.n_basis_funcs.x.p, A.n_basis_funcs.x.m, A.n_basis_funcs.Ω.p, A.n_basis_funcs.Ω.m
    if i == 1 || i == 2
        return nxp*nΩp + nxm*nΩm
    else
        @assert(false)
    end
end


### OLD STUFF!!

std_ρ_vals1 = zeros(size(true_ρ_vals))
std_ρ_vals2 = zeros(size(true_ρ_vals))
std_ρ_vals1[:, 1] .= 0.8
std_ρ_vals2[:, 2] .= 1.2
std_measurement = measure_std(std_ρ_vals1, param)
std_measurement[:, :, :, 2] = measure_std(std_ρ_vals2, param)[:, :, :, 2]

function measure_std(ρ_vals, params)
    for i ∈ 1:length(params.mass_concentrations)
        params.mass_concentrations[i].free_values .= ρ_vals[:, i]
    end
    update_space_matrices!(params.X, params.solver, params.mass_concentrations)
    update_extractions!(params.μh, params.solver, params.mass_concentrations)
    std_measurements = measure_adjoint(params.solver, params.X, params.Ω, params.physics, params.ϵ, params.N, params.gh, params.μh)
    return std_measurements
end

function measure_raw(ρ_vals, params)
    for i ∈ 1:length(params.mass_concentrations)
        params.mass_concentrations[i].free_values .= ρ_vals[:, i]
    end
    update_space_matrices!(params.X, params.solver, params.mass_concentrations)
    update_extractions!(params.μh, params.solver, params.mass_concentrations)
    measurements = measure_adjoint(params.solver, params.X, params.Ω, params.physics, params.ϵ, params.N, params.gh, params.μh)
    measurements .= measurements./std_measurement
    return measurements
end

Zygote.@adjoint function measure_raw(ρ_vals, params)
    for i ∈ 1:length(params.mass_concentrations)
        params.mass_concentrations[i].free_values .= ρ_vals[:, i]
    end
    update_space_matrices!(params.X, params.solver, params.mass_concentrations)
    update_extractions!(params.μh, params.solver, params.mass_concentrations)
    measurements, ψxss = measure_adjoint(params.solver, params.X, params.Ω, params.physics, params.ϵ, params.N, params.gh, params.μh, true)
    measurements .= measurements./std_measurement
    function measure_raw_adjoint(measurements_)
        measurements_ .= measurements_ ./ std_measurement
        function gh(ϵ, l)
            n_basis = number_of_basis_functions(params.solver)
            res = zeros(n_basis.x.p*n_basis.Ω.p + n_basis.x.m*n_basis.Ω.m)
            for (i, ghx) ∈ enumerate(params.gh.x)
                for (j, ghΩ) ∈ enumerate(params.gh.Ω)
                    for (k, ghϵ) ∈ enumerate(params.gh.ϵ)
                        res .+= measurements_[i, j, k, l] * 2.0*ghϵ(ϵ) * vcat(ghx.p⊗ghΩ.p, ghx.m⊗ghΩ.m)
                    end
                end
            end
            return res
        end
        ## hacked ... (first we project all the x.p basis to L2 -> orthogonal basis and we dont need space derivatives from now on...)
        ψxss_proj = [[project_disassemble_solution(params.solver, ψx, params.proj) for ψx ∈ ψxs] for ψxs ∈ ψxss]
        ρ_vals_ = zeros(size(ρ_vals))
        ϵs = range(params.ϵ[1], params.ϵ[2], params.N)
        Δϵ = ϵs[2] - ϵs[1]
        for (l, (μhx, μhϵ)) ∈ enumerate(zip(params.μh.x, params.μh.ϵ))
            _, ψs_ = solve_forward(params.solver, params.X, params.Ω, params.physics, params.ϵ, params.N, ϵ -> gh(ϵ, l))
            ψs_proj = [project_disassemble_solution(params.solver, ψ, params.proj) for  ψ ∈ ψs_]
            K = Diagonal([diag(params.Ω.Kpp); diag(param.Ω.Kmm)])
            for i ∈ 1:2 # number of elements
                for k ∈ 1:params.N-1
                    s_2 = params.physics.s((ϵs[k] + ϵs[k+1])/2)
                    ρ_vals_[:, i] .+= params.gram * (s_2[i] .* sum(ψs_proj[k] .* ψxss_proj[l][k+1] .- ψs_proj[k+1] .* ψxss_proj[l][k], dims=1)[:])
                end
                
                # initial and final are 0.0 (because initial and final conditions of ψ and λ)
                for k ∈ 2:params.N-1
                    τ = params.physics.τ(ϵs[k])
                    σ = params.physics.σ(ϵs[k])
                    ρ_vals_[:, i] .+= params.gram * ((Δϵ*τ[i]).*sum(ψs_proj[k] .* ψxss_proj[l][k], dims=1)[:])
                    ρ_vals_[:, i] .-= params.gram * ((Δϵ*σ[i]).*sum(K * (ψs_proj[k] .* ψxss_proj[l][k]), dims=1)[:])
                end
            end
            #for i ∈ 1:2 # number of elements
                # final is 0.0 (forward solution)
                μ_ϵ = μhϵ(ϵs[1])
                μ_Ω = [params.μh.Ω[1].p; params.μh.Ω[1].m]
                ρ_vals_[:, l] .+= params.gram * ((Δϵ * μ_ϵ / 2) .* (μ_Ω' * ψs_proj[1]))[:]
                for k ∈ 2:params.N-1
                    μ_ϵ = μhϵ(ϵs[k])
                    ρ_vals_[:, l] .+= params.gram * ((Δϵ*μ_ϵ) .* (μ_Ω' * ψs_proj[k]))[:]
                end
            # add dot_c 
            # project to ρ_vals
            # end
        end
        # for i in 1:2
        #     ρ_vals_[:, i] .= params.gram * ρ_vals_[:, i]
        # end
        return (ρ_vals_, nothing)
    end
    return measurements, measure_raw_adjoint
end

function project_disassemble_solution(solver, ψ, projection)
    n_basis = number_of_basis_functions(solver)
    ψp = reshape(@view(ψ[1:n_basis.x.p*n_basis.Ω.p]), n_basis.Ω.p, n_basis.x.p)
    ψm = reshape(@view(ψ[1 + n_basis.x.p*n_basis.Ω.p:n_basis.x.p*n_basis.Ω.p + n_basis.x.m*n_basis.Ω.m]), n_basis.Ω.m, n_basis.x.m)
    return vcat(ψp*projection, ψm)
end

true_measurements = measure_raw(true_ρ_vals, param)

plot(true_measurements[:, 2, 1, 2])
plot!(true_measurements[:, 1, 1, 2])
plot!(true_measurements[:, 3, 1, 2])
p = plot(measurements[:, 2, 1, 1], color=1)
plot()
p = plot!(true_measurements[:, 2, 1, 1], color=1, linestyle=:dash)
p = plot!(measurements[:, 2, 1, 2], color=2)
p = plot!(true_measurements[:, 2, 1, 2], color=2, linestyle=:dash)
p = plot!(measurements[:, 1, 1, 1], color=3)
p = plot!(true_measurements[:, 1, 1, 1], color=3, linestyle=:dash)
counter = [0]

function objective(ρ_vals)
    n = length(true_measurements)
    measurements = measure_raw(ρ_vals, param)
    Zygote.ignore() do
        p = plot(measurements[:, 2, 1, 1], color=1)
        p = plot!(true_measurements[:, 2, 1, 1], color=1, linestyle=:dash)
        p = plot!(measurements[:, 2, 1, 2], color=2)
        p = plot!(true_measurements[:, 2, 1, 2], color=2, linestyle=:dash)
        p = plot!(measurements[:, 1, 1, 1], color=3)
        p = plot!(true_measurements[:, 1, 1, 1], color=3, linestyle=:dash)
        display(p)
        savefig(p, "measurements/plot$(counter[1]).pdf")
        counter[1] += 1
    end
    return 1/n * sum((true_measurements .- measurements).^2)
end

xys = mean.(Gridap.get_cell_coordinates(M.fe_basis.trian.grid))[:]
xys_mat = [xys[i][j] for j in 1:2, i in 1:length(xys)]

# model 1
NN = Chain(Dense(2, 10, tanh), Dense(10, 10, tanh), Dense(10, 2), Lux.softmax)

#model 2
NN = Chain(Dense(2, 10, tanh), Dense(10, 2), Lux.softmax)
ps, st = Lux.setup(Random.default_rng(), NN)
p0, restruct = Optimisers.destructure(ps)

function p_trans(p)
    NN_params = restruct(p)
    p_ = Lux.apply(NN, xys_mat, NN_params, st)[1]
    # pxs = choose_p.(xy, Ref(p))
    # p_ = Lux.NNlib.σ.(pxs)
    ρ_vals = p_[1, :] .* [0.0, 1.2]' .+ p_[2, :] .* [0.8, 0.0]'
    @show sum(ρ_vals, dims=2)
    @assert all(sum(ρ_vals, dims=2) .> 0.75)
    return ρ_vals
end

function fg!(F, G, x)
    if G !== nothing
        res = Zygote.withgradient(objective ∘ p_trans, x)
        G .= res.grad[1]
        return res.val
    end
    if F !== nothing
        return objective(p_trans(x))
    end
end

counter2 = [0]

function optim_callback(trace)
    m_temp = FEFunction(M, p_trans(trace[end].metadata["x"])[:, 2])
    p = contourf(-1:0.05:1, 0:0.05:1, (x, z) -> m_temp(Point(z, x)))
    display(p)
    savefig(p, "material/plot$(counter2[1]).pdf")
    serialize("traces/trace$(counter2[1]).jls", trace)
    counter2[1] += 1
    return false
end

res = Optim.optimize(Optim.only_fg!(fg!), p0, Optim.LBFGS(), Optim.Options(store_trace=true, extended_trace=true, iterations=1000, time_limit=60*60*30, g_abstol=1e-5, g_reltol=1e-5, callback=optim_callback))


# ### run only the following line:
# res_cont = Optim.optimize(Optim.only_fg!(fg!), res.trace[end].metadata["x"], res.method, Optim.Options(store_trace=true, extended_trace=true, iterations=1000, time_limit=60*60*10, g_abstol=1e-6, g_reltol=1e-6, callback=optim_callback))

# res_cont2 = Optim.optimize(Optim.only_fg!(fg!), res_cont.trace[end].metadata["x"], res_cont.method, Optim.Options(store_trace=true, extended_trace=true, iterations=1000, time_limit=60*60*10, g_abstol=1e-6, g_reltol=1e-6, callback=optim_callback))


nothing


#### end
@gif for i in 1:length(res.trace)
    m_temp = FEFunction(M, p_trans(res.trace[i].metadata["x"])[:, 1])
    p = contourf(-1:0.05:1, 0:0.05:1, (x, z) -> m_temp(Point(z, x)), clims=(0, 0.8))
    hline!([0.85])
    vline!([-0.15, 0.15])
    title!("iteration: $(i)")
end fps=5

trace_measurements = [measure_raw(p_trans(res.trace[i].metadata["x"]), param) for i in 1:length(res.trace)]

@gif for i in 1:length(res.trace)
    plot(true_measurements[:, 2, 1, 1], color=1)
    plot!(true_measurements[:, 2, 1, 2], color=2)
    plot!(true_measurements[:, 1, 1, 1], color=3)
    # measurements = measure_raw(p_trans(res.trace[i].metadata["x"]), param)
    plot!(trace_measurements[i][:, 2, 1, 1], color=1, linestyle=:dash)
    plot!(trace_measurements[i][:, 2, 1, 2], color=2, linestyle=:dash)
    plot!(trace_measurements[i][:, 1, 1, 1], color=3, linestyle=:dash)
    title!("interation: $(i)")
end fps=5


function finite_diff(f, x, h, index)
    val0 = f(x)
    x_ = copy(x)
    x_[index...] += h
    return (objective(x_) - val0)/h
end

finite_diff(objective, ρ_vals, 0.01, (875, 1))
grad[1][875, 1]

argmax(grad[1])

m = FEFunction(M, p_trans(grad.grad[1])[:, 1])
m = FEFunction(M, p_trans(p0)[:, 1])
m = FEFunction(M, p_trans(res.minimizer)[:, 2])
contourf(0:0.05:1, -1:0.05:1, (z, x) -> m(Point(z, x)))

@gif for i in 1:16
    m_i = FEFunction(M, p_trans(res.trace[i].metadata["x"])[:, 2])
    contourf(0:0.05:1, -1:0.05:1, (z, x) -> m_i(Point(z, x)), clims=(0, 1))
end fps=2
p_trans(p0)

# test_vec = reshape(ψs[50][1:n_basis.x.p*n_basis.Ω.p], n_basis.Ω.p, n_basis.x.p)
# test_vec_proj = test_vec * projection_matrix
# u = FEFunction(solver.U[1], test_vec[1, :])
# m = FEFunction(M, test_vec_proj[1, :])

# contourf(0:0.1:1, -1:0.1:1, (z, x) -> u(Point(z, x)))
# contourf(0:0.1:1, -1:0.1:1, (z, x) -> m(Point(z, x)))


# function integrate_direction(solver, ψ, ϕ)
#     n_basis = number_of_basis_functions(solver)
#     ψp = reshape(@view(ψ[1:n_basis.x.p*n_basis.Ω.p]), n_basis.Ω.p, n_basis.x.p)
#     ψm = reshape(@view(ψ[1 + n_basis.x.p*n_basis.Ω.p:n_basis.x.p*n_basis.Ω.p + n_basis.x.m*n_basis.Ω.m]), n_basis.Ω.m, n_basis.x.m)
#     ϕp = reshape(@view(ϕ[1:n_basis.x.p*n_basis.Ω.p]), n_basis.Ω.p, n_basis.x.p)
#     ϕm = reshape(@view(ϕ[1 + n_basis.x.p*n_basis.Ω.p:n_basis.x.p*n_basis.Ω.p + n_basis.x.m*n_basis.Ω.m]), n_basis.Ω.m, n_basis.x.m)
#     return (p=sum(ψp .* ϕp, dims=1)[:], m=sum(ψm .* ϕm, dims=1)[:])
# end

@time res = integrate_direction(solver, ψs[10], ψs[20])
res.p

plot(measurementsx2[:, 2, 1, 1])
plot!(measurementsx2[:, 2, 1, 2])

plot!(measurementsx2[:, 2, end, 1])
plot!(measurementsx2[:, 2, end, 2])

plot!(measurementsx2[:, 1, 1, 1])
plot!(measurementsx2[:, 1, 1, 2])

plot!(measurementsx2[:, 3, 1, 1])
plot!(measurementsx2[:, 3, 1, 2])

plot(measurementsx2[:, 2, 1, 1])
plot!(measurementsx2[:, 1, 1, 2])


z_coords = range(0, 1, length=100)
x_coords = range(-0.7, 0.7, length=150)

e_x = eval_space(solver.U, Point.(z_coords, x_coords'))
sol = zeros(length(z_coords), length(x_coords))
n_basis = number_of_basis_functions(solver)
e_Ω_p = spzeros(n_basis.Ω.p)
# e_Ω_p = gh.Ω[1].p
e_Ω_m = spzeros(n_basis.Ω.m)
e_Ω_p[1] = 1.0

cgrad_electron = cgrad([:black, :lightblue, :white])
cgrad_phase1 = cgrad([:black, :lightgreen, :white])
cgrad_phase2 = cgrad([:black, :orange, :white])
cgrad_phases = cgrad([:green, :orange])

#### VISUALIZATION OF THE EXPERIMENT

## VISUALIZATION OF THE MATERIAL
using LaTeXStrings

let
    z_coords_phase = range(0, 1, length=300)
    x_coords_phase = range(-0.7, 0.7, length=500)
    phases = reshape(getindex.(ρ.(Point.(z_coords_phase, x_coords_phase')[:]), 2), length(z_coords_phase), length(x_coords_phase))

    # p1 = scatter(range(-0.5, 0.5, 40)[1:10], true_measurements[1:10, 1, 1, 2], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), size=(600, 400))
    p1 = scatter(xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1))
    p1 = ylabel!("Intensity")

    # p1 = scatter!(range(-0.5, 0.5, 40)[10:10], true_measurements[10:10, 1, 1, 2], label=nothing, color=:lightblue)
    # p1 = xlims!(-0.75, 0.75)

    #p2 = contourf(x_coords, z_coords, max.(sol, 0.0), linewidth=0.0, levels=10, aspect_ratio=:equal, c=cgrad_electron, colorbar=false, xticks=-0.7:0.2:0.7)
    p2 = contourf(x_coords_phase, z_coords_phase, phases, linewidth=0.0, levels=10, c=cgrad_phases, colorbar=false, xticks=-0.7:0.2:0.7)
    p2 = plot!(ellipse_points(p_ellipse)..., color=:gray, label=nothing, linewidth=2)
    p2 = plot!([-0.75, 0.75], [1.0, 1.0], color=:black, label=nothing, linewidth=2)
    p2 = plot!([-0.3, -0.3], [1.07, 1.01], arrow=true, color=:white, label=nothing)
    p2 = xlabel!("Position x")
    p2 = ylabel!("Depth z")
    # p2 = scatter!([range(-0.5, 0.5, 40)[10]], [1.0], color=:white, markerstyle=:x, label=nothing)
    # p2 = plot!([range(-0.5, 0.5, 40)[10], range(-0.5, 0.5, 40)[10]], [1.1, 1.0], arrow=true, color=:black, label=nothing)

    plot(p1, p2, layout=(2, 1), link=:x, size=(650, 800))
    savefig("full_visualization/01_material.pdf")
end


## VISUALIZATION OF ONE FULL FORWARD SOLUTION
I_fwrd = 10
forward_sol = solve_forward(solver, X, Ω, physics, (0, 1), 100, ϵ -> 2*gh.ϵ[1](ϵ)*vcat(gh.x[I_fwrd].p⊗gh.Ω[2].p, gh.x[I_fwrd].m⊗gh.Ω[2].m))
let
    counter = [0]
    for k in 100:-1:1
        @show k
        for (i, x_) in enumerate(z_coords)
            for (j, y_) in enumerate(x_coords)
                e_x_p, e_x_m = e_x[i, j]
                full_basis = vcat(e_x_p⊗e_Ω_p, e_x_m⊗e_Ω_m)
                sol[i, j] = dot(forward_sol[2][k], full_basis)
                #proj_basis = vcat((μh.x[2].p .* e_x_p)⊗μh.Ω[1].p, (μh.x[2].m .* e_x_m)⊗μh.Ω[1].m)
                #sol[i, j] = μh.ϵ[2](forward_sol[1][k])*dot(forward_sol[2][k], proj_basis)
            end
        end
        # p1 = scatter(range(-0.5, 0.5, 40)[1:10], true_measurements[1:10, 1, 1, 2], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), size=(600, 400))
        p1 = plot(range(-0.7, 0.7, length=200), x -> 1/650*g.ϵ[1](forward_sol[1][k])*g.x[I_fwrd]([1.0, x]), xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=1)
        p1 = ylabel!("Beam Intensity")

        # p1 = scatter!(range(-0.5, 0.5, 40)[10:10], true_measurements[10:10, 1, 1, 2], label=nothing, color=:lightblue)
        # p1 = xlims!(-0.75, 0.75)

        p2 = contourf(x_coords, z_coords, max.(sol, 0.0), linewidth=0.0, levels=10, c=cgrad_electron, colorbar=false, clims=(0, 140), xticks=-0.7:0.2:0.7)
        p2 = plot!(ellipse_points(p_ellipse)..., color=:gray, label=nothing, linewidth=2)
        p2 = plot!([-0.75, 0.75], [1.0, 1.0], color=:black, label=nothing, linewidth=2)
        p2 = plot!([range(-0.5, 0.5, length=40)[I_fwrd], range(-0.5, 0.5, length=40)[I_fwrd]], [1.07, 1.01], arrow=true, color=:black, label=nothing)
        p2 = xlabel!("Position x")
        p2 = ylabel!("Depth z")
        # p2 = scatter!([range(-0.5, 0.5, 40)[10]], [1.0], color=:white, markerstyle=:x, label=nothing)
        # p2 = plot!([range(-0.5, 0.5, 40)[10], range(-0.5, 0.5, 40)[10]], [1.1, 1.0], arrow=true, color=:black, label=nothing)

        plot(p1, p2, layout=(2, 1), link=:x, size=(650, 800))
        savefig("full_visualization/02_forward/$(string(counter[1], pad=3))_fluence.pdf")
        counter[1] += 1
    end 
    # savefig("material.pdf")
end

## VISUALIZATION OF THE IONIZATION
let
    counter = [0]
    sol1 = zeros(length(z_coords), length(x_coords))
    sol2 = zeros(length(z_coords), length(x_coords))

    for k in 100:-1:1
        @show k
        for (i, x_) in enumerate(z_coords)
            for (j, y_) in enumerate(x_coords)
                e_x_p, e_x_m = e_x[i, j]
                #full_basis = vcat(e_x_p⊗e_Ω_p, e_x_m⊗e_Ω_m)
                # sol[i, j] = dot(forward_sol[2][k], full_basis)
                if is_in_ellipse([x_, y_], p_ellipse)
                    proj_basis1 = vcat((e_x_p .* μh.x[1].p)⊗e_Ω_p, (e_x_m .* μh.x[1].m)⊗e_Ω_m)
                    sol1[i, j] = μh.ϵ[1](forward_sol[1][k])*dot(forward_sol[2][k], proj_basis1)
                    sol2[i, j] = NaN
                else
                    proj_basis2 = vcat((e_x_p .* μh.x[2].p)⊗e_Ω_p, (e_x_m .* μh.x[2].m)⊗e_Ω_m)
                    sol1[i, j] = 0.0
                    sol2[i, j] = μh.ϵ[2](forward_sol[1][k])*dot(forward_sol[2][k], proj_basis2)
                end
            end
        end
        # p1 = scatter(range(-0.5, 0.5, 40)[1:10], true_measurements[1:10, 1, 1, 2], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), size=(600, 400))
        p1 = plot(range(-0.7, 0.7, length=200), x -> 1/650*g.ϵ[1](forward_sol[1][k])*g.x[I_fwrd]([1.0, x]), xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=1)
        p1 = ylabel!("Beam Intensity")

        # p1 = scatter!(range(-0.5, 0.5, 40)[10:10], true_measurements[10:10, 1, 1, 2], label=nothing, color=:lightblue)
        # p1 = xlims!(-0.75, 0.75)

        p2 = contourf(x_coords, z_coords, max.(sol1, 0.0), linewidth=0.0, levels=10, clims=(0, 0.03), c=cgrad_phase1, colorbar=false, xticks=-0.7:0.2:0.7)
        p2 = contourf!(x_coords, z_coords, max.(sol2, 0.0), linewidth=0.0, levels=10, clims=(0, 0.03), c=cgrad_phase2, colorbar=false, xticks=-0.7:0.2:0.7)
        p2 = plot!(ellipse_points(p_ellipse, true)..., color=:black, label=nothing, linewidth=5)
        p2 = plot!(ellipse_points(p_ellipse)..., color=:gray, label=nothing, linewidth=2)
        p2 = plot!([-0.75, 0.75], [1.0, 1.0], color=:black, label=nothing, linewidth=2)
        p2 = plot!([range(-0.5, 0.5, length=40)[I_fwrd], range(-0.5, 0.5, length=40)[I_fwrd]], [1.07, 1.01], arrow=true, color=:black, label=nothing)
        p2 = xlabel!("Position x")
        p2 = ylabel!("Depth z")
        # p2 = scatter!([range(-0.5, 0.5, 40)[10]], [1.0], color=:white, markerstyle=:x, label=nothing)
        # p2 = plot!([range(-0.5, 0.5, 40)[10], range(-0.5, 0.5, 40)[10]], [1.1, 1.0], arrow=true, color=:black, label=nothing)

        plot(p1, p2, layout=(2, 1), link=:x, size=(650, 800))
        savefig("full_visualization/03_ionization/$(string(counter[1], pad=3))_ionization.pdf")
        counter[1] += 1
    end
end

## VISUALIZATION OF THE INTERACTION VOLUME
let
    Ωint = [Diagonal(ones(n_basis.x.p)) ⊗ e_Ω_p spzeros(n_basis.x.p*n_basis.Ω.p, n_basis.x.m)
            spzeros(n_basis.x.m*n_basis.Ω.m, n_basis.x.p) Diagonal(ones(n_basis.x.m)) ⊗ e_Ω_m]
    temp1 = zeros(n_basis.x.p + n_basis.x.m)
    temp2 = zeros(n_basis.x.p + n_basis.x.m)
    # temp2 = zeros(forward_sol[2][1] |> size)
    Δϵ = forward_sol[1][2] - forward_sol[1][1]

    for k in 1:100
        temp1 += (Δϵ* μh.ϵ[1](forward_sol[1][k])).*(Ωint'*forward_sol[2][k])
        temp2 += (Δϵ* μh.ϵ[2](forward_sol[1][k])).*(Ωint'*forward_sol[2][k])
    end

    temp1 .*= vcat(μh.x[1].p, μh.x[1].m)
    temp2 .*= vcat(μh.x[2].p, μh.x[2].m)
    
    sol1 = zeros(length(z_coords), length(x_coords))
    sol2 = zeros(length(z_coords), length(x_coords))
    for (i, x_) in enumerate(z_coords)
        for (j, y_) in enumerate(x_coords)
            e_x_p, e_x_m = e_x[i, j]
            full_basis = vcat(e_x_p, e_x_m)
            if is_in_ellipse([x_, y_], p_ellipse)
                # proj_basis1 = vcat((e_x_p .* μh.x[1].p)⊗e_Ω_p, (e_x_m .* μh.x[1].m)⊗e_Ω_m)
                sol1[i, j] = dot(temp1, full_basis)
                sol2[i, j] = NaN
            else
                # proj_basis2 = vcat((e_x_p .* μh.x[2].p)⊗e_Ω_p, (e_x_m .* μh.x[2].m)⊗e_Ω_m)
                sol1[i, j] = 0.0
                sol2[i, j] = dot(temp2, full_basis)
            end
        end
    end
    # p1 = scatter(range(-0.5, 0.5, 40)[1:10], true_measurements[1:10, 1, 1, 2], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), size=(600, 400))
    p1 = scatter([range(-0.5, 0.5, length=40)[I_fwrd]], [true_measurements[I_fwrd, 1, 1, 1]], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=:green, markerstrokewidth=0.5)
    p1 = scatter!([range(-0.5, 0.5, length=40)[I_fwrd]], [true_measurements[I_fwrd, 1, 1, 2]], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=:orange, markerstrokewidth=0.5)
    p1 = ylabel!("Photon Intensity")

    # p1 = scatter!(range(-0.5, 0.5, 40)[10:10], true_measurements[10:10, 1, 1, 2], label=nothing, color=:lightblue)
    # p1 = xlims!(-0.75, 0.75)

    p2 = contourf(x_coords, z_coords, max.(sol1, 0.0), linewidth=0.0, levels=10, c=cgrad_phase1, colorbar=false, xticks=-0.7:0.2:0.7)
    p2 = contourf!(x_coords, z_coords, max.(sol2, 0.0), linewidth=0.0, levels=10, c=cgrad_phase2, colorbar=false, xticks=-0.7:0.2:0.7)
    p2 = plot!(ellipse_points(p_ellipse, true)..., color=:black, label=nothing, linewidth=5)
    p2 = plot!(ellipse_points(p_ellipse)..., color=:gray, label=nothing, linewidth=2)
    p2 = plot!([-0.75, 0.75], [1.0, 1.0], color=:black, label=nothing, linewidth=2)
    p2 = plot!([range(-0.5, 0.5, length=40)[I_fwrd], range(-0.5, 0.5, length=40)[I_fwrd]], [1.07, 1.01], arrow=true, color=:black, label=nothing)
    p2 = xlabel!("Position x")
    p2 = ylabel!("Depth z")
    # p2 = scatter!([range(-0.5, 0.5, 40)[10]], [1.0], color=:white, markerstyle=:x, label=nothing)
    # p2 = plot!([range(-0.5, 0.5, 40)[10], range(-0.5, 0.5, 40)[10]], [1.1, 1.0], arrow=true, color=:black, label=nothing)

    plot(p1, p2, layout=(2, 1), link=:x, size=(650, 800))
    savefig("full_visualization/04_interaction_volume.pdf")
end

## VISUALIZATION OF A LINESCAN

function compute_and_integrate_not_space(i_x, i_ϵ, i_Ω)
    forward_sol = solve_forward(solver, X, Ω, physics, (0, 1), 100, ϵ -> 2*gh.ϵ[i_ϵ](ϵ)*vcat(gh.x[i_x].p⊗gh.Ω[i_Ω].p, gh.x[i_x].m⊗gh.Ω[i_Ω].m))
    Ωint = [Diagonal(ones(n_basis.x.p)) ⊗ e_Ω_p spzeros(n_basis.x.p*n_basis.Ω.p, n_basis.x.m)
            spzeros(n_basis.x.m*n_basis.Ω.m, n_basis.x.p) Diagonal(ones(n_basis.x.m)) ⊗ e_Ω_m]
    temp1 = zeros(n_basis.x.p + n_basis.x.m)
    temp2 = zeros(n_basis.x.p + n_basis.x.m)
    # temp2 = zeros(forward_sol[2][1] |> size)
    Δϵ = forward_sol[1][2] - forward_sol[1][1]

    for k in 1:100
        temp1 += (Δϵ* μh.ϵ[1](forward_sol[1][k])).*(Ωint'*forward_sol[2][k])
        temp2 += (Δϵ* μh.ϵ[2](forward_sol[1][k])).*(Ωint'*forward_sol[2][k])
    end

    temp1 .*= vcat(μh.x[1].p, μh.x[1].m)
    temp2 .*= vcat(μh.x[2].p, μh.x[2].m)
    return temp1, temp2
end

temps = [compute_and_integrate_not_space(i, 1, 2) for i in 1:5]
serialize("full_visualization/temps.jls", temps)

let
    sol1 = zeros(length(z_coords), length(x_coords))
    sol2 = zeros(length(z_coords), length(x_coords))
    counter = [0]
    for k in 1:length(temps)
        for (i, x_) in enumerate(z_coords)
            for (j, y_) in enumerate(x_coords)
                e_x_p, e_x_m = e_x[i, j]
                full_basis = vcat(e_x_p, e_x_m)
                if is_in_ellipse([x_, y_], p_ellipse)
                    # proj_basis1 = vcat((e_x_p .* μh.x[1].p)⊗e_Ω_p, (e_x_m .* μh.x[1].m)⊗e_Ω_m)
                    sol1[i, j] = dot(temps[k][1], full_basis)
                    sol2[i, j] = NaN
                else
                    # proj_basis2 = vcat((e_x_p .* μh.x[2].p)⊗e_Ω_p, (e_x_m .* μh.x[2].m)⊗e_Ω_m)
                    sol1[i, j] = 0.0
                    sol2[i, j] = dot(temps[k][2], full_basis)
                end
            end
        end

        # p1 = scatter(range(-0.5, 0.5, 40)[1:10], true_measurements[1:10, 1, 1, 2], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), size=(600, 400))
        p1 = plot(range(-0.5, 0.5, length=40)[1:k], [true_measurements[1:k, 2, 1, 1]], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=:green)
        p1 = plot!(range(-0.5, 0.5, length=40)[1:k], [true_measurements[1:k, 2, 1, 2]], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=:orange)
        p1 = scatter!([range(-0.5, 0.5, length=40)[k]], [true_measurements[k, 2, 1, 1]], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=:green, markerstrokewidth=0.5)
        p1 = scatter!([range(-0.5, 0.5, length=40)[k]], [true_measurements[k, 2, 1, 2]], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=:orange, markerstrokewidth=0.5)
        p1 = ylabel!("Photon Intensity")

        # p1 = scatter!(range(-0.5, 0.5, 40)[10:10], true_measurements[10:10, 1, 1, 2], label=nothing, color=:lightblue)
        # p1 = xlims!(-0.75, 0.75)

        p2 = contourf(x_coords, z_coords, max.(sol1, 0.0), linewidth=0.0, levels=10, c=cgrad_phase1, colorbar=false, xticks=-0.7:0.2:0.7)
        p2 = contourf!(x_coords, z_coords, max.(sol2, 0.0), linewidth=0.0, levels=10, c=cgrad_phase2, colorbar=false, xticks=-0.7:0.2:0.7)
        p2 = plot!(ellipse_points(p_ellipse, true)..., color=:black, label=nothing, linewidth=5)
        p2 = plot!(ellipse_points(p_ellipse)..., color=:gray, label=nothing, linewidth=2)
        p2 = plot!([-0.75, 0.75], [1.0, 1.0], color=:black, label=nothing, linewidth=2)
        p2 = plot!([range(-0.5, 0.5, length=40)[k], range(-0.5, 0.5, length=40)[k]], [1.07, 1.01], arrow=true, color=:black, label=nothing)
        p2 = xlabel!("Position x")
        p2 = ylabel!("Depth z")
        # p2 = scatter!([range(-0.5, 0.5, 40)[10]], [1.0], color=:white, markerstyle=:x, label=nothing)
        # p2 = plot!([range(-0.5, 0.5, 40)[10], range(-0.5, 0.5, 40)[10]], [1.1, 1.0], arrow=true, color=:black, label=nothing)

        plot(p1, p2, layout=(2, 1), link=:x, size=(650, 800))
        savefig("full_visualization/05_linescan1/$(string(counter[1], pad=3))_linescan1.pdf")
        counter[1] += 1
    end
end

## VISUALIZATION OF LINESCAN 2
temps2 = [compute_and_integrate_not_space(i, 1, 3) for i in 1:5]
serialize("full_visualization/temps2.jls", temps2)

let
    sol1 = zeros(length(z_coords), length(x_coords))
    sol2 = zeros(length(z_coords), length(x_coords))
    counter = [0]
    @gif for k in 1:40
        for (i, x_) in enumerate(z_coords)
            for (j, y_) in enumerate(x_coords)
                e_x_p, e_x_m = e_x[i, j]
                full_basis = vcat(e_x_p, e_x_m)
                if is_in_ellipse([x_, y_], p_ellipse)
                    # proj_basis1 = vcat((e_x_p .* μh.x[1].p)⊗e_Ω_p, (e_x_m .* μh.x[1].m)⊗e_Ω_m)
                    sol1[i, j] = dot(temps2[1][1], full_basis)
                    sol2[i, j] = NaN
                else
                    # proj_basis2 = vcat((e_x_p .* μh.x[2].p)⊗e_Ω_p, (e_x_m .* μh.x[2].m)⊗e_Ω_m)
                    sol1[i, j] = 0.0
                    sol2[i, j] = dot(temps2[1][2], full_basis)
                end
            end
        end

        # p1 = scatter(range(-0.5, 0.5, 40)[1:10], true_measurements[1:10, 1, 1, 2], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), size=(600, 400))
        p1 = plot(range(-0.5, 0.5, length=40), true_measurements[:, 2, 1, 1], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=:green)
        p1 = plot!(range(-0.5, 0.5, length=40), true_measurements[:, 2, 1, 2], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=:orange)
        p1 = plot!(range(-0.5, 0.5, length=40)[1:k], true_measurements[1:k, 3, 1, 1], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=:green, linestyle=:dash)
        p1 = plot!(range(-0.5, 0.5, length=40)[1:k], true_measurements[1:k, 3, 1, 2], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=:orange, linestyle=:dash)
        p1 = scatter!([range(-0.5, 0.5, length=40)[k]], [true_measurements[k, 3, 1, 1]], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=:green, markerstrokewidth=0.5)
        p1 = scatter!([range(-0.5, 0.5, length=40)[k]], [true_measurements[k, 3, 1, 2]], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=:orange, markerstrokewidth=0.5)
        p1 = ylabel!("Photon Intensity")

        # p1 = scatter!(range(-0.5, 0.5, 40)[10:10], true_measurements[10:10, 1, 1, 2], label=nothing, color=:lightblue)
        # p1 = xlims!(-0.75, 0.75)

        p2 = contourf(x_coords, z_coords, max.(sol1, 0.0), linewidth=0.0, levels=10, c=cgrad_phase1, colorbar=false, xticks=-0.7:0.2:0.7)
        p2 = contourf!(x_coords, z_coords, max.(sol2, 0.0), linewidth=0.0, levels=10, c=cgrad_phase2, colorbar=false, xticks=-0.7:0.2:0.7)
        p2 = plot!(ellipse_points(p_ellipse, true)..., color=:black, label=nothing, linewidth=5)
        p2 = plot!(ellipse_points(p_ellipse)..., color=:gray, label=nothing, linewidth=2)
        p2 = plot!([-0.75, 0.75], [1.0, 1.0], color=:black, label=nothing, linewidth=2)
        p2 = plot!([range(-0.5, 0.5, length=40)[k] + 0.03, range(-0.5, 0.5, length=40)[k]], [1.07, 1.01], arrow=true, color=:black, label=nothing)
        p2 = xlabel!("Position x")
        p2 = ylabel!("Depth z")
        # p2 = scatter!([range(-0.5, 0.5, 40)[10]], [1.0], color=:white, markerstyle=:x, label=nothing)
        # p2 = plot!([range(-0.5, 0.5, 40)[10], range(-0.5, 0.5, 40)[10]], [1.1, 1.0], arrow=true, color=:black, label=nothing)

        plot(p1, p2, layout=(2, 1), link=:x, size=(650, 800))
        #savefig("full_visualization/06_linescan2/$(string(counter[1], pad=3))_linescan2.pdf")
        counter[1] += 1
    end fps=5
end

## VISUALIZATION OF LINESCAN 2
temps3 = [compute_and_integrate_not_space(i, 5, 2) for i in 1:3]
serialize("full_visualization/temps3.jls", temps3)

let
    sol1 = zeros(length(z_coords), length(x_coords))
    sol2 = zeros(length(z_coords), length(x_coords))
    counter = [0]
    @gif for k in 1:3
        for (i, x_) in enumerate(z_coords)
            for (j, y_) in enumerate(x_coords)
                e_x_p, e_x_m = e_x[i, j]
                full_basis = vcat(e_x_p, e_x_m)
                if is_in_ellipse([x_, y_], p_ellipse)
                    # proj_basis1 = vcat((e_x_p .* μh.x[1].p)⊗e_Ω_p, (e_x_m .* μh.x[1].m)⊗e_Ω_m)
                    sol1[i, j] = dot(temps3[k][1], full_basis)
                    sol2[i, j] = NaN
                else
                    # proj_basis2 = vcat((e_x_p .* μh.x[2].p)⊗e_Ω_p, (e_x_m .* μh.x[2].m)⊗e_Ω_m)
                    sol1[i, j] = 0.0
                    sol2[i, j] = dot(temps3[k][2], full_basis)
                end
            end
        end

        # p1 = scatter(range(-0.5, 0.5, 40)[1:10], true_measurements[1:10, 1, 1, 2], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), size=(600, 400))
        p1 = plot(range(-0.5, 0.5, length=40), true_measurements[:, 2, 1, 1], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=:green)
        p1 = plot!(range(-0.5, 0.5, length=40), true_measurements[:, 2, 1, 2], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=:orange)
        p1 = plot!(range(-0.5, 0.5, length=40), true_measurements[:, 3, 1, 1], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=:green, linestyle=:dash)
        p1 = plot!(range(-0.5, 0.5, length=40), true_measurements[:, 3, 1, 2], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=:orange, linestyle=:dash)
        p1 = plot!(range(-0.5, 0.5, length=40)[1:k], true_measurements[1:k, 2, 5, 1], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=:green, linestyle=:dot)
        p1 = plot!(range(-0.5, 0.5, length=40)[1:k], true_measurements[1:k, 2, 5, 2], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=:orange, linestyle=:dot)
        p1 = scatter!([range(-0.5, 0.5, length=40)[k]], [true_measurements[k, 2, 5, 1]], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=:green, markerstrokewidth=0.5)
        p1 = scatter!([range(-0.5, 0.5, length=40)[k]], [true_measurements[k, 2, 5, 2]], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=:orange, markerstrokewidth=0.5)
        p1 = ylabel!("Photon Intensity")

        # p1 = scatter!(range(-0.5, 0.5, 40)[10:10], true_measurements[10:10, 1, 1, 2], label=nothing, color=:lightblue)
        # p1 = xlims!(-0.75, 0.75)

        p2 = contourf(x_coords, z_coords, max.(sol1, 0.0), linewidth=0.0, levels=10, c=cgrad_phase1, colorbar=false, xticks=-0.7:0.2:0.7)
        p2 = contourf!(x_coords, z_coords, max.(sol2, 0.0), linewidth=0.0, levels=10, c=cgrad_phase2, colorbar=false, xticks=-0.7:0.2:0.7)
        p2 = plot!(ellipse_points(p_ellipse, true)..., color=:black, label=nothing, linewidth=5)
        p2 = plot!(ellipse_points(p_ellipse)..., color=:gray, label=nothing, linewidth=2)
        p2 = plot!([-0.75, 0.75], [1.0, 1.0], color=:black, label=nothing, linewidth=2)
        p2 = plot!([range(-0.5, 0.5, length=40)[k], range(-0.5, 0.5, length=40)[k]], [1.07, 1.01], arrow=true, color=:black, label=nothing)
        p2 = xlabel!("Position x")
        p2 = ylabel!("Depth z")
        # p2 = scatter!([range(-0.5, 0.5, 40)[10]], [1.0], color=:white, markerstyle=:x, label=nothing)
        # p2 = plot!([range(-0.5, 0.5, 40)[10], range(-0.5, 0.5, 40)[10]], [1.1, 1.0], arrow=true, color=:black, label=nothing)

        plot(p1, p2, layout=(2, 1), link=:x, size=(650, 800))
        #savefig("full_visualization/07_linescan3/$(string(counter[1], pad=3))_linescan3.pdf")
        counter[1] += 1
    end fps=5
end

let
    sol1 = zeros(length(z_coords), length(x_coords))
    sol2 = zeros(length(z_coords), length(x_coords))
    counter = [0]
    for k = 1:40
        # for (i, x_) in enumerate(z_coords)
        #     for (j, y_) in enumerate(x_coords)
        #         e_x_p, e_x_m = e_x[i, j]
        #         full_basis = vcat(e_x_p, e_x_m)
        #         if is_in_ellipse([x_, y_], p_ellipse)
        #             # proj_basis1 = vcat((e_x_p .* μh.x[1].p)⊗e_Ω_p, (e_x_m .* μh.x[1].m)⊗e_Ω_m)
        #             sol1[i, j] = dot(temps3[k][1], full_basis)
        #             sol2[i, j] = NaN
        #         else
        #             # proj_basis2 = vcat((e_x_p .* μh.x[2].p)⊗e_Ω_p, (e_x_m .* μh.x[2].m)⊗e_Ω_m)
        #             sol1[i, j] = 0.0
        #             sol2[i, j] = dot(temps3[k][2], full_basis)
        #         end
        #     end
        # end

        # p1 = scatter(range(-0.5, 0.5, 40)[1:10], true_measurements[1:10, 1, 1, 2], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), size=(600, 400))
        p1 = plot(range(-0.5, 0.5, length=40), true_measurements[:, 2, 1, 1], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=:green)
        p1 = plot!(range(-0.5, 0.5, length=40), true_measurements[:, 2, 1, 2], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=:orange)
        p1 = plot!(range(-0.5, 0.5, length=40), true_measurements[:, 3, 1, 1], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=:green, linestyle=:dash)
        p1 = plot!(range(-0.5, 0.5, length=40), true_measurements[:, 3, 1, 2], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=:orange, linestyle=:dash)
        p1 = plot!(range(-0.5, 0.5, length=40), true_measurements[:, 2, 5, 1], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=:green, linestyle=:dot)
        p1 = plot!(range(-0.5, 0.5, length=40), true_measurements[:, 2, 5, 2], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=:orange, linestyle=:dot)
        for (i, j) ∈ [(1, 1), (1, 2), (2, 2), (3, 2), (1, 3), (2, 3), (3, 3), (1, 4), (2, 4), (3, 4), (1, 5), (3, 5), (1, 6), (2, 6), (3, 6)]
            p1 = plot!(range(-0.5, 0.5, length=40)[1:k], true_measurements[1:k, i, j, 1], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=:green, linestyle=:dashdot)
            p1 = plot!(range(-0.5, 0.5, length=40)[1:k], true_measurements[1:k, i, j, 2], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=:orange, linestyle=:dashdot)
            p1 = scatter!([range(-0.5, 0.5, length=40)[k]], [true_measurements[k, i, j, 1]], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=:green, markerstrokewidth=0.5)
            p1 = scatter!([range(-0.5, 0.5, length=40)[k]], [true_measurements[k, i, j, 2]], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 1.1), color=:orange, markerstrokewidth=0.5)
        end

        p1 = ylabel!("Photon Intensity")

        # p1 = scatter!(range(-0.5, 0.5, 40)[10:10], true_measurements[10:10, 1, 1, 2], label=nothing, color=:lightblue)
        # p1 = xlims!(-0.75, 0.75)

        p2 = contourf(x_coords, z_coords, fill(NaN, size(sol1)), linewidth=0.0, levels=10, clims=(0, 1), c=cgrad_phase1, colorbar=false, xticks=-0.7:0.2:0.7)
        # p2 = contourf!(x_coords, z_coords, max.(sol2, 0.0), linewidth=0.0, levels=10, c=cgrad_phase2, colorbar=false, xticks=-0.7:0.2:0.7)
        # p2 = plot!(ellipse_points(p_ellipse, true)..., color=:black, label=nothing, linewidth=5)
        p2 = plot!(ellipse_points(p_ellipse)..., color=:gray, label=nothing, linewidth=2)
        p2 = plot!([-0.75, 0.75], [1.0, 1.0], color=:black, label=nothing, linewidth=2)
        p2 = plot!([range(-0.5, 0.5, length=40)[k], range(-0.5, 0.5, length=40)[k]], [1.07, 1.01], arrow=true, color=:black, label=nothing)
        p2 = plot!([range(-0.5, 0.5, length=40)[k]-0.03, range(-0.5, 0.5, length=40)[k]], [1.07, 1.01], arrow=true, color=:black, label=nothing)
        p2 = plot!([range(-0.5, 0.5, length=40)[k]+0.03, range(-0.5, 0.5, length=40)[k]], [1.07, 1.01], arrow=true, color=:black, label=nothing)
        p2 = annotate!(0.0, 0.5, "too expensive to compute", color=:green)
        p2 = xlabel!("Position x")
        p2 = ylabel!("Depth z")
        # p2 = scatter!([range(-0.5, 0.5, 40)[10]], [1.0], color=:white, markerstyle=:x, label=nothing)
        # p2 = plot!([range(-0.5, 0.5, 40)[10], range(-0.5, 0.5, 40)[10]], [1.1, 1.0], arrow=true, color=:black, label=nothing)

        plot(p1, p2, layout=(2, 1), link=:x, size=(650, 800))
        savefig("full_visualization/08_full_measurements/$(string(counter[1], pad=3))_full_measurements.pdf")
        counter[1] += 1
    end
end

plot(range(0, 1, length=100), ϵ -> g.ϵ[1](ϵ))

plot(range(-0.7, 0.7, length=200), x -> 1/65 *g.x[I_fwrd]([1.0, x]), xticks=-0.7:0.2:0.7, label=nothing)

contourf(x_coords, z_coords, max.(sol, 0.0), linewidth=0.0, levels=10, aspect_ratio=:equal, clims=(0, 200), c=cgrad_electron, colorbar=false)
plot!(ellipse_points(p_ellipse)..., color=:gray, label=nothing)
#contourf(x_coords, z_coords, reverse(sol, dims=2), linewidth=0.0, levels=10, aspect_ratio=:equal)
#plot(x_coords, -sol[1, :])
title!("energy: $(round(forward_sol[1][k], digits=2))")

@gif for k in 100:-1:1
    @show k
    for (i, x_) in enumerate(z_coords)
        for (j, y_) in enumerate(x_coords)
            e_x_p, e_x_m = e_x[i, j]
            full_basis = vcat(e_x_p⊗e_Ω_p, e_x_m⊗e_Ω_m)
            sol[i, j] = dot(forward_sol[2][k], full_basis)
            #proj_basis = vcat((μh.x[2].p .* e_x_p)⊗μh.Ω[1].p, (μh.x[2].m .* e_x_m)⊗μh.Ω[1].m)
            #sol[i, j] = μh.ϵ[2](forward_sol[1][k])*dot(forward_sol[2][k], proj_basis)
        end
    end
    contourf(x_coords, z_coords, max.(sol, 0.0), linewidth=0.0, levels=10, aspect_ratio=:equal, clims=(0, 200), c=cgrad_electron, colorbar=false)
    plot!(ellipse_points(p_ellipse)..., color=:gray, label=nothing)
    #contourf(x_coords, z_coords, reverse(sol, dims=2), linewidth=0.0, levels=10, aspect_ratio=:equal)
    #plot(x_coords, -sol[1, :])
    title!("energy: $(round(forward_sol[1][k], digits=2))")
end fps=10


for k in 100:-1:1
    @show k
    for (i, x_) in enumerate(z_coords)
        for (j, y_) in enumerate(x_coords)
            e_x_p, e_x_m = e_x[i, j]
            #full_basis = vcat(e_x_p⊗e_Ω_p, e_x_m⊗e_Ω_m)
            #sol[i, j] = dot(forward_sol[2][k], full_basis)
            proj_basis = vcat((μh.x[2].p .* e_x_p)⊗μh.Ω[1].p, (μh.x[2].m .* e_x_m)⊗μh.Ω[1].m)
            sol[i, j] += μh.ϵ[2](forward_sol[1][k])*dot(forward_sol[2][k], proj_basis)
        end
    end
end


p1 = scatter(range(-0.5, 0.5, 40)[1:10], true_measurements[1:10, 1, 1, 2], xticks=-0.7:0.2:0.7, label=nothing, ylims=(-0.1, 2.1), xlims=(-0.71, 0.71), aspect_ratio=:equal)
p1 = scatter!(range(-0.5, 0.5, 40)[10:10], true_measurements[10:10, 1, 1, 2], label=nothing, color=:lightblue)
p1 = xlims!(-0.75, 0.75)

#p2 = contourf(x_coords, z_coords, max.(sol, 0.0), linewidth=0.0, levels=10, aspect_ratio=:equal, c=cgrad_electron, colorbar=false, xticks=-0.7:0.2:0.7)
p2 = contourf(x_coords, z_coords, phases, linewidth=0.0, levels=10, aspect_ratio=:equal, c=cgrad_phases, colorbar=false, xticks=-0.7:0.2:0.7, xlims=(-0.71, 0.71))
p2 = plot!(ellipse_points(p_ellipse)..., color=:gray, label=nothing, linewidth=3)
# p2 = scatter!([range(-0.5, 0.5, 40)[10]], [1.0], color=:white, markerstyle=:x, label=nothing)
p2 = plot!([range(-0.5, 0.5, 40)[10], range(-0.5, 0.5, 40)[10]], [1.1, 1.0], arrow=true, color=:black, label=nothing)

plot(p1, p2, layout=(2, 1), link=:x)

@gif for k in 100:-1:1
    @show k
    for (i, x_) in enumerate(z_coords)
        for (j, y_) in enumerate(x_coords)
            e_x_p, e_x_m = e_x[i, j]
            full_basis = vcat(e_x_p⊗e_Ω_p, e_x_m⊗e_Ω_m)
            sol[i, j] = dot(forward_sol[2][k], full_basis)
            #proj_basis = vcat((μh.x[2].p .* e_x_p)⊗μh.Ω[1].p, (μh.x[2].m .* e_x_m)⊗μh.Ω[1].m)
            #sol[i, j] = μh.ϵ[2](forward_sol[1][k])*dot(forward_sol[2][k], proj_basis)
        end
    end
    contourf(x_coords, z_coords, max.(sol, 0.0), linewidth=0.0, levels=10, aspect_ratio=:equal, clims=(0, 200), c=cgrad_electron, colorbar=false)
    plot!(ellipse_points(p_ellipse)..., color=:gray, label=nothing)
    #contourf(x_coords, z_coords, reverse(sol, dims=2), linewidth=0.0, levels=10, aspect_ratio=:equal)
    #plot(x_coords, -sol[1, :])
    title!("energy: $(round(forward_sol[1][k], digits=2))")
end fps=10
