using Revise

using Gridap
using Serialization
using SparseArrays
using HCubature
using LinearAlgebra
using Enzyme
using Distributions
using Plots
using UnsafeArrays

using IterativeSolvers
using Unitful
using UnitfulChainRules
using ConcreteStructs

using ForwardDiff
using Krylov
using CUDA
using Zygote
using Lux
using Optim, Lux, Random, Optimisers
using BenchmarkTools
using StaticArrays
using InteractiveUtils

include("epmaequations.jl")
include("pnequations.jl")
include("spherical_harmonics.jl")
using .SphericalHarmonicsMatrices
#include("kroneckerblockmatrix.jl")
#include("pnsystemmatrix.jl")
include("epma-fem.jl")
include("overrides.jl")

include("pnsemidiscretization.jl")
include("pnexplicitimplicitmatrix.jl")
include("pnsolver.jl")
include("pnimplicitmidpoint.jl")
include("pnexpliciteuler.jl")
include("pniterators.jl")
include("pnlowrank.jl")

epma_eq = dummy_epma_equations(
    [0.85u"keV"],                                               # beam energy
    range(-0.5u"nm", 0.5u"nm", length=10),                      # beam position
    [[-0.5, 0.0, -0.5], [0.0, 0.0, -1.0], [0.5, 0.0, -0.5]],    # beam direction. Careful, this is (x, y, z), not (z, x, y)
    10.0,                                                      # beam concentration
    [0.1u"keV", 0.2u"keV"],                                     # extraction energy
    [(1.0, 3.0, 10.0)]                                          # takeoff directions
)

pn_equ = PNEquations(epma_eq)

n_z = 70

space_model = CartesianDiscreteModel((0.0, 1.0, -1.0, 1.0, -1.0, 1.0), (n_z, 2*n_z, 2*n_z))
space_model = CartesianDiscreteModel((0.0, 1.0, -1.0, 1.0), (n_z, 2*n_z))
space_model = CartesianDiscreteModel((0.0, 1.0), (n_z))

# using GridapGmsh
# model2 = DiscreteModelFromFile("square.msh")

model = PNGridapModel(space_model, range(0, 1, 100), 11)
problem = discretize(pn_equ, model) |> cuda
solver = pn_schurimplicitmidpointsolver(pn_equ, model) |> cuda

nb = number_of_basis_functions(model)

@gif for ϵ in hightolow(problem, solver)
    @show ϵ
    ψ = Vector(current_solution(solver))
    heatmap(reshape(ψ[1:nb.x.p], (n_z+1, 2*n_z+1)))
end

@gif for ϵ in lowtohigh(problem, solver)
    @show ϵ
    ψ = current_solution(solver)
    heatmap(reshape(ψ[1:nb.x.p], (n_z+1, 2*n_z+1)))
end


it = forward(disc_eq, pn_solver_imp)

iterate(it, 100)


# pn_sys = build_solver(model, 21, 2)
# pn_sys2 = build_solver(model2, 21, 2)

grid = 
prob = DiscretePNProblem()
solv = PNSolver()
parameters = PNParams()

sol = solve(prob, solv, parameters)

epma_prob = DiscreteEPMAProblem()
solv = EPMASolver()
parameters = EPMAParams()


for ϵ ∈ forward(prob, solv, parameters)
    ψ = current_solution(solv)
    plot(ψ)
end

intensities = solve(epma_prob, parameters)

pn_semi = pn_semidiscretization(pn_sys, pn_equ, true)

A = Matrix(pn_semi.Ωpm[1]' * pn_semi.Ωpm[1])

pn_semi_cu = cuda(pn_semi)

model |> typeof |> fieldnames
model.grid.elements

N = 50
pn_solver_exp = pn_expliciteulersolver(pn_semi, N)
pn_solver_exp_cu = pn_expliciteulersolver(pn_semi_cu, N)

# initialize!(pn_solver_exp)
# @time step_forward!(pn_solver_exp, 0.86, 0.85, (1, 1, 1));

pn_solver_dlr = pn_dlrfullimplicitmidpointsolver(pn_semi_cu, N, 50)

pn_solver_imp = pn_fullimplicitmidpointsolver(pn_semi, N)
pn_solver_imp_cu = pn_fullimplicitmidpointsolver(pn_semi_cu, N)
pn_solver_imp_schur = pn_schurimplicitmidpointsolver(pn_semi, N)
pn_solver_imp_schur_cu = pn_schurimplicitmidpointsolver(pn_semi_cu, N)

n_basis = number_of_basis_functions(pn_sys)

ψs1 = zeros(n_basis.x.p, N)
# ψs_imp = zeros(n_basis.x.p, pn_solver_imp.N)
ψs2 = zeros(n_basis.x.p, N)
ψs3 = zeros(n_basis.x.p, N)
ψs4 = zeros(n_basis.x.p, N)

function new_mass_conc(x, e)
    z_ = x[1]
    x_ = x[2]
    if abs(z_) < 0.8
        return (0.0, 1.2)[e]
    else
        return (0.8, 0.0)[e]
    end
    return 1.0
end

ρs = [project_function(pn_sys.U[2], pn_sys.model, x -> new_mass_conc(x, e)).free_values for e in 1:number_of_elements(pn_equ)]

update_mass_concentrations!(pn_semi, ρs)

ψ_cpu = zeros(n_basis.x.p)
pranks = zeros(Int64, N)
mranks = zeros(Int64, N)

function doit!(storage, solver::PNDLRFullImplicitMidpointSolver, n_basis)
    for (i, ϵ) in enumerate(forward(solver, (1, 5, 2)))
        ψ = current_solution(solver)
        pranks[i] = solver.ranks[1]
        mranks[i] = solver.ranks[2]
        #p = heatmap(reshape(pn_solver.b[1:n_basis.x.p], (n_z+1, 2*n_z+1)))
        #display(p)
        @show ϵ
        copyto!(ψ_cpu, @view(ψ[1:n_basis.x.p]))
        copyto!(@view(storage[:, i]), ψ_cpu)
        # unsafe_copyto!(pointer(@view(storage[:, i])), pointer(ψ[1:n_basis.x.p]), n_basis.x.p)
            # ϵs[i÷10] = ϵ 
    end
end

function doit!(storage, solver, n_basis)
    for (i, ϵ) in enumerate(forward(solver, (1, 5, 2)))
        ψ = current_solution(solver)
        #p = heatmap(reshape(pn_solver.b[1:n_basis.x.p], (n_z+1, 2*n_z+1)))
        #display(p)
        @show ϵ
        copyto!(ψ_cpu, @view(ψ[1:n_basis.x.p]))
        copyto!(@view(storage[:, i]), ψ_cpu)
        # unsafe_copyto!(pointer(@view(storage[:, i])), pointer(ψ[1:n_basis.x.p]), n_basis.x.p)
            # ϵs[i÷10] = ϵ 
    end
end



@time doit!(ψs1, pn_solver_exp, n_basis)
using MKLSparse

pn_solver_dlr.ranks .= [1, 1]
doit!(ψs1, pn_solver_dlr, n_basis)
doit!(ψs2, pn_solver_imp_cu, n_basis)

doit!(ψs3, pn_solver_imp_schur, n_basis)
doit!(ψs2, pn_solver_imp, n_basis)
doit!(ψs4, pn_solver_imp_schur_cu, n_basis)

@time step_forward!(pn_solver_imp_schur, 0.9, 0.91, (1, 5, 2))
@time step_forward!(pn_solver_imp, 0.9, 0.91, (1, 5, 2))

@gif for i in 1:N
    z_coords = range(0.0, 1.0, length=n_z+1)
    #x_coords = range(-1.0, 1.0, length=2*n_z+1)
    temp_1 = reshape(@view(ψs3[1:n_basis.x.p, i]), (n_z+1))
    temp_2 = reshape(@view(ψs4[1:n_basis.x.p, i]), (n_z+1))
    #temp_imp = reshape(@view(ψs_imp[1:n_basis.x.p, i]), (n_z+1, 2*n_z+1))
    # temp_2 = reshape(@view(ψs2[1:n_basis.x.p, i]), (n_z+1, 2*n_z+1))
    # temp_3 = reshape(@view(ψs3[1:n_basis.x.p, i]), (n_z+1, 2*n_z+1))
    # temp_4 = reshape(@view(ψs4[1:n_basis.x.p, i]), (n_z+1, 2*n_z+1))

    p1 = Plots.plot(z_coords, temp_1)
    p2 = Plots.plot!(z_coords, temp_2)
    # p2 = Plots.heatmap(x_coords, z_coords, temp_2)
    # p3 = Plots.heatmap(x_coords, z_coords, temp_3)
    # p4 = Plots.heatmap(x_coords, z_coords, temp_4)
    #plot(p1, p2)
    # # savefig("plots/$(i).pdf")
end fps=20
plot(pranks)
plot!(mranks)

@gif for i in 1:N
    z_coords = range(0.0, 1.0, length=n_z+1)
    x_coords = range(-1.0, 1.0, length=2*n_z+1)
    temp_1 = reshape(@view(ψs1[1:n_basis.x.p, i]), (n_z+1, 2*n_z+1))
    #temp_imp = reshape(@view(ψs_imp[1:n_basis.x.p, i]), (n_z+1, 2*n_z+1))
    temp_2 = reshape(@view(ψs2[1:n_basis.x.p, i]), (n_z+1, 2*n_z+1))
    temp_3 = reshape(@view(ψs3[1:n_basis.x.p, i]), (n_z+1, 2*n_z+1))
    temp_4 = reshape(@view(ψs4[1:n_basis.x.p, i]), (n_z+1, 2*n_z+1))


    clims = (0, 0.15)
    p1 = Plots.heatmap(x_coords, z_coords, temp_1)
    p2 = Plots.heatmap(x_coords, z_coords, temp_2)
    p3 = Plots.heatmap(x_coords, z_coords, temp_3)
    p4 = Plots.heatmap(x_coords, z_coords, temp_4)
    Plots.plot(p1, p2, p3, p4)
    # Plots.plot(p1, p4)
    # savefig("plots/$(i).pdf")
end fps=20

@gif for i in 1:N
    z_coords = range(0.0, 1.0, length=n_z+1)
    x_coords = range(-1.0, 1.0, length=2*n_z+1)
    y_coords = range(-1.0, 1.0, length=2*n_z+1)
    temp_4 = reshape(@view(ψs1[1:n_basis.x.p, i]), (n_z+1, 2*n_z+1, 2*n_z+1))

    heatmap(x_coords, y_coords, -temp_4[1, :, :], clim=(0, 0.8))
    savefig("plots/$(i).pdf")
end

using GLMakie
Makie.inline!(false)
temp_4 = reshape(@view(ψs4[1:n_basis.x.p, 10]), (n_z+1, 2*n_z+1, 2*n_z+1))
# GLMakie.volume(temp_4)
# _, _, pl = GLMakie.contour(permutedims(temp_4[:, 1:51, 1:51], [2, 3, 1]))
_, _, pl = GLMakie.contour(x_coords[1:30], y_coords[1:end], z_coords[:], permutedims(temp_4[:, 1:30, :], (2, 3, 1)))
# _, _, pl = GLMakie.volume(x_coords[1:51], y_coords[1:51], z_coords[:], max.(permutedims(temp_4[:, 1:51, 1:51], [2, 3, 1]), 0.0), algorithm=:iso, isorange=0.1, isovalue=0.1)

pl2 = GLMakie.contour!(x_coords[52:end], y_coords[1:51], z_coords[:], permutedims(temp_4[:, 52:end, 1:51], [2, 3, 1]), levels=range(0.01, 0.1, length=20))
pl3 = GLMakie.contour!(x_coords[1:51], y_coords[52:end], z_coords[:], permutedims(temp_4[:, 1:51,52:end], [2, 3, 1]), levels=range(0.01, 0.1, length=20))
# _, _, pl = GLMakie.contour!(permutedims(temp_4[:, 52:end, 1:51], [2, 3, 1]))

for i in repeat(1:50, 10)
    temp_4 = reshape(@view(ψs1[1:n_basis.x.p, i]), (n_z+1, 2*n_z+1, 2*n_z+1))
    pl[4] = max.(permutedims(temp_4[:, 1:30, :], (2, 3, 1)), 0.0)
    # pl2[4] = max.(permutedims(temp_4[:, 52:end, 1:51], [2, 3, 1]), 0.0)
    # pl3[4] = max.(permutedims(temp_4[:, 1:51,52:end], [2, 3, 1]), 0.0)
    sleep(0.1)
end
surface(x_coords, y_coords, -reshape(@view(ψs4[1:n_basis.x.p, N÷ 2]), (n_z+1, 2*n_z+1, 2*n_z+1))[1, :, :])

