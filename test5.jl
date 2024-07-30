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
using InteractiveUtils

include("spherical_harmonics.jl")
using .SphericalHarmonicsMatrices
#include("kroneckerblockmatrix.jl")
#include("pnsystemmatrix.jl")
include("epma-fem.jl")
include("pnequations.jl")
include("overrides.jl")

include("pnsemidiscretization.jl")
include("pnexplicitimplicitmatrix.jl")
include("pnsolver.jl")
include("pnimplicitmidpoint.jl")
include("pnexpliciteuler.jl")
include("pniterators.jl")

pn_equ = dummy_equations(
    [0.85],                                                     # beam energy
    range(-0.5, 0.5, length=10),                                # beam position
    [[-0.5, 0.0, -0.5], [0.0, 0.0, -1.0], [0.5, 0.0, -0.5]],    # beam direction. Careful, this is (x, y, z), not (z, x, y)
    [0.1, 0.2])                                                 # extraction energy 

n_z = 60
model = CartesianDiscreteModel((0.0, 1.0, -1.0, 1.0), (n_z, 2*n_z))
pn_sys = build_solver(model, 11, 2)

pn_semi = pn_semidiscretization(pn_sys, pn_equ)

pn_semi_cu = cuda(pn_semi)

N = 50
pn_solver_exp = pn_expliciteulersolver(pn_semi, N)
pn_solver_exp_cu = pn_expliciteulersolver(pn_semi_cu, N)

# initialize!(pn_solver_exp)
# @time step_forward!(pn_solver_exp, 0.86, 0.85, (1, 1, 1));

pn_solver_imp = pn_fullimplicitmidpointsolver(pn_semi, N)
pn_solver_imp_cu = pn_fullimplicitmidpointsolver(pn_semi_cu, N)
pn_solver_imp_schur = pn_schurimplicitmidpointsolver(pn_semi, N)
pn_solver_imp_schur_cu = pn_schurimplicitmidpointsolver(pn_semi_cu, N)

# initialize!(pn_solver_imp_schur)
# @time step_forward!(pn_solver_imp_schur, 0.86, 0.85, (1, 1, 1));

# pn_solver_imp = pn_fullimplicitmidpointsolver(pn_semi_cu, (0.0, 1.0), N)

# pn_semi_cu = cuda(pn_semi)
# #pn_semi = pn_semid(pn_sys, pn_equ)

# pn_solver = pn_implicitmidpointsolver(pn_semi, (0.0, 1.0))
# pn_solver_cu = pn_implicitmidpointsolver(pn_semi_cu, (0.0, 1.0))

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
doit!(ψs1, pn_solver_imp_cu, n_basis)
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

@gif for i in 1:N
    z_coords = range(0.0, 1.0, length=n_z+1)
    x_coords = range(-1.0, 1.0, length=2*n_z+1)
    temp_1 = reshape(@view(ψs1[1:n_basis.x.p, i]), (n_z+1, 2*n_z+1))
    #temp_imp = reshape(@view(ψs_imp[1:n_basis.x.p, i]), (n_z+1, 2*n_z+1))
    temp_2 = reshape(@view(ψs2[1:n_basis.x.p, i]), (n_z+1, 2*n_z+1))
    temp_3 = reshape(@view(ψs3[1:n_basis.x.p, i]), (n_z+1, 2*n_z+1))
    temp_4 = reshape(@view(ψs4[1:n_basis.x.p, i]), (n_z+1, 2*n_z+1))

    p1 = Plots.heatmap(x_coords, z_coords, temp_1)
    p2 = Plots.heatmap(x_coords, z_coords, temp_2)
    p3 = Plots.heatmap(x_coords, z_coords, temp_3)
    p4 = Plots.heatmap(x_coords, z_coords, temp_4)
    plot(p1, p2, p3, p4)
    savefig("plots/$(i).pdf")
end fps=20

@gif for i in 1:N
    z_coords = range(0.0, 1.0, length=n_z+1)
    x_coords = range(-1.0, 1.0, length=2*n_z+1)
    y_coords = range(-1.0, 1.0, length=2*n_z+1)
    temp_4 = reshape(@view(ψs4[1:n_basis.x.p, i]), (n_z+1, 2*n_z+1, 2*n_z+1))

    heatmap(x_coords, y_coords, -temp_4[1, :, :], clim=(0, 0.8))
    savefig("plots/$(i).pdf")
end

surface(x_coords, y_coords, -reshape(@view(ψs4[1:n_basis.x.p, N÷ 2]), (n_z+1, 2*n_z+1, 2*n_z+1))[1, :, :])

