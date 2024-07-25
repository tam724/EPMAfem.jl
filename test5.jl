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

n_z = 100
model = CartesianDiscreteModel((0.0, 1.0, -1.0, 1.0), (n_z, 2*n_z))
pn_sys = build_solver(model, 21, 2)
pn_semi = pn_semidiscretization(pn_sys, pn_equ)

pn_semi_cu = cuda(pn_semi)

N = 400
pn_solver_exp = pn_expliciteulersolver(pn_semi_cu, (0.0, 1.0), N)

pn_solver_imp_schur = pn_schurimplicitmidpointsolver(pn_semi, (0.0, 1.0), N)

pn_solver_imp = pn_fullimplicitmidpointsolver(pn_semi_cu, (0.0, 1.0), N)

# pn_semi_cu = cuda(pn_semi)
# #pn_semi = pn_semid(pn_sys, pn_equ)

# pn_solver = pn_implicitmidpointsolver(pn_semi, (0.0, 1.0))
# pn_solver_cu = pn_implicitmidpointsolver(pn_semi_cu, (0.0, 1.0))

n_basis = number_of_basis_functions(pn_sys)

ψs_imp_schur = zeros(n_basis.x.p, pn_solver_imp_schur.N)
ψs_imp = zeros(n_basis.x.p, pn_solver_imp.N)
ψs_exp = zeros(n_basis.x.p, pn_solver_exp.N)

for (i, ϵ) in enumerate(forward(pn_solver_exp, (1, 5, 2)))
    ψ = current_solution(pn_solver_exp)
    #p = heatmap(reshape(pn_solver.b[1:n_basis.x.p], (n_z+1, 2*n_z+1)))
    #display(p)
    @show ϵ
    copyto!(@view(ψs_exp[:, i]), Vector(@view(ψ[1:n_basis.x.p])))
        # ϵs[i÷10] = ϵ 
end

for (i, ϵ) in enumerate(forward(pn_solver_imp, (1, 5, 2)))
    ψ = current_solution(pn_solver_imp)
    #p = heatmap(reshape(pn_solver.b[1:n_basis.x.p], (n_z+1, 2*n_z+1)))
    #display(p)
    @show ϵ
    copyto!(@view(ψs_imp[:, i]), Vector(@view(ψ[1:n_basis.x.p])))
        # ϵs[i÷10] = ϵ
end

@profview for (i, ϵ) in enumerate(forward(pn_solver_imp_schur, (1, 5, 2)))
    ψ = current_solution(pn_solver_imp_schur)
    #p = heatmap(reshape(pn_solver.b[1:n_basis.x.p], (n_z+1, 2*n_z+1)))
    #display(p)
    @show ϵ
    copyto!(@view(ψs_imp_schur[:, i]), Vector(@view(ψ[1:n_basis.x.p])))
        # ϵs[i÷10] = ϵ
    
end

@gif for i in 1:N÷2
    z_coords = range(0.0, 1.0, length=n_z+1)
    x_coords = range(-1.0, 1.0, length=2*n_z+1)
    temp_exp = reshape(@view(ψs_exp[1:n_basis.x.p, i*2]), (n_z+1, 2*n_z+1))
    #temp_imp = reshape(@view(ψs_imp[1:n_basis.x.p, i]), (n_z+1, 2*n_z+1))
    #temp_imp_schur = reshape(@view(ψs_imp_schur[1:n_basis.x.p, i]), (n_z+1, 2*n_z+1))

    p1 = Plots.heatmap(x_coords, z_coords, temp_exp)
    #p2 = Plots.heatmap(x_coords, z_coords, temp_imp)
    #p3 = Plots.heatmap(x_coords, z_coords, temp_imp_schur)
    #plot(p1, p3)
    # savefig("plots/$(i).pdf")
end fps=20

