using Revise
using EPMAfem
using Gridap
using LinearAlgebra
using Plots
using Distributions
include("plot_overloads.jl")
Makie.inline!(false)

# parameters
σ = 5.670374 * 10^-8 # Stefan Boltzmann constant
T_i = 1000
T_o = 300
beta_i = 10^3
beta_o = 10^-1


eq = EPMAfem.MonochromPNEquations()
space_model = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((0, 0.03), 2))
direction_model = EPMAfem.SphericalHarmonicsModels.EOSphericalHarmonicsModel(3, 1)
model = EPMAfem.DiscreteMonochromPNModel(space_model, direction_model)


problem = EPMAfem.discretize_problem(eq, model, EPMAfem.cpu())


# source term of RTE
function qx(x)
    print(x)
    if x[1]<=0.01
        return beta_i * σ/π * T_i^4
    else
        return beta_o * σ/π * T_o^4
    end
end

function qΩ(Ω)
    return 1.0
end

bc_source = EPMAfem.PNXΩSource(qx, qΩ)
rhs_source = EPMAfem.discretize_rhs(bc_source, model, EPMAfem.cpu())

system = EPMAfem.system(problem, EPMAfem.PNSchurSolver)

x_source = EPMAfem.allocate_solution_vector(system)

Ωp, Ωm = EPMAfem.SphericalHarmonicsModels.eval_basis(EPMAfem.direction_model(model), Ω -> 1.0) |> problem.arch

xp_source, xm_source = EPMAfem.pmview(x_source, model)

func_source = EPMAfem.SpaceModels.interpolable((p=xp_source*Ωp|> collect, m=xm_source*Ωm |> collect), space_model)

p1 = plot(0:0.001:0.03, x -> func_source((x,)))
