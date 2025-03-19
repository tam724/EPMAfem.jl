using Revise
using EPMAfem
using Gridap

using GridapGmsh
using LinearAlgebra
using Plots
using Distributions
include("plot_overloads.jl")
Makie.inline!(false)

grid_gen_2D((-0.5, 0.5, -0.5, 0.5), min_res=0.05, max_res=0.005)
space_model = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((-0.5, 0.5, -0.5, 0.5), (100, 100)))

# space_model = EPMAfem.SpaceModels.GridapSpaceModel(DiscreteModelFromFile("/tmp/tmp_msh.msh"); even=(order=1, conformity=:H1), odd=(order=1, conformity=:H1))
eq = EPMAfem.DegeneratePNEquations(VectorValue(-1.0, 0.1) |> normalize)
problem = EPMAfem.discretize(eq, space_model, EPMAfem.cpu())
system = EPMAfem.system(problem, EPMAfem.PNKrylovMinres2Solver)

smooth_in(x, l, r, α) = (tanh(α*(x-l)) - tanh(α*(x-r)))/2
function qx((z, x))
    return exp(-100*((z-0.4)^2))*smooth_in(x, -0.4, 0.4, 30.0)
end

b_source = EPMAfem.PNSpaceSource(x -> -qx(x))
rhs = EPMAfem.discretize(b_source, space_model, EPMAfem.cpu())

x = EPMAfem.allocate_solution_vector(system)
EPMAfem.solve(x, system, rhs)
((nxp, nxm), _) = EPMAfem.n_basis(problem)
xp, xm = @view(x[1:nxp]), @view(x[nxp+1:nxp+nxm])

func = EPMAfem.SpaceModels.interpolable((p=xp|>collect, m=xm |> collect), space_model)
heatmap(-0.5:0.01:0.5, -0.5:0.01:0.5, (x, z) -> func(VectorValue(z, x)))
surface(-0.5:0.01:0.5, -0.5:0.01:0.5, (x, z) -> func(VectorValue(z, x)))



## 1D
space_model = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((-0.5, 0.5), (20)))
eq = EPMAfem.DegeneratePNEquations(VectorValue(1.0) |> normalize)
problem = EPMAfem.discretize(eq, space_model, EPMAfem.cpu())

system = EPMAfem.system(problem, EPMAfem.PNKrylovMinres2Solver)

function qx((z, ))
    return exp(-100*(z^2))
end


b_source = EPMAfem.PNSpaceSource(x -> -qx(x))
rhs = EPMAfem.discretize(b_source, space_model, EPMAfem.cpu())

x = EPMAfem.allocate_solution_vector(system)
EPMAfem.solve(x, system, rhs)
((nxp, nxm), _) = EPMAfem.n_basis(problem)
xp, xm = @view(x[1:nxp]), @view(x[nxp+1:nxp+nxm])

func = EPMAfem.SpaceModels.interpolable((p=xp, m=xm), space_model)
plot(-0.5:0.001:0.5, (z) -> func(VectorValue(z)))
