using Revise
using EPMAfem
using Gridap
using LinearAlgebra
using Plots
using Distributions
Makie.inline!(false)
include("plot_overloads.jl")

heatmap(-0.5:0.01:0.5, -0.5:0.01:0.5, (x, z) -> EPMAfem.mass_concentrations(eq, 1, Gridap.Point(z, x)))
plot(-1:0.01:1, μ -> EPMAfem.monochrom_scattering_kernel_func(eq, μ))

function scat_coeff(x)
    return sum(EPMAfem.mass_concentrations(eq, e, x)*EPMAfem.scattering_coefficient(eq, e) for e in 1:EPMAfem.number_of_elements(eq))
end


eq = EPMAfem.MonochromPNEquations()
space_model = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((-0.5, 0.5, -0.5, 0.5), (200, 200)))
direction_model = EPMAfem.SphericalHarmonicsModels.EEEOSphericalHarmonicsModel(27, 2)
model = EPMAfem.DiscreteMonochromPNModel(space_model, direction_model)

function fx((; z))
    return exp(-200*(z)^2)
end
function fΩ_left(Ω)
    return pdf(VonMisesFisher([0.0, 1.0, 0.0] |> normalize, 50.0), Ω)
end
function fΩ_right(Ω)
    return pdf(VonMisesFisher([0.0, -1.0, 0.0] |> normalize, 50.0), Ω)
end
bc_left = EPMAfem.PNXΩBoundaryCondition(EPMAfem.Dimensions.X(), EPMAfem.Dimensions.LeftBoundary(), fx, fΩ_left)
bc_right = EPMAfem.PNXΩBoundaryCondition(EPMAfem.Dimensions.X(), EPMAfem.Dimensions.RightBoundary(), fx, fΩ_right)


problem = EPMAfem.discretize_problem(eq, model, EPMAfem.cuda())
rhs_left = EPMAfem.discretize_rhs(bc_left, model, EPMAfem.cuda())
rhs_right = EPMAfem.discretize_rhs(bc_right, model, EPMAfem.cuda())

system = EPMAfem.system(problem, EPMAfem.PNSchurSolver)
x_left = EPMAfem.allocate_solution_vector(system)
x_right = EPMAfem.allocate_solution_vector(system)
EPMAfem.solve(x_left, system, rhs_left)
EPMAfem.solve(x_right, system, rhs_right)

Ωp, Ωm = EPMAfem.SphericalHarmonicsModels.eval_basis(EPMAfem.direction_model(model), Ω -> 1.0) |> arch

xp_left, xm_left = EPMAfem.pmview(x_left, model)
xp_right, xm_right = EPMAfem.pmview(x_right, model)
func_left = EPMAfem.SpaceModels.interpolable((p=xp_left*Ωp|> collect, m=xm_left*Ωm |> collect), space_model)
func_right = EPMAfem.SpaceModels.interpolable((p=xp_right*Ωp|> collect, m=xm_right*Ωm |> collect), space_model)
    
p1 = Plots.contourf(-0.5:0.001:0.5, -0.5:0.001:0.5, (z, x) -> func_left(VectorValue(x, z)), cmap=:grays, aspect_ratio=:equal)
p2 = Plots.contourf(-0.5:0.001:0.5, -0.5:0.001:0.5, (z, x) -> func_right(VectorValue(x, z)), cmap=:grays, aspect_ratio=:equal)
plot(p1, p2, size=(1500, 1000))



# 1D
eq = EPMAfem.MonochromPNEquations()
space_model = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((-0.5, 0.5), (50)))
direction_model = EPMAfem.SphericalHarmonicsModels.EEEOSphericalHarmonicsModel(7, 1)
model = EPMAfem.DiscreteMonochromPNModel(space_model, direction_model)

function fx((; ))
    return 1.0
end
function fΩ_left(Ω)
    return 1.0 # pdf(VonMisesFisher([0.0, 1.0, 0.0] |> normalize, 50.0), Ω)
end
function fΩ_right(Ω)
    return 1.0  #pdf(VonMisesFisher([0.0, -1.0, 0.0] |> normalize, 50.0), Ω)
end
bc_left = EPMAfem.PNXΩBoundaryCondition(EPMAfem.Dimensions.Z(), EPMAfem.Dimensions.LeftBoundary(), fx, fΩ_left)
bc_right = EPMAfem.PNXΩBoundaryCondition(EPMAfem.Dimensions.Z(), EPMAfem.Dimensions.RightBoundary(), fx, fΩ_right)

problem = EPMAfem.discretize_problem(eq, model, EPMAfem.cpu(Float64))
rhs_left = EPMAfem.discretize_rhs(bc_left, model, EPMAfem.cpu(Float64))
rhs_right = EPMAfem.discretize_rhs(bc_right, model, EPMAfem.cpu(Float64))

system = EPMAfem.system(problem, EPMAfem.PNSchurSolver)
# system = EPMAfem.system(problem, EPMAfem.PNKrylovMinresSolver)
x_left = EPMAfem.allocate_solution_vector(system)
x_right = EPMAfem.allocate_solution_vector(system)
EPMAfem.solve(x_left, system, rhs_left)
EPMAfem.solve(x_right, system, rhs_right)

Ωp, Ωm = EPMAfem.SphericalHarmonicsModels.eval_basis(EPMAfem.direction_model(model), Ω -> 1.0) |> problem.arch

xp_left, xm_left = EPMAfem.pmview(x_left, model)
xp_right, xm_right = EPMAfem.pmview(x_right, model)
func_left = EPMAfem.SpaceModels.interpolable((p=xp_left*Ωp|> collect, m=xm_left*Ωm |> collect), space_model)
func_right = EPMAfem.SpaceModels.interpolable((p=xp_right*Ωp|> collect, m=xm_right*Ωm |> collect), space_model)
    
p1 = Plots.plot(-0.5:0.001:0.5, (z) -> func_left(VectorValue(z)))
p2 = Plots.plot(-0.5:0.001:0.5, (z) -> func_right(VectorValue(z)))
plot(p1, p2, size=(1500, 1000))


mat = EPMAfem.assemble_from_op(system.A)
