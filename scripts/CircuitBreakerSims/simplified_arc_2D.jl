using Revise
using EPMAfem
using Gridap
using LinearAlgebra
using Plots
using Distributions
using ConcreteStructs
include("plot_overloads.jl")
Makie.inline!(false)

# parameters
σ = 5.670374 * 10^-8 # Stefan Boltzmann constant
T_i = 1000
T_o = 300
beta_i = 10^3
beta_o = 10^-1

@concrete struct ElectricArc2DPNEquations <: EPMAfem.AbstractMonochromPNEquations end
EPMAfem.number_of_elements(eq::ElectricArc2DPNEquations) = 1
EPMAfem.scattering_coefficient(eq::ElectricArc2DPNEquations, e) = 0.0
EPMAfem.scattering_kernel(eq::ElectricArc2DPNEquations, e) = μ -> 0.0
EPMAfem.absorption_coefficient(eq::ElectricArc2DPNEquations, e) = 1.0
function EPMAfem.mass_concentrations(eq::ElectricArc2DPNEquations, e, x)
    if x[2] < 0.01
        return float(beta_i)
    else
        return float(beta_o)
    end
end

# heatmap(0:0.001:0.03, -1:0.1:1, (x,z) -> EPMAfem.mass_concentrations(eq, 1, Gridap.Point(z, x)))

## 2D
eq = ElectricArc2DPNEquations()
space_model = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((0, 0.03, -0.06, 0.06), (30, 120)))
direction_model = EPMAfem.SphericalHarmonicsModels.EOSphericalHarmonicsModel(3, 2)
model = EPMAfem.DiscreteMonochromPNModel(space_model, direction_model)


function qx((x,z))
    if x<=0.01
        return beta_i * σ/π * T_i^4
    else
        return beta_o * σ/π * T_o^4
    end
end
function qΩ(Ω)
    return 1.0
end

function fx_left(;x)
    return σ/π * T_i^4
end
function fΩ_left(Ω)
    return 1.0
end

function fx_bottom(;z)
    if z<=0.01
        return σ/π * T_i^4
    else
        return σ/π * T_o^4
    end
end
function fΩ_bottom(Ω)
    return 1.0
end

function fx_right(;x)
    return σ/π * T_o^4
end
function fΩ_right(Ω)
    return 1.0
end

function fx_top(;z)
    if z<=0.01
        return beta_i * σ/π * T_i^4
    else
        return beta_o * σ/π * T_o^4
    end
end
function fΩ_top(Ω)
    return 1.0
end

source = EPMAfem.PNXΩSource(qx, qΩ)

bc_left = EPMAfem.PNXΩBoundaryCondition(EPMAfem.Dimensions.Z(), EPMAfem.Dimensions.LeftBoundary(), fx_left, fΩ_left)
bc_bottom = EPMAfem.PNXΩBoundaryCondition(EPMAfem.Dimensions.X(), EPMAfem.Dimensions.LeftBoundary(), fx_bottom, fΩ_bottom)
bc_right = EPMAfem.PNXΩBoundaryCondition(EPMAfem.Dimensions.Z(), EPMAfem.Dimensions.RightBoundary(), fx_right, fΩ_right)
bc_top = EPMAfem.PNXΩBoundaryCondition(EPMAfem.Dimensions.X(), EPMAfem.Dimensions.RightBoundary(), fx_top, fΩ_top)

rhs_source = EPMAfem.discretize_rhs(source, model, EPMAfem.cpu())

rhs_bc_left = EPMAfem.discretize_rhs(bc_left, model, EPMAfem.cpu())
rhs_bc_bottom = EPMAfem.discretize_rhs(bc_bottom, model, EPMAfem.cpu())
rhs_bc_right = EPMAfem.discretize_rhs(bc_right, model, EPMAfem.cpu())
rhs_bc_top = EPMAfem.discretize_rhs(bc_top, model, EPMAfem.cpu())

problem = EPMAfem.discretize_problem(eq, model, EPMAfem.cpu())

system = EPMAfem.system(problem, EPMAfem.PNSchurSolver)

x = EPMAfem.allocate_solution_vector(system)
EPMAfem.solve(x, system, [rhs_bc_bottom, rhs_bc_left, rhs_bc_right, rhs_bc_top, rhs_source])

Ωp, Ωm = EPMAfem.SphericalHarmonicsModels.eval_basis(EPMAfem.direction_model(model), Ω -> 1.0) |> problem.arch

xp, xm = EPMAfem.pmview(x, model)

interpolated_result = EPMAfem.SpaceModels.interpolable((p=xp*Ωp|> collect, m=xm*Ωm |> collect), space_model)
    
p3 = Plots.contourf(0:0.001:0.03, -0.06:0.1:0.06, (x,z) -> interpolated_result(VectorValue(x,z)), cmap=:plasma)

plot(0:0.001:0.03, x -> interpolated_result(VectorValue(x,0)))
# Plots.contourf(0:0.001:0.03, -1:0.1:1, (x, z) -> qx(VectorValue(x, z)), cmap=:grays)