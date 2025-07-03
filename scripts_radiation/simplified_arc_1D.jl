using Revise
using EPMAfem
using Gridap
using LinearAlgebra
using Plots
using Distributions
include("../scripts/plot_overloads.jl")
include("analytical_simplified_arc_1D.jl")
Makie.inline!(false)

# Define one-dimensional electric arc equations
struct ElectricArc1DPNEquations <: EPMAfem.AbstractMonochromPNEquations end
EPMAfem.number_of_elements(eq::ElectricArc1DPNEquations) = 1
EPMAfem.scattering_coefficient(eq::ElectricArc1DPNEquations, e) = 0.0
EPMAfem.scattering_kernel(eq::ElectricArc1DPNEquations, e) = μ -> 0.0
EPMAfem.absorption_coefficient(eq::ElectricArc1DPNEquations, e) = 1.0
function EPMAfem.mass_concentrations(eq::ElectricArc1DPNEquations, e, x)
    if x[1] <= 0.01 && x[1] >= -0.01
        # println("in")
        return float(β_i)
    else
        # println("out")
        return float(β_o)
    end
end

# Define Source Term and boundary conditions
function qx(z)
    if z[1]<=0.01 && z[1]>=-0.01
        return β_i * σ/π * T_i^4
    else
        return β_o * σ/π * T_o^4
    end
end
function qΩ(Ω)
    return 1.0
end

function fx_left()
    return σ/π * T_w^4
end
function fΩ_left(Ω)
    return 1.0
end

function fx_right()
    return σ/π * T_w^4
end
function fΩ_right(Ω)
    return 1.0
end

# solve whole problem for given N (PN order)
function solve_problem(N)
    eq = ElectricArc1DPNEquations()
    space_model = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((-0.03, 0.03), 100))
    direction_model = EPMAfem.SphericalHarmonicsModels.EOSphericalHarmonicsModel(N, 1)
    model = EPMAfem.DiscreteMonochromPNModel(space_model, direction_model)

    problem = EPMAfem.discretize_problem(eq, model, EPMAfem.cpu())

    source = EPMAfem.PNXΩSource(qx, qΩ)
    bc_left = EPMAfem.PNXΩBoundaryCondition(EPMAfem.Dimensions.Z(), EPMAfem.Dimensions.LeftBoundary(), fx_left, fΩ_left)
    bc_right = EPMAfem.PNXΩBoundaryCondition(EPMAfem.Dimensions.Z(), EPMAfem.Dimensions.RightBoundary(), fx_right, fΩ_right)

    rhs_source = EPMAfem.discretize_rhs(source, model, EPMAfem.cpu())
    rhs_bc_left = EPMAfem.discretize_rhs(bc_left, model, EPMAfem.cpu())
    rhs_bc_right = EPMAfem.discretize_rhs(bc_right, model, EPMAfem.cpu())

    system = EPMAfem.system(problem, EPMAfem.PNDirectSolver)

    x_all = EPMAfem.allocate_solution_vector(system)
    # x_source = EPMAfem.allocate_solution_vector(system)
    EPMAfem.solve(x_all, system, [rhs_source, rhs_bc_left, rhs_bc_right])
    # EPMAfem.solve(x_source, system, [rhs_source])
    println(size(x_all))

    Ωp, Ωm = EPMAfem.SphericalHarmonicsModels.eval_basis(EPMAfem.direction_model(model), Ω -> 1.0) |> problem.arch
    xp, xm = EPMAfem.pmview(x_all, model)
    println(size(Ωp))
    println(Ωm)
    println(size(xp))
    println(size(xm))
    p=xp*Ωp|> collect
    println(size(p))

    interpolable_all = EPMAfem.SpaceModels.interpolable((p=xp*Ωp|> collect, m=xm*Ωm |> collect), space_model)
    # interpolable_source = create_interpolable(x_source)
    return interpolable_all
end


interpolable_1 = solve_problem(1)
interpolable_3 = solve_problem(3)
interpolable_7 = solve_problem(7)
interpolable_21 = solve_problem(21)
interpolable_27 = solve_problem(27)


scaling_factor = interpolable_27(VectorValue(0.0))/slab_zeroth_moment(0.0)

# plot(0:0.001:0.03, z -> interpolable_1(VectorValue(z)), label="P1", size=(800, 500))
# plot!(-0.03:0.001:0.03, z -> interpolable_3(VectorValue(z)), label="P3")
# plot!(-0.03:0.001:0.03, z -> interpolable_7(VectorValue(z)), label="P7")
# plot!(0:0.001:0.03, z -> interpolable_21(VectorValue(z)), label="P21")


# plot!(0:0.001:0.03, z -> interpolable_source(VectorValue(z)), label="Only Source")
# plot!(0:0.0003:0.03, z -> scaling_factor*zeroth_moment(z), label="Analytical")
# p2 = plot(0:0.0003:0.03, z -> qx(z))
plot(0.0:0.0003:0.03, z -> zeroth_moment(z), label="Analytical, 1D in radius", size=(800, 500))
plot!(0.0:0.0003:0.03, z -> slab_zeroth_moment(z), label="Analytical, 1D in x", size=(800, 500))

## brute force the scaling scaling
plot!(0:0.0003:0.03, z -> 1/scaling_factor*interpolable_1(VectorValue(z)), label="P1")
plot!(0:0.0003:0.03, z -> 1/scaling_factor*interpolable_3(VectorValue(z)), label="P3")
plot!(0:0.0003:0.03, z -> 1/scaling_factor*interpolable_21(VectorValue(z)), label="P21")
plot!(0:0.0003:0.03, z -> 1/scaling_factor*interpolable_27(VectorValue(z)), label="P27")


# plot(0.0:0.0003:0.03, z -> scaling_factor*zeroth_moment(z), label="Analytical", size=(800, 500))
# plot!(0.0:0.0003:0.03, z -> scaling_factor*slab_zeroth_moment(z), label="Analytical", size=(800, 500))

# ## brute force the scaling scaling
# plot!(0:0.0003:0.03, z -> interpolable_1(VectorValue(z)), label="P1")
# plot!(0:0.0003:0.03, z -> interpolable_3(VectorValue(z)), label="P3")
# plot!(0:0.0003:0.03, z -> interpolable_21(VectorValue(z)), label="P21")
# plot!(0:0.0003:0.03, z -> interpolable_27(VectorValue(z)), label="P27")