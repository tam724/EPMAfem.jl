using Revise
using EPMAfem
include("plot_overloads.jl")

space_model = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((-1, 0, -1, 1, -1, 1), (10, 10, 10)))
direction_model = EPMAfem.SphericalHarmonicsModels.EOSphericalHarmonicsModel(13, 3)

equations = EPMAfem.PNEquations()
excitation = EPMAfem.pn_excitation([(x=0.0, y=0.0)], [0.8], [VectorValue(-1.0, 0.0, 0.0)])
extraction = EPMAfem.PNExtraction([0.1, 0.2], equations)

model = EPMAfem.DiscretePNModel(space_model, 0.0:0.01:1.0, direction_model)

prob = EPMAfem.discretize_problem(equations, model, EPMAfem.cpu())

plot(prob.direction_discretization.kp[1][1].diag)


discrete_system = EPMAfem.implicit_midpoint(updatable_pnproblem.problem, EPMAfem.PNSchurSolver; solver=EPMAfem.PNKrylovMinresSolver)

discrete_rhs = EPMAfem.discretize_rhs(excitation, model, EPMAfem.cuda())
discrete_ext = EPMAfem.discretize_extraction(extraction, model, EPMAfem.cuda(), updatable=true)


ψ = EPMAfem.saveall(discrete_system * discrete_rhs[1])

discrete_ext[1].vector * discrete_system * discrete_rhs[1]
discrete_ext[1].vector * ψ

ρs = EPMAfem.discretize_mass_concentrations(equations, model)

EPMAfem.update_vector!(discrete_ext[1], ρs)


ext = discrete_ext[1]

old = ext.bxp_updater
new = EPMAfem.PNAbsorption(model, EPMAfem.cuda(), old.ρ_proj, EPMAfem.compute_line_integral_contribs(space_model, VectorValue(1.0, 0.4) |> normalize), [1.0, 2.0], old.element_index)

new_ext = EPMAfem.UpdatableRank1DiscretePNVector(ext.vector, new, nothing, ext.n_parameters, nothing)
# probe = EPMAfem.PNProbe(model, EPMAfem.cuda(); Ω = Ω -> 1.0, ϵ = ϵ -> 1.0)
# f = EPMAfem.interpolable(probe, ψ)
# heatmap(-1:0.01:0, -1:0.01:1, (z, x) -> f(VectorValue(z, x)))


EPMAfem.update_vector!(new_ext, ρs)

@time EPMAfem.tangent(new_ext, ρs) * ψ



## test

function func2(p)
    EPMAfem.update_vector!(new_ext, p)
    return new_ext.vector * ψ
end

grad = finite_difference_grad(func2, ρs, 0.01)

plot(grad_ad[1, :])
plot!(grad[1, :])
