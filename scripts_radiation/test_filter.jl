using Revise
using EPMAfem

using EPMAfem.PNLazyMatrices
using EPMAfem.Krylov
using EPMAfem.Gridap
using LinearAlgebra
using Plots
using Distributions
using EPMAfem.HCubature
using LaTeXStrings

struct LineSourceEquations{S} <: EPMAfem.AbstractPNEquations end
EPMAfem.number_of_elements(::LineSourceEquations) = 1
EPMAfem.number_of_scatterings(::LineSourceEquations) = 1
EPMAfem.stopping_power(::LineSourceEquations, e, ϵ) = 1.0
EPMAfem.absorption_coefficient(eq::LineSourceEquations, e, ϵ) = 1.0
EPMAfem.scattering_coefficient(eq::LineSourceEquations, e, i, ϵ) = 1.0
EPMAfem.mass_concentrations(::LineSourceEquations, e, x) = 1.0

EPMAfem.scattering_kernel(::LineSourceEquations{Inf}, e, i) = μ -> 1/(4π)
@generated μ₀(::LineSourceEquations{T}) where T = return :( $ (2π*hquadrature(μ -> exp(-T*(μ-1)^2), -1, 1)[1]))
EPMAfem.scattering_kernel(eq::LineSourceEquations{T}, e, i) where T = μ -> exp(-T*(μ-1)^2)/μ₀(eq)

space_model = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((-1.5, 1.5), (300)))
direction_model = EPMAfem.SphericalHarmonicsModels.EOSphericalHarmonicsModel(21, 1)
direction_model2 = EPMAfem.SphericalHarmonicsModels.EOSphericalHarmonicsModel(47, 1)
energy_model = 0:0.01:1.0

model = EPMAfem.DiscretePNModel(space_model, energy_model, direction_model)
model2 = EPMAfem.DiscretePNModel(space_model, energy_model, direction_model2)

source = EPMAfem.Rank1DiscretePNVector(false, model, EPMAfem.cpu(), zeros(EPMAfem.n_basis(model).nϵ), zeros(EPMAfem.n_basis(model).nx.p), zeros(EPMAfem.n_basis(model).nΩ.p))
source2 = EPMAfem.Rank1DiscretePNVector(false, model2, EPMAfem.cpu(), zeros(EPMAfem.n_basis(model2).nϵ), zeros(EPMAfem.n_basis(model2).nx.p), zeros(EPMAfem.n_basis(model2).nΩ.p))
Mp = EPMAfem.SpaceModels.assemble_bilinear(EPMAfem.SpaceModels.∫R_uv, EPMAfem.space_model(model), EPMAfem.SpaceModels.even(EPMAfem.space_model(model)), EPMAfem.SpaceModels.even(EPMAfem.space_model(model)))
Mm = EPMAfem.SpaceModels.assemble_bilinear(EPMAfem.SpaceModels.∫R_uv, EPMAfem.space_model(model), EPMAfem.SpaceModels.odd(EPMAfem.space_model(model)), EPMAfem.SpaceModels.odd(EPMAfem.space_model(model)))

σ = 0.03
init_x(x) = 1/(σ*sqrt(2π))*exp(-1/2*(x[1]-0.0)^2/σ^2)
init_Ω(Ω) = 1.0 # pdf(VonMisesFisher([1, 0, 0], 2.0), [Ω...])
bxp = collect(Mp) \ EPMAfem.SpaceModels.assemble_linear(EPMAfem.SpaceModels.∫R_μv(init_x), EPMAfem.space_model(model), EPMAfem.SpaceModels.even(EPMAfem.space_model(model)))
bxm = collect(Mm) \ EPMAfem.SpaceModels.assemble_linear(EPMAfem.SpaceModels.∫R_μv(init_x), EPMAfem.space_model(model), EPMAfem.SpaceModels.odd(EPMAfem.space_model(model)))
bΩp = EPMAfem.SphericalHarmonicsModels.assemble_linear(EPMAfem.SphericalHarmonicsModels.∫S²_hv(init_Ω), EPMAfem.direction_model(model), EPMAfem.SphericalHarmonicsModels.even(EPMAfem.direction_model(model)))
bΩm = EPMAfem.SphericalHarmonicsModels.assemble_linear(EPMAfem.SphericalHarmonicsModels.∫S²_hv(init_Ω), EPMAfem.direction_model(model), EPMAfem.SphericalHarmonicsModels.odd(EPMAfem.direction_model(model)))
bΩp2 = EPMAfem.SphericalHarmonicsModels.assemble_linear(EPMAfem.SphericalHarmonicsModels.∫S²_hv(init_Ω), EPMAfem.direction_model(model2), EPMAfem.SphericalHarmonicsModels.even(EPMAfem.direction_model(model2)))
bΩm2 = EPMAfem.SphericalHarmonicsModels.assemble_linear(EPMAfem.SphericalHarmonicsModels.∫S²_hv(init_Ω), EPMAfem.direction_model(model2), EPMAfem.SphericalHarmonicsModels.odd(EPMAfem.direction_model(model2)))

nb = EPMAfem.n_basis(model)
initial_condition = EPMAfem.allocate_vec(EPMAfem.cpu(), nb.nx.p*nb.nΩ.p + nb.nx.m*nb.nΩ.m)
ψ0p, ψ0m = EPMAfem.pmview(initial_condition, model)
copy!(ψ0p, bxp .* bΩp')
copy!(ψ0m, bxm .* bΩm')

nb2 = EPMAfem.n_basis(model2)
initial_condition2 = EPMAfem.allocate_vec(EPMAfem.cpu(), nb2.nx.p*nb2.nΩ.p + nb2.nx.m*nb2.nΩ.m)
ψ0p2, ψ0m2 = EPMAfem.pmview(initial_condition2, model2)
copy!(ψ0p2, bxp .* bΩp2')
copy!(ψ0m2, bxm .* bΩm2')


equations = LineSourceEquations{Inf}()
problem = EPMAfem.discretize_problem(equations, model, EPMAfem.cpu())
system = EPMAfem.implicit_midpoint2(problem, A -> PNLazyMatrices.schur_complement(A, Krylov.minres, PNLazyMatrices.cache ∘ LinearAlgebra.inv!));
sol = EPMAfem.IterableDiscretePNSolution(system, source, initial_solution=initial_condition);

problem2 = EPMAfem.discretize_problem(equations, model2, EPMAfem.cpu())
system2 = EPMAfem.implicit_midpoint2(problem2, A -> PNLazyMatrices.schur_complement(A, Krylov.minres, PNLazyMatrices.cache ∘ LinearAlgebra.inv!));
sol2 = EPMAfem.IterableDiscretePNSolution(system2, source2, initial_solution=initial_condition2);


equations_filter = EPMAfem.filter_exp(LineSourceEquations{Inf}(), 1.0, 10)
problem_filter = EPMAfem.discretize_problem(equations_filter, model, EPMAfem.cpu())
system_filter = EPMAfem.implicit_midpoint2(problem_filter, A -> PNLazyMatrices.schur_complement(A, Krylov.minres, PNLazyMatrices.cache ∘ LinearAlgebra.inv!));
sol_filter = EPMAfem.IterableDiscretePNSolution(system_filter, source, initial_solution=initial_condition);

#### GIF
Ωp, Ωm = EPMAfem.SphericalHarmonicsModels.eval_basis(EPMAfem.direction_model(model), Ω -> 1.0) |> EPMAfem.architecture(problem)
Ωp2, Ωm2 = EPMAfem.SphericalHarmonicsModels.eval_basis(EPMAfem.direction_model(model2), Ω -> 1.0) |> EPMAfem.architecture(problem2)
@gif for ((ϵ, ψ), (ϵ2, ψ2), (ϵ_filter, ψ_filter)) in zip(sol, sol2, sol_filter)
    ψp, ψm = EPMAfem.pmview(ψ, model)
    ψp2, ψm2 = EPMAfem.pmview(ψ2, model2)
    ψp_filter, ψm_filter = EPMAfem.pmview(ψ_filter, model)
    func = EPMAfem.SpaceModels.interpolable((p=collect(ψp*Ωp), m=collect(ψm*Ωm)), EPMAfem.space_model(model))
    func2 = EPMAfem.SpaceModels.interpolable((p=collect(ψp2*Ωp2), m=collect(ψm2*Ωm2)), EPMAfem.space_model(model))
    func_filter = EPMAfem.SpaceModels.interpolable((p=collect(ψp_filter*Ωp), m=collect(ψm_filter*Ωm)), EPMAfem.space_model(model))
    plot(-1.5:0.01:1.5, x -> func(VectorValue(x)))
    plot!(-1.5:0.01:1.5, x -> func2(VectorValue(x)))
    plot!(-1.5:0.01:1.5, x -> func_filter(VectorValue(x)))
    # heatmap(-1.5:0.01:1.5, -1.5:0.01:1.5, (x, y) -> func(VectorValue(x, y)), aspect_ratio=:equal)
end
