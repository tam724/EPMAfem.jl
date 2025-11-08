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
using EPMAfem.HCubature


struct PlaneSourceEquations{S} <: EPMAfem.AbstractPNEquations end
EPMAfem.number_of_elements(::PlaneSourceEquations) = 1
EPMAfem.number_of_scatterings(::PlaneSourceEquations) = 1
EPMAfem.stopping_power(::PlaneSourceEquations, e, ϵ) = 1.0
EPMAfem.absorption_coefficient(eq::PlaneSourceEquations, e, ϵ) = 0.0 # 1.0 
EPMAfem.scattering_coefficient(eq::PlaneSourceEquations, e, i, ϵ) = 0.0 #1.0
EPMAfem.mass_concentrations(::PlaneSourceEquations, e, x) = 1.0

EPMAfem.scattering_kernel(::PlaneSourceEquations{Inf}, e, i) = μ -> 1/(4π)
@generated μ₀(::PlaneSourceEquations{T}) where T = return :( $ (2π*hquadrature(μ -> exp(-T*(μ-1)^2), -1, 1)[1]))
EPMAfem.scattering_kernel(eq::PlaneSourceEquations{T}, e, i) where T = μ -> exp(-T*(μ-1)^2)/μ₀(eq)

energy_model = 0:0.01:1.0

T = Inf
N = 23
equations = PlaneSourceEquations{T}()
space_model = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((-1.5, 1.5), (200)))

σ = 0.08
init_x(x) = 1/(σ*sqrt(2π))*exp(-1/2*(x[1]-0.0)^2/σ^2)
init_Ω(Ω) = 1.0

Mp = EPMAfem.SpaceModels.assemble_bilinear(EPMAfem.SpaceModels.∫R_uv, space_model, EPMAfem.SpaceModels.plus(space_model), EPMAfem.SpaceModels.plus(space_model))
Mm = EPMAfem.SpaceModels.assemble_bilinear(EPMAfem.SpaceModels.∫R_uv, space_model, EPMAfem.SpaceModels.minus(space_model), EPMAfem.SpaceModels.minus(space_model))
bxp = collect(Mp) \ EPMAfem.SpaceModels.assemble_linear(EPMAfem.SpaceModels.∫R_μv(init_x), space_model, EPMAfem.SpaceModels.plus(space_model))
bxm = collect(Mm) \ EPMAfem.SpaceModels.assemble_linear(EPMAfem.SpaceModels.∫R_μv(init_x), space_model, EPMAfem.SpaceModels.minus(space_model))

sols = []
probes = []
for N in 1:10
    direction_model = EPMAfem.SphericalHarmonicsModels.EOSphericalHarmonicsModel(N, 1, :OE)
    model = EPMAfem.DiscretePNModel(space_model, energy_model, direction_model)
    problem = EPMAfem.discretize_problem(equations, model, EPMAfem.cpu())

    # source / boundary condition (here: zero)
    source = EPMAfem.Rank1DiscretePNVector(false, model, EPMAfem.cpu(), zeros(EPMAfem.n_basis(model).nϵ), (p=zeros(EPMAfem.n_basis(model).nx.p), m=zeros(EPMAfem.n_basis(model).nx.m)), (p=zeros(EPMAfem.n_basis(model).nΩ.p), m=zeros(EPMAfem.n_basis(model).nΩ.m)))
    system = EPMAfem.implicit_midpoint2(problem, A -> PNLazyMatrices.schur_complement(A, Krylov.minres, PNLazyMatrices.cache ∘ LinearAlgebra.inv!));

    # initial condition
    bΩp = EPMAfem.SphericalHarmonicsModels.assemble_linear(EPMAfem.SphericalHarmonicsModels.∫S²_hv(init_Ω), EPMAfem.direction_model(model), EPMAfem.SphericalHarmonicsModels.plus(EPMAfem.direction_model(model)))
    bΩm = EPMAfem.SphericalHarmonicsModels.assemble_linear(EPMAfem.SphericalHarmonicsModels.∫S²_hv(init_Ω), EPMAfem.direction_model(model), EPMAfem.SphericalHarmonicsModels.minus(EPMAfem.direction_model(model)))
    initial_condition = EPMAfem.allocate_solution_vector(system)
    ψ0p, ψ0m = EPMAfem.pmview(initial_condition, model)
    copy!(ψ0p, bxp .* bΩp')
    copy!(ψ0m, bxm .* bΩm')

    push!(sols, EPMAfem.IterableDiscretePNSolution(system, source, initial_solution=initial_condition));
    Ωp, Ωm = EPMAfem.SphericalHarmonicsModels.eval_basis(direction_model, Ω -> 1.0)
    push!(probes, ψ -> begin
        ψp, ψm = EPMAfem.pmview(ψ, model)
        return EPMAfem.SpaceModels.interpolable((p=ψp*Ωp, m=ψm*Ωm), EPMAfem.space_model(model))
    end)
end

@gif for states in zip(sols...)
    plot()
    for (i, s) in enumerate(states)
        plot!(-1:0.01:1, x -> probes[i](s[2])(VectorValue(x)))
    end
end


