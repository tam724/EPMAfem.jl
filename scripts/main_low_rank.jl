using Revise
using EPMAfem
using EPMAfem.CUDA
using Plots
using Gridap
using LinearAlgebra
include("plot_overloads.jl")

space_model = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((-1, 0, -1, 1), (50, 100)))
direction_model = EPMAfem.SphericalHarmonicsModels.EEEOSphericalHarmonicsModel(13, 2)

eq = EPMAfem.PNEquations()
beam = EPMAfem.pn_excitation([(x=0.0, y=0.0)], [1.8], [VectorValue(-1.0, 0.0, 0.0) |> normalize]; beam_position_σ=0.05, beam_energy_σ=0.05)

model = EPMAfem.DiscretePNModel(space_model, 0.0:0.02:2.0, direction_model)

problem = EPMAfem.discretize_problem(eq, model, EPMAfem.cuda(), updatable=false)
system = EPMAfem.implicit_midpoint(problem, EPMAfem.PNSchurSolver)

rhs = EPMAfem.discretize_rhs(beam, model, EPMAfem.cuda())[1]

@gif for (ϵ, ψ) in system * rhs
    @show ϵ
    ψp, ψm = EPMAfem.pmview(ψ, model)
    plot(svd(ψp).S |> collect)
    plot!(svd(ψm).S |> collect)
end

probe = EPMAfem.PNProbe(model, EPMAfem.cuda(); Ω = Ω -> 1.0)
res = probe(system * rhs)

@gif for i in reverse(1:size(res.p, 2))
    func = EPMAfem.SpaceModels.interpolable((p=res.p[:, i], m=res.m[:, i]), EPMAfem.space_model(model))
    heatmap(-1:0.01:1, -1:0.01:0, (x, z) -> func(Gridap.VectorValue(z, x)), aspect_ratio=:equal)
end
