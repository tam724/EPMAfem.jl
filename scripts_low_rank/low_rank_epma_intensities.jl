using Revise
using EPMAfem

using EPMAfem.PNLazyMatrices
using EPMAfem.Krylov
using EPMAfem.Gridap
using LinearAlgebra
using Plots
using LaTeXStrings
using BenchmarkTools
# include("plot_overloads.jl")
using NeXLCore
using Unitful
using Plots
NExt = Base.get_extension(EPMAfem, :NeXLCoreExt)

figpath = mkpath(joinpath(dirname(@__FILE__), "figures/2D_epma_adjoint"))

equations = NExt.epma_equations(
    [n"Al", n"Cr"],
    [NExt.EPMADetector(n"Al K-L2", VectorValue(1.0, 0.0, 0.0)), NExt.EPMADetector(n"Cr K-L2", VectorValue(1.0, 0.0, 0.0))],
    range(50u"eV", 20u"keV", length=100), 27)

meas1 = Dict()
meas2 = Dict()

ranks1 = Dict()
ranks2 = Dict()

N = 27
model = NExt.epma_model(equations, (-2000u"nm", 0.0u"nm", -2000u"nm", 2000u"nm"), (150, 300), N)
arch = EPMAfem.cuda(Float64)
problem = EPMAfem.discretize_problem(equations, model, arch, updatable=true)

excitation = EPMAfem.pn_excitation([(x=NExt.dimless(x_, equations.dim_basis), y=0.0) for x_ in range(-500u"nm", 500u"nm", 100)], [NExt.dimless(15u"keV", equations.dim_basis)], [VectorValue(-1.0, 0.0, 0.0)], beam_position_σ=0.02, beam_energy_σ=0.05, beam_direction_κ=50)
discrete_rhs = EPMAfem.discretize_rhs(excitation, model, arch)
discrete_ext = NExt.discretize_detectors(equations, model, arch, absorption=false)
discrete_ext[1].vector.bϵ .*= 0.01/maximum(discrete_ext[1].vector.bϵ) # (normalize TODO: there should be a general way to normalize coeffs)
discrete_ext[2].vector.bϵ .*= 0.01/maximum(discrete_ext[2].vector.bϵ)

# discrete_ext[1].vector.bϵ .= 0.01 # (normalize TODO: there should be a general way to normalize coeffs)
# discrete_ext[2].vector.bϵ .= 0.01


function mass_concentrations(e, x_)
    # return e == 1 ? NExt.dimless(n"Al".density, equations.dim_basis) : 0.0
    z = NExt.dimful(x_[1], u"nm", equations.dim_basis)
    x = NExt.dimful(x_[2], u"nm", equations.dim_basis)

    if (x - 80u"nm")^2 + (z - (-200u"nm"))^2 < (80u"nm")^2
        return e == 1 ? 0.0 : NExt.dimless(n"Cr".density, equations.dim_basis)
    else
        return e == 1 ? NExt.dimless(n"Al".density, equations.dim_basis) : 0.0
    end
end
ρs = EPMAfem.discretize_mass_concentrations([x -> mass_concentrations(e, x) for e in 1:EPMAfem.number_of_elements(equations)], model)
heatmap(reshape(ρs[1, :], 150, 300), aspect_ratio=:equal)
heatmap(reshape(ρs[2, :], 150, 300), aspect_ratio=:equal)

EPMAfem.update_problem!(problem, ρs)
EPMAfem.update_vector!(discrete_ext[1], ρs)
EPMAfem.update_vector!(discrete_ext[2], ρs)

system_full = EPMAfem.implicit_midpoint2(problem.problem, A -> PNLazyMatrices.schur_complement(A, Krylov.minres, PNLazyMatrices.cache ∘ LinearAlgebra.inv!));

# Ωp, Ωm = EPMAfem.SphericalHarmonicsModels.eval_basis(EPMAfem.direction_model(model), one) |> arch
# nb = EPMAfem.n_basis(model)
# for (ϵ, ψ) in discrete_ext[1].vector*system_
#     ψp, ψm = EPMAfem.pmview(ψ, model)

#     res = ψm*Ωm |> collect
#     p = heatmap(reshape(res, (40, 80)))
#     # func = EPMAfem.SpaceModels.interpolable((p=zeros(nb.nx.p), m=ψm*Ωm |> collect), EPMAfem.space_model(model))
#     # p = heatmap(-1500u"nm":1u"nm":1500u"nm", -1500u"nm":1u"nm":0u"nm", (x, z) -> func(VectorValue(NExt.dimless.((z, x), Ref(equations.dim_basis)))))
#     display(p)
#     sleep(0.1)
# end

timings = Dict()

sol = EPMAfem.IterableDiscretePNSolution(adjoint(system_full), discrete_ext[1].vector, step_callback=(ϵ, ψ) -> @show ϵ)
for _ in Iterators.take(sol, 2) end # warmup
timings[(1, "full")] = @elapsed meas1["full"] = ((sol)*discrete_rhs)[:]
for _ in Iterators.take(discrete_ext[2].vector*system_full, 2) end # warmup
timings[(2, "full")] = @elapsed meas2["full"] = ((discrete_ext[2].vector*system_full)*discrete_rhs)[:]

plot(meas1["full"])
plot(meas2["full"])

for (i_r, tol_r) in collect(enumerate([0.025, 0.0125, 0.00625]))
    system_lowrank = EPMAfem.implicit_midpoint_dlr5(problem.problem, max_ranks=(p=30, m=30), tolerance=tol_r);
    ranks1[(:noaug, tol_r)] = (p=zeros(length(EPMAfem.energy_model(model))), m=zeros(length(EPMAfem.energy_model(model))))
    sol = EPMAfem.IterableDiscretePNSolution(adjoint(system_lowrank), discrete_ext[1].vector; step_callback=(ϵ, ψ) -> begin
        @show ϵ
        ranks1[(:noaug, tol_r)].p[EPMAfem.plus½(ϵ)] = ψ.ranks.p[]
        ranks1[(:noaug, tol_r)].m[EPMAfem.plus½(ϵ)] = ψ.ranks.m[]
    end)
    for _ in Iterators.take(sol, 2) end# warmup
    timings[(1, :noaug, tol_r)] = @elapsed meas1[(:noaug, tol_r)] = (sol*discrete_rhs)[:]

    ranks2[(:noaug, tol_r)] = (p=zeros(length(EPMAfem.energy_model(model))), m=zeros(length(EPMAfem.energy_model(model))))
    sol = EPMAfem.IterableDiscretePNSolution(adjoint(system_lowrank), discrete_ext[2].vector; step_callback=(ϵ, ψ) -> begin
        @show ϵ
        ranks2[(:noaug, tol_r)].p[EPMAfem.plus½(ϵ)] = ψ.ranks.p[]
        ranks2[(:noaug, tol_r)].m[EPMAfem.plus½(ϵ)] = ψ.ranks.m[]
    end)
    for _ in Iterators.take(sol, 2) end # warmup
    timings[(2, :noaug, tol_r)] = @elapsed meas2[(:noaug, tol_r)] = (sol*discrete_rhs)[:]

    system_lowrank_aug = EPMAfem.implicit_midpoint_dlr5(problem.problem, max_ranks=(p=30, m=30), tolerance=tol_r, basis_augmentation=:mass);
    ranks1[(:aug, tol_r)] = (p=zeros(length(EPMAfem.energy_model(model))), m=zeros(length(EPMAfem.energy_model(model))))
    sol = EPMAfem.IterableDiscretePNSolution(adjoint(system_lowrank_aug), discrete_ext[1].vector; step_callback=(ϵ, ψ) -> begin
        @show ϵ
        ranks1[(:aug, tol_r)].p[EPMAfem.plus½(ϵ)] = ψ.ranks.p[]
        ranks1[(:aug, tol_r)].m[EPMAfem.plus½(ϵ)] = ψ.ranks.m[]
    end)
    for _ in Iterators.take(sol, 2) end# warmup
    timings[(1, :aug, tol_r)] = @elapsed meas1[(:aug, tol_r)] = (sol*discrete_rhs)[:]

    ranks2[(:aug, tol_r)] = (p=zeros(length(EPMAfem.energy_model(model))), m=zeros(length(EPMAfem.energy_model(model))))
    sol = EPMAfem.IterableDiscretePNSolution(adjoint(system_lowrank_aug), discrete_ext[2].vector; step_callback=(ϵ, ψ) -> begin
        @show ϵ
        ranks2[(:aug, tol_r)].p[EPMAfem.plus½(ϵ)] = ψ.ranks.p[]
        ranks2[(:aug, tol_r)].m[EPMAfem.plus½(ϵ)] = ψ.ranks.m[]
    end)
    for _ in Iterators.take(sol, 2) end # warmup
    timings[(2, :aug, tol_r)] = @elapsed meas2[(:aug, tol_r)] = (sol*discrete_rhs)[:]
end


plot(meas1["full"])
plot!(meas1[(:noaug, 0.025)])
plot!(meas1[(:noaug, 0.0125)])
plot!(meas1[(:noaug, 0.00625)])

plot!(meas1[(:aug, 0.025)])
plot!(meas1[(:aug, 0.0125)])
plot!(meas1[(:aug, 0.00625)])

plot(meas2["full"])
plot!(meas2[(:noaug, 0.025)])
plot!(meas2[(:noaug, 0.0125)])
plot!(meas2[(:noaug, 0.00625)])

plot!(meas2[(:aug, 0.025)])
plot!(meas2[(:aug, 0.0125)])
plot!(meas2[(:aug, 0.00625)])

plot(ranks1[(:aug, 0.00625)].p)
plot!(ranks1[(:aug, 0.00625)].m)
plot(ranks2[(:aug, 0.00625)].p)
plot!(ranks2[(:aug, 0.00625)].m)
