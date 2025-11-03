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

meas1 = zeros(100)
meas1_low_pn = zeros(100)
meas1_lowrank = zeros(100, 4)
meas1_lowrank_mass = zeros(100, 4)
meas1_lowrank_massbc = zeros(100, 4)
meas2 = zeros(100)
meas2_lowrank = zeros(100, 4)
meas2_lowrank_mass = zeros(100, 4)

ranks1_p = zeros(100, 4)
ranks1_m = zeros(100, 4)
ranks2_p = zeros(100, 4)
ranks2_m = zeros(100, 4)
ranks1_mass_p = zeros(100, 4)
ranks1_mass_m = zeros(100, 4)
ranks2_mass_p = zeros(100, 4)
ranks2_mass_m = zeros(100, 4)

N = 7
model = NExt.epma_model(equations, (-2200u"nm", 0.0u"nm", -2200u"nm", 2200u"nm"), (100, 200), N)
T = Float64
problem = EPMAfem.discretize_problem(equations, model, EPMAfem.cuda(T), updatable=true)

excitation = EPMAfem.pn_excitation([(x=NExt.dimless(x_, equations.dim_basis), y=0.0) for x_ in range(-500u"nm", 500u"nm", 100)], [NExt.dimless(15u"keV", equations.dim_basis)], [VectorValue(-1.0, 0.0, 0.0)], beam_position_σ=0.02, beam_energy_σ=0.05, beam_direction_κ=80)
discrete_rhs = EPMAfem.discretize_rhs(excitation, model, EPMAfem.cuda(T))
discrete_ext = NExt.discretize_detectors(equations, model, EPMAfem.cuda(T), absorption=false)
discrete_ext[1].vector.bϵ .*= 1e15 # (normalize TODO: there should be a general way to normalize coeffs)
discrete_ext[2].vector.bϵ .*= 1e15
function mass_concentrations(e, x_)
    z = NExt.dimful(x_[1], u"nm", equations.dim_basis)
    x = NExt.dimful(x_[2], u"nm", equations.dim_basis)

    if (x - 80u"nm")^2 + (z - (-200u"nm"))^2 < (80u"nm")^2
        return e == 1 ? 0.0 : NExt.dimless(n"Cr".density, equations.dim_basis)
    else
        return e == 1 ? NExt.dimless(n"Al".density, equations.dim_basis) : 0.0
    end
end
ρs = EPMAfem.discretize_mass_concentrations([x -> mass_concentrations(e, x) for e in 1:EPMAfem.number_of_elements(equations)], model)
heatmap(reshape(ρs[1, :], 100, 200), aspect_ratio=:equal)
heatmap(reshape(ρs[2, :], 100, 200), aspect_ratio=:equal)

EPMAfem.update_problem!(problem, ρs)
EPMAfem.update_vector!(discrete_ext[1], ρs)
EPMAfem.update_vector!(discrete_ext[2], ρs)

system_full = EPMAfem.implicit_midpoint2(problem.problem, A -> PNLazyMatrices.schur_complement(A, Krylov.minres, PNLazyMatrices.cache ∘ LinearAlgebra.inv!));

meas1_low_pn[:] .= ((discrete_ext[1].vector*system_full)*discrete_rhs)[:]
# meas2[:] .= ((discrete_ext[2].vector*system_full)*discrete_rhs)[:]

for (i_r, m_r) in collect(enumerate([4, 6, 10, 15]))
    system_lowrank = EPMAfem.implicit_midpoint_dlr5(problem.problem, max_ranks=(p=m_r, m=m_r));
    sol = EPMAfem.IterableDiscretePNSolution(adjoint(system_lowrank), discrete_ext[1].vector; step_callback=(ϵ, ψ) -> begin
        ranks1_p[EPMAfem.plus½(ϵ), i_r] = ψ.ranks.p[]
        ranks1_m[EPMAfem.plus½(ϵ), i_r] = ψ.ranks.m[]
    end)
    meas1_lowrank[:, i_r] .= (sol*discrete_rhs)[:]
    # meas2_lowrank_nb[:, i_r] .= ((discrete_ext[2].vector*system_lowrank)*discrete_rhs)[:]

    system_lowrank = EPMAfem.implicit_midpoint_dlr5(problem.problem, max_ranks=(p=m_r, m=m_r), basis_augmentation=:mass);
    sol = EPMAfem.IterableDiscretePNSolution(adjoint(system_lowrank), discrete_ext[1].vector; step_callback=(ϵ, ψ) -> begin
        ranks1_mass_p[EPMAfem.plus½(ϵ), i_r] = ψ.ranks.p[]
        ranks1_mass_m[EPMAfem.plus½(ϵ), i_r] = ψ.ranks.m[]
    end)
    meas1_lowrank_mass[:, i_r] .= (sol*discrete_rhs)[:]
    # meas2_lowrank_nb[:, i_r] .= ((discrete_ext[2].vector*system_lowrank)*discrete_rhs)[:]
end

for (i_r, m_r) in collect(enumerate([4, 6, 10, 15]))
    system_lowrank = EPMAfem.implicit_midpoint_dlr5(problem.problem, max_ranks=(p=m_r, m=m_r), basis_augmentation=:mass_outflux);
    sol = EPMAfem.IterableDiscretePNSolution(adjoint(system_lowrank), discrete_ext[1].vector; step_callback=(ϵ, ψ) -> begin
        ranks1_mass_p[EPMAfem.plus½(ϵ), i_r] = ψ.ranks.p[]
        ranks1_mass_m[EPMAfem.plus½(ϵ), i_r] = ψ.ranks.m[]
    end)
    meas1_lowrank_massbc[:, i_r] .= (sol*discrete_rhs)[:]
    # meas2_lowrank_nb[:, i_r] .= ((discrete_ext[2].vector*system_lowrank)*discrete_rhs)[:]
end

plot(meas1, color=:black)
plot!(meas1_low_pn, color=:gray)
plot!(meas1_lowrank[:, 1], ls=:dash)
plot!(meas1_lowrank[:, 2], ls=:dash)
plot!(meas1_lowrank[:, 3], ls=:dash)
plot!(meas1_lowrank[:, 4], ls=:dash)
plot!(meas1_lowrank_mass[:, 1], ls=:dot)
plot!(meas1_lowrank_mass[:, 2], ls=:dot)
plot!(meas1_lowrank_mass[:, 3], ls=:dot)
plot!(meas1_lowrank_mass[:, 4], ls=:dot)

plot!(meas1_lowrank_massbc[:, 1], ls=:dot)
plot!(meas1_lowrank_massbc[:, 2], ls=:dot)
plot!(meas1_lowrank_massbc[:, 3], ls=:dot)
plot!(meas1_lowrank_massbc[:, 4], ls=:dot)

p1 = contourf(-500u"nm":1u"nm":500u"nm", -500u"nm":1u"nm":0u"nm",
        (x, z) -> -func1(VectorValue(NExt.dimless(z, equations.dim_basis), NExt.dimless(x, equations.dim_basis))), aspect_ratio=:equal, linewidth=0, clims=(3, 6))
plot!(80u"nm" .+ 80u"nm".*cos.(0:0.01:2π), -200u"nm" .+ 80u"nm".*sin.(0:0.01:2π), color=:lightgray, ls=:dash, label=nothing)
p2 = contourf(-500u"nm":1u"nm":500u"nm", -500u"nm":1u"nm":0u"nm",
        (x, z) -> -func1_lr(VectorValue(NExt.dimless(z, equations.dim_basis), NExt.dimless(x, equations.dim_basis))), aspect_ratio=:equal, linewidth=0, clims=(3, 6))
plot!(80u"nm" .+ 80u"nm".*cos.(0:0.01:2π), -200u"nm" .+ 80u"nm".*sin.(0:0.01:2π), color=:lightgray, ls=:dash, label=nothing)
plot(p1, p2, size=(1000, 500))

p1 = contourf(-500u"nm":1u"nm":500u"nm", -500u"nm":1u"nm":0u"nm",
        (x, z) -> -func2(VectorValue(NExt.dimless(z, equations.dim_basis), NExt.dimless(x, equations.dim_basis))), aspect_ratio=:equal, linewidth=0)
plot!(80u"nm" .+ 80u"nm".*cos.(0:0.01:2π), -200u"nm" .+ 80u"nm".*sin.(0:0.01:2π), color=:lightgray, ls=:dash, label=nothing)
p2 = contourf(-500u"nm":1u"nm":500u"nm", -500u"nm":1u"nm":0u"nm",
        (x, z) -> -func2_lr(VectorValue(NExt.dimless(z, equations.dim_basis), NExt.dimless(x, equations.dim_basis))), aspect_ratio=:equal, linewidth=0)
plot!(80u"nm" .+ 80u"nm".*cos.(0:0.01:2π), -200u"nm" .+ 80u"nm".*sin.(0:0.01:2π), color=:lightgray, ls=:dash, label=nothing)
plot(p1, p2, size=(1000, 500), clims=(-0.01, 0.1))

contourf(-500u"nm":1u"nm":500u"nm", -500u"nm":1u"nm":0u"nm",
        (x, z) -> -func2(VectorValue(NExt.dimless(z, equations.dim_basis), NExt.dimless(x, equations.dim_basis))), aspect_ratio=:equal, linewidth=0)
plot!(80u"nm" .+ 80u"nm".*cos.(0:0.01:2π), -200u"nm" .+ 80u"nm".*sin.(0:0.01:2π), color=:lightgray, ls=:dash, label=nothing)

let
    p1 = plot(meas1, label=L"P_{11}", color=:black)
    plot!(meas1_lowrank[:, 1], label=L"t=0.05", color=1)
    plot!(meas1_lowrank[:, 2], label=L"t=0.025", color=2)
    plot!(meas1_lowrank[:, 3], label=L"t=0.0125", color=3)
    plot!(meas1_lowrank[:, 3], label=L"t=0.00625", color=4)

    savefig(("figures_low_rank/k_ratios1.png"))

    p2 = plot(meas2, label=L"P_{11}", color=:black)
    plot!(meas2_lowrank[:, 1], label=L"r=3", color=1)
    plot!(meas2_lowrank[:, 2], label=L"r=5", color=2)
    plot!(meas2_lowrank[:, 3], label=L"r=10", color=3)
    plot!(meas2_lowrank[:, 3], label=L"r=20", color=4)
    savefig(("figures_low_rank/k_ratios2.png"))
end
