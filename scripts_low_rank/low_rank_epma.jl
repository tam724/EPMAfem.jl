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

figpath = mkpath(joinpath(dirname(@__FILE__), "figures/2D_epma"))

equations = NExt.epma_equations(
    [n"Al", n"Cr"],
    [NExt.EPMADetector(n"Al K-L2", VectorValue(1.0, 0.0, 0.0)), NExt.EPMADetector(n"Cr K-L2", VectorValue(1.0, 0.0, 0.0))],
    range(50u"eV", 20u"keV", length=50), 27)
timings = []
plotdata = Dict()
Ns = [1, 5, 11, 21, 27]
ranks = [3, 5, 10]

for N in Ns
    model = NExt.epma_model(equations, (-2200u"nm", 0.0u"nm", -2200u"nm", 2200u"nm"), (150, 300), N)
    problem = EPMAfem.discretize_problem(equations, model, EPMAfem.cuda(), updatable=true)

    excitation = EPMAfem.pn_excitation([(x=0.0, y=0.0)], [NExt.dimless(15u"keV", equations.dim_basis)], [VectorValue(-1.0, 0.0, 0.0)], beam_position_σ=0.02, beam_energy_σ=0.05, beam_direction_κ=80)
    discrete_rhs = EPMAfem.discretize_rhs(excitation, model, EPMAfem.cuda())[1]
    function mass_concentrations(e, x_)
        z = NExt.dimful(x_[1], u"nm", equations.dim_basis)
        x = NExt.dimful(x_[2], u"nm", equations.dim_basis)
        # if x > 0u"nm"
        #     return e == 1 ? NExt.dimless(n"Al".density, equations.dim_basis) : 0.0
        # else
        #     return e == 1 ? 0.0 : NExt.dimless(n"Au".density, equations.dim_basis)
        # end
        if (x - 80u"nm")^2 + (z - (-200u"nm"))^2 < (80u"nm")^2
            return e == 1 ? 0.0 : NExt.dimless(n"Cr".density, equations.dim_basis)
        else
            return e == 1 ? NExt.dimless(n"Al".density, equations.dim_basis) : 0.0
        end
    end
    ρs = EPMAfem.discretize_mass_concentrations([x -> mass_concentrations(e, x) for e in 1:EPMAfem.number_of_elements(equations)], model)
    EPMAfem.update_problem!(problem, ρs)

    probe = EPMAfem.PNProbe(model, EPMAfem.cuda(), Ω=Ω -> 1.0, ϵ=ϵ -> 1.0)
    system_full = EPMAfem.implicit_midpoint2(problem.problem, A -> PNLazyMatrices.schur_complement(A, Krylov.minres, PNLazyMatrices.cache ∘ LinearAlgebra.inv!))
    @show N
    # run twice
    EPMAfem.interpolable(probe, system_full * discrete_rhs)
    time_full = @elapsed func_full = EPMAfem.interpolable(probe, system_full * discrete_rhs)
    push!(timings, ((N, Inf), time_full))
    plotdata[(N, Inf)] = func_full

    for rank in ranks
        if 2 * rank > EPMAfem.n_basis(problem.problem).nΩ.p || rank > EPMAfem.n_basis(problem.problem).nΩ.p
            continue
        end
        system_lowrank = EPMAfem.implicit_midpoint_dlr5(problem.problem, max_ranks=(p=rank, m=rank))
        EPMAfem.interpolable(probe, system_lowrank * discrete_rhs)
        time_lowrank = @elapsed func_lowrank = EPMAfem.interpolable(probe, system_lowrank * discrete_rhs)
        @show N, rank
        push!(timings, ((N, rank), time_lowrank))
        plotdata[(N, rank)] = func_lowrank
    end
end

using Serialization
serialize(joinpath(figpath, "plotdata.jls"), plotdata)
serialize(joinpath(figpath, "timings.jls"), timings)
plotdata = deserialize(joinpath(figpath, "plotdata.jls"))
timings = deserialize(joinpath(figpath, "timings.jls"))

for N in Ns
    contourf(
        NExt.dimless.(-1500u"nm":1u"nm":1500u"nm", equations.dim_basis),
        NExt.dimless.(-1500u"nm":1u"nm":0u"nm", equations.dim_basis),
        (x, z) -> plotdata[(N, Inf)](Gridap.Point(z, x)), aspect_ratio=:equal, levels=12, clims=(0.0, 0.08))
        plot!(size=(400, 300), dpi=1000, fontfamily="Computer Modern", right_margin=2Plots.mm)
        savefig(joinpath(figpath, "N$(N)_rankinf.png"))
    for rank in ranks
        if haskey(plotdata, (N, rank))
            contourf(
                NExt.dimless.(-1500u"nm":1u"nm":1500u"nm", equations.dim_basis),
                NExt.dimless.(-1500u"nm":1u"nm":0u"nm", equations.dim_basis),
                (x, z) -> plotdata[(N, rank)](Gridap.Point(z, x)), aspect_ratio=:equal, levels=12, clims=(0.0, 0.08))
            plot!(size=(400, 300), dpi=1000, fontfamily="Computer Modern", right_margin=2Plots.mm)
            savefig(joinpath(figpath, "N$(N)_rank$(rank).png"))
        end
    end
end


function clampabs(x, ϵ=1e-15)
    return x / 0.08
    # x_ = abs(x)
    # if x_ < ϵ
    #     return ϵ
    # end
    # return x_
end

for N in Ns
    @show N
    heatmap(
        NExt.dimless.(-1500u"nm":1u"nm":1500u"nm", equations.dim_basis),
        NExt.dimless.(-1500u"nm":1u"nm":0u"nm", equations.dim_basis),
        (x, z) -> clampabs(plotdata[(N, Inf)](Gridap.Point(z, x)) - plotdata[(27, Inf)](Gridap.Point(z, x))), aspect_ratio=:equal, levels=12, clims=(-0.125, 0.125), cmap=:bluesreds)
        plot!(size=(400, 300), dpi=1000, fontfamily="Computer Modern", right_margin=2Plots.mm)
        savefig(joinpath(figpath, "diff_N$(N)_rankinf.png"))
    for rank in ranks
        @show N, rank
        if haskey(plotdata, (N, rank))
            heatmap(
                NExt.dimless.(-1500u"nm":1u"nm":1500u"nm", equations.dim_basis),
                NExt.dimless.(-1500u"nm":1u"nm":0u"nm", equations.dim_basis),
                (x, z) -> clampabs(plotdata[(N, rank)](Gridap.Point(z, x)) - plotdata[(27, Inf)](Gridap.Point(z, x))), aspect_ratio=:equal, levels=12, clims=(-0.125, 0.125), cmap=:bluesreds)
            plot!(size=(400, 300), dpi=1000, fontfamily="Computer Modern", right_margin=2Plots.mm)
            savefig(joinpath(figpath, "diff_N$(N)_rank$(rank).png"))
        end
    end
end

plot(range(NExt.dimless.((-1500u"nm", 1500u"nm"), equations.dim_basis)..., length=200), x -> EPMAfem.beam_space_distribution(excitation, 1, VectorValue(0.0, x, 0.0)))
plot(range(NExt.dimless.((50u"eV", 20u"keV"), equations.dim_basis)..., length=50), ϵ -> EPMAfem.beam_energy_distribution(excitation, 1, ϵ))

time_map = zeros(Union{Float64,Missing}, 4, 5)
for (i, N) in enumerate([1, 5, 11, 21, 27])
    for (j, r) in enumerate([Inf, 10, 5, 3])
        idx = findfirst((((N_, r_), _),) -> (N_ == N && r_ == r), timings)
        if !isnothing(idx)
            time_map[j, i] = timings[idx][2]
        else
            time_map[j, i] = missing
        end
    end
end

begin
    heatmap(log.(10, time_map), yflip=true, colorbar=false, cmap=cgrad([:green, :white, :red]))
    for (i, N) in enumerate([1, 5, 11, 21, 27])
        for (j, r) in enumerate([Inf, 10, 5, 3])
            if !ismissing(time_map[j, i])
                annotate!(i, j, Plots.text("$(round(time_map[j, i], digits=2))s", "Computer Modern", 10), :black)
            end
        end
    end
    plot!(xticks=(1:5, [L"P_{1}", L"P_{5}", L"P_{11}", L"P_{21}", L"P_{27}"]))
    plot!(yticks=(1:4, [L"r=\infty", L"r=10", L"r=5", L"r=3"]))
    plot!(size=(400, 300), dpi=1000, fontfamily="Computer Modern", right_margin=2Plots.mm)

    savefig(joinpath(figpath, "timings_2d.png"))
end

plot!()

##### INTENSITIES
# space_model = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((-1, 0, -1, 1), (100, 100)))
# direction_model = EPMAfem.SphericalHarmonicsModels.EOSphericalHarmonicsModel(3, 2)
equations = NExt.epma_equations(
    [n"Al", n"Cr"],
    [NExt.EPMADetector(n"Al K-L2", VectorValue(1.0, 0.0, 0.0)), NExt.EPMADetector(n"Cr K-L2", VectorValue(1.0, 0.0, 0.0))],
    range(50u"eV", 20u"keV", length=50), 27)

meas1 = zeros(100)
meas1_lowrank = zeros(100, 4)
meas2 = zeros(100)
meas2_lowrank = zeros(100, 4)

for N in [11] # [1, 5, 11, 21, 27]
    model = NExt.epma_model(equations, (-2200u"nm", 0.0u"nm", -2200u"nm", 2200u"nm"), (150, 300), N)
    problem = EPMAfem.discretize_problem(equations, model, EPMAfem.cuda(), updatable=true)

    excitation = EPMAfem.pn_excitation([(x=NExt.dimless(x_, equations.dim_basis), y=0.0) for x_ in range(-500u"nm", 500u"nm", 100)], [NExt.dimless(15u"keV", equations.dim_basis)], [VectorValue(-1.0, 0.0, 0.0)], beam_position_σ=0.02, beam_energy_σ=0.05)
    discrete_rhs = EPMAfem.discretize_rhs(excitation, model, EPMAfem.cuda())
    discrete_ext = NExt.discretize_detectors(equations, model, EPMAfem.cuda(), absorption=false)
    discrete_ext[1].vector.bϵ .*= 1e15 # (normalize)
    discrete_ext[2].vector.bϵ .*= 1e15
    function mass_concentrations(e, x_)
        z = NExt.dimful(x_[1], u"nm", equations.dim_basis)
        x = NExt.dimful(x_[2], u"nm", equations.dim_basis)
        # if x > 0u"nm"
        #     return e == 1 ? NExt.dimless(n"Al".density, equations.dim_basis) : 0.0
        # else
        #     return e == 1 ? 0.0 : NExt.dimless(n"Au".density, equations.dim_basis)
        # end
        if (x - 80u"nm")^2 + (z - (-200u"nm"))^2 < (80u"nm")^2
            return e == 1 ? 0.0 : NExt.dimless(n"Cr".density, equations.dim_basis)
        else
            return e == 1 ? NExt.dimless(n"Al".density, equations.dim_basis) : 0.0
        end
    end
    ρs = EPMAfem.discretize_mass_concentrations([x -> mass_concentrations(e, x) for e in 1:EPMAfem.number_of_elements(equations)], model)
    # heatmap(reshape(ρs[1, :], 150, 300), aspect_ratio=:equal)
    # heatmap(reshape(ρs[2, :], 150, 300), aspect_ratio=:equal)

    EPMAfem.update_problem!(problem, ρs)
    EPMAfem.update_vector!(discrete_ext[1], ρs)
    EPMAfem.update_vector!(discrete_ext[2], ρs)

    system_full = EPMAfem.implicit_midpoint2(problem.problem, A -> PNLazyMatrices.schur_complement(A, Krylov.minres, PNLazyMatrices.cache ∘ LinearAlgebra.inv!))

    meas1[:] .= ((discrete_ext[1].vector*system_full)*discrete_rhs)[:]
    meas2[:] .= ((discrete_ext[2].vector*system_full)*discrete_rhs)[:]

    for (i_r, rank) in enumerate([3, 5, 10, 20])
        if 2 * rank > EPMAfem.n_basis(problem.problem).nΩ.p || rank > EPMAfem.n_basis(problem.problem).nΩ.p
            continue
        end
        system_lowrank = EPMAfem.implicit_midpoint_dlr5(problem.problem, max_ranks=(p=rank, m=rank))

        meas1_lowrank[:, i_r] .= ((discrete_ext[1].vector*system_lowrank)*discrete_rhs)[:]
        meas2_lowrank[:, i_r] .= ((discrete_ext[2].vector*system_lowrank)*discrete_rhs)[:]
    end
end

let
    p1 = plot(meas1, label=L"P_{11}", color=:black)
    plot!(meas1_lowrank[:, 1], label=L"r=3", color=1)
    plot!(meas1_lowrank[:, 2], label=L"r=5", color=2)
    plot!(meas1_lowrank[:, 3], label=L"r=10", color=3)
    plot!(meas1_lowrank[:, 3], label=L"r=20", color=4)
    savefig(("figures_low_rank/k_ratios1.png"))

    p2 = plot(meas2, label=L"P_{11}", color=:black)
    plot!(meas2_lowrank[:, 1], label=L"r=3", color=1)
    plot!(meas2_lowrank[:, 2], label=L"r=5", color=2)
    plot!(meas2_lowrank[:, 3], label=L"r=10", color=3)
    plot!(meas2_lowrank[:, 3], label=L"r=20", color=4)
    savefig(("figures_low_rank/k_ratios2.png"))
end
