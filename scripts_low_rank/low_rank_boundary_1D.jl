using Revise
using EPMAfem

using EPMAfem.PNLazyMatrices
using EPMAfem.Krylov
using EPMAfem.Gridap
using LinearAlgebra
using Plots
using LaTeXStrings
using BenchmarkTools

figpath = mkpath(joinpath(dirname(@__FILE__), "figures/1D_vacuum_boundary"))
# let
    struct TransportEquations <: EPMAfem.AbstractPNEquations end

    EPMAfem.number_of_elements(::TransportEquations) = 1
    EPMAfem.number_of_scatterings(::TransportEquations) = 1
    EPMAfem.stopping_power(::TransportEquations, e, ϵ) = 1.0
    EPMAfem.absorption_coefficient(eq::TransportEquations, e, ϵ) = 0.0 # e == 1 ? 0.0 : 0.0
    EPMAfem.scattering_coefficient(eq::TransportEquations, e, i, ϵ) = 0.0
    EPMAfem.mass_concentrations(::TransportEquations, e, x) = 1.0
    EPMAfem.scattering_kernel(::TransportEquations, e, i) = μ -> 0.0

    equations = TransportEquations()
    space_model = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((-1, 0), (200,)))
    direction_model = EPMAfem.SphericalHarmonicsModels.EOSphericalHarmonicsModel(47, 1)
    nϵ = 200
    model = EPMAfem.DiscretePNModel(space_model, range(0, 2, length=nϵ), direction_model)
    problem = EPMAfem.discretize_problem(equations, model, EPMAfem.cpu())
    excitation = EPMAfem.pn_excitation([(x=0.0, y=0.0)], [1.7], [VectorValue(-1.0, 0.0, 0.0)], beam_energy_σ=0.08, beam_direction_κ=10.0);
    discrete_rhs = EPMAfem.discretize_rhs(excitation, model, EPMAfem.cpu())[1];
    discrete_rhs.bΩp .*= 2 # this should be fixed in the discretization...

    system_full = EPMAfem.implicit_midpoint2(problem, Krylov.minres);
    system_lowrank = EPMAfem.implicit_midpoint_dlr5(problem; max_ranks=(p=2, m=2));
    system_lowrank5 = EPMAfem.implicit_midpoint_dlr5(problem; max_ranks=(p=5, m=5));
    sol = system_full * discrete_rhs;
    sol_lowrank = system_lowrank * discrete_rhs;
    sol_lowrank5 = system_lowrank5 * discrete_rhs;

    # full PN (lower N = 7)
    direction_model2 = EPMAfem.SphericalHarmonicsModels.EOSphericalHarmonicsModel(7, 1)
    model2 = EPMAfem.DiscretePNModel(space_model, range(0, 2, length=nϵ), direction_model2)
    problem2 = EPMAfem.discretize_problem(equations, model2, EPMAfem.cpu())
    discrete_rhs2 = EPMAfem.discretize_rhs(excitation, model2, EPMAfem.cpu())[1];
    discrete_rhs2.bΩp .*= 2 # this should be fixed in the discretization...
    system_full2 = EPMAfem.implicit_midpoint2(problem2, Krylov.minres);
    sol_full2 = system_full2 * discrete_rhs2;

    # full PN (lower N = 21)
    direction_model3 = EPMAfem.SphericalHarmonicsModels.EOSphericalHarmonicsModel(21, 1)
    model3 = EPMAfem.DiscretePNModel(space_model, range(0, 2, length=nϵ), direction_model3)
    problem3 = EPMAfem.discretize_problem(equations, model3, EPMAfem.cpu())
    discrete_rhs3 = EPMAfem.discretize_rhs(excitation, model3, EPMAfem.cpu())[1];
    discrete_rhs3.bΩp .*= 2 # this should be fixed in the discretization...
    system_full3 = EPMAfem.implicit_midpoint2(problem3, Krylov.minres);
    sol_full3 = system_full3 * discrete_rhs3;

    # full PN (high_res)
    direction_model4 = EPMAfem.SphericalHarmonicsModels.EOSphericalHarmonicsModel(47, 1)
    space_model4 = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((-1, 0), (600,)))
    model4 = EPMAfem.DiscretePNModel(space_model4, range(0, 2, length=3*nϵ), direction_model4)
    problem4 = EPMAfem.discretize_problem(equations, model4, EPMAfem.cpu())
    discrete_rhs4 = EPMAfem.discretize_rhs(excitation, model4, EPMAfem.cpu())[1];
    discrete_rhs4.bΩp .*= 2 # this should be fixed in the discretization...
    system_full4 = EPMAfem.implicit_midpoint2(problem4, Krylov.minres);
    sol_full4 = system_full4 * discrete_rhs4;

    function exact_solution(ϵ, x, Ω)
        return EPMAfem.beam_direction_distribution(excitation, 1, Ω) * EPMAfem.beam_energy_distribution(excitation, 1, ϵ - x/abs(Ω[1]))
    end

    function exact_solution_average(ϵ, x)
        return EPMAfem.SphericalHarmonicsModels.lebedev_quadrature_max()(Ω -> exact_solution(ϵ, x, Ω))
    end

    probe0 = EPMAfem.PNProbe(model, EPMAfem.cpu(), Ω=Ω -> 1.0, ϵ=1.7)
    probe20 = EPMAfem.PNProbe(model2, EPMAfem.cpu(), Ω=Ω -> 1.0, ϵ=1.7)
    func_full_0 = EPMAfem.interpolable(probe0, sol)
    func_full2_0 = EPMAfem.interpolable(probe20, sol_full2)
    func_lowrank_0 = EPMAfem.interpolable(probe0, sol_lowrank)
    func_lowrank5_0 = EPMAfem.interpolable(probe0, sol_lowrank5)

    probe1 = EPMAfem.PNProbe(model, EPMAfem.cpu(), Ω=Ω -> 1.0, ϵ=1.3)
    probe21 = EPMAfem.PNProbe(model2, EPMAfem.cpu(), Ω=Ω -> 1.0, ϵ=1.3)
    func_full_1 = EPMAfem.interpolable(probe1, sol)
    func_full2_1 = EPMAfem.interpolable(probe21, sol_full2)
    func_lowrank_1 = EPMAfem.interpolable(probe1, sol_lowrank)
    func_lowrank5_1 = EPMAfem.interpolable(probe1, sol_lowrank5)

    probe2 = EPMAfem.PNProbe(model, EPMAfem.cpu(), Ω=Ω -> 1.0, ϵ=0.8)
    probe22 = EPMAfem.PNProbe(model2, EPMAfem.cpu(), Ω=Ω -> 1.0, ϵ=0.8)
    func_full_2 = EPMAfem.interpolable(probe2, sol)
    func_full2_2 = EPMAfem.interpolable(probe22, sol_full2)
    func_lowrank_2 = EPMAfem.interpolable(probe2, sol_lowrank)
    func_lowrank5_2 = EPMAfem.interpolable(probe2, sol_lowrank5)

    plot(-1:0.01:0, x -> exact_solution_average(1.7, x), color=:black, ls=:solid, label=L"\textrm{exact}", linewidth=2)
    plot!(size=(400, 300), dpi=1000, fontfamily="Computer Modern", right_margin=2Plots.mm, xflip=true, legend=:topright)

    plot!(-1:0.01:0, x -> func_full2_0(VectorValue(x)), color=1, ls=:solid, label=L"P_{7}")
    plot!(-1:0.01:0, x -> func_full_0(VectorValue(x)), color=2, ls=:solid, label=L"P_{47}")
    plot!(-1:0.01:0, x -> func_lowrank_0(VectorValue(x)), color=4, ls=:solid, label=L"P_{47}, r=2")
    plot!(-1:0.01:0, x -> func_lowrank5_0(VectorValue(x)), color=3, ls=:solid, label=L"P_{47}, r=5")
    annotate!(-0.12, 0.9, (L"t=0.3", 10))

    plot!(-1:0.01:0, x -> exact_solution_average(1.3, x), color=:black, ls=:dash, label=nothing, linewidth=2)
    plot!(-1:0.01:0, x -> func_full2_1(VectorValue(x)), color=1, label=nothing, ls=:dash)
    plot!(-1:0.01:0, x -> func_full_1(VectorValue(x)), color=2, label=nothing, ls=:solid)
    plot!(-1:0.01:0, x -> func_lowrank_1(VectorValue(x)), color=4, label=nothing, ls=:dash)
    plot!(-1:0.01:0, x -> func_lowrank5_1(VectorValue(x)), color=3, label=nothing, ls=:dash)
    annotate!(-0.49, 0.8, (L"t=0.7", 10))

    plot!(-1:0.01:0, x -> exact_solution_average(0.8, x), color=:black, ls=:dashdot, label=nothing, linewidth=2)
    plot!(-1:0.01:0, x -> func_full2_2(VectorValue(x)), color=1, label=nothing, ls=:dashdot)
    plot!(-1:0.01:0, x -> func_full_2(VectorValue(x)), color=2, label=nothing, ls=:solid)
    plot!(-1:0.01:0, x -> func_lowrank_2(VectorValue(x)), color=4, label=nothing, ls=:dashdot)
    plot!(-1:0.01:0, x -> func_lowrank5_2(VectorValue(x)), color=3, label=nothing, ls=:dashdot)
    annotate!(-0.78, 0.25, (L"t=1.2", 10))
    xlabel!(L"x")
    ylabel!(L"\psi_0")
    ylims!(-0.1, 1.2)
    savefig(joinpath(figpath, "mass_over_time_energy.png"))

    energy_full = zeros(length(sol))
    energy_full2 = zeros(length(sol))
    energy_full3 = zeros(length(sol))
    energy_full4 = zeros(length(sol_full4))
    energy_lowrank = zeros(length(sol))
    energy_lowrank5 = zeros(length(sol))
    mass_full = zeros(length(sol))
    mass_full2 = zeros(length(sol))
    mass_full3 = zeros(length(sol))
    mass_full4 = zeros(length(sol_full4))
    mass_lowrank = zeros(length(sol))
    mass_lowrank5 = zeros(length(sol))
    Mp = EPMAfem.SpaceModels.assemble_bilinear(EPMAfem.SpaceModels.∫R_uv, EPMAfem.space_model(model), EPMAfem.SpaceModels.even(EPMAfem.space_model(model)), EPMAfem.SpaceModels.even(EPMAfem.space_model(model)))
    Mm = EPMAfem.SpaceModels.assemble_bilinear(EPMAfem.SpaceModels.∫R_uv, EPMAfem.space_model(model), EPMAfem.SpaceModels.odd(EPMAfem.space_model(model)), EPMAfem.SpaceModels.odd(EPMAfem.space_model(model)))
    for (i, ((_, ψfull), (_, ψ_full2), (_, ψ_full3), (_, ψ_lowrank), (_, ψ_lowrank5))) in enumerate(zip(sol, sol_full2, sol_full3, sol_lowrank, sol_lowrank5))
        ψfullp, ψfullm = EPMAfem.pmview(ψfull, model)
        ψfull2p, ψfull2m = EPMAfem.pmview(ψ_full2, model2)
        ψfull3p, ψfull3m = EPMAfem.pmview(ψ_full3, model3)
        ψlowrankp, ψlowrankm = EPMAfem.pmview(ψ_lowrank, model)
        ψlowrank5p, ψlowrank5m = EPMAfem.pmview(ψ_lowrank5, model)

        energy_full[i] = dot(ψfullp, Mp*ψfullp) + dot(ψfullm, Mm*ψfullm)
        energy_full2[i] = dot(ψfull2p, Mp*ψfull2p) + dot(ψfull2m, Mm*ψfull2m)
        energy_full3[i] = dot(ψfull3p, Mp*ψfull3p) + dot(ψfull3m, Mm*ψfull3m)
        energy_lowrank[i] = dot(ψlowrankp, Mp*ψlowrankp) + dot(ψlowrankm, Mm*ψlowrankm)
        energy_lowrank5[i] = dot(ψlowrank5p, Mp*ψlowrank5p) + dot(ψlowrank5m, Mm*ψlowrank5m)

        Ωp, Ωm = EPMAfem.SphericalHarmonicsModels.eval_basis(EPMAfem.direction_model(model), Ω -> 1.0/(4π))
        Ω2p, Ω2m = EPMAfem.SphericalHarmonicsModels.eval_basis(EPMAfem.direction_model(model2), Ω -> 1.0/(4π))
        Ω3p, Ω3m = EPMAfem.SphericalHarmonicsModels.eval_basis(EPMAfem.direction_model(model3), Ω -> 1.0/(4π))
        xp, xm = EPMAfem.SpaceModels.eval_basis(EPMAfem.space_model(model), x -> 1.0)
        mass_full[i] = dot(xp, ψfullp, Ωp) + dot(xm, ψfullm, Ωm) 
        mass_full2[i] = dot(xp, ψfull2p, Ω2p) + dot(xm, ψfull2m, Ω2m)
        mass_full3[i] = dot(xp, ψfull3p, Ω3p) + dot(xm, ψfull3m, Ω3m)
        mass_lowrank[i] = dot(xp, ψlowrankp, Ωp) + dot(xm, ψlowrankm, Ωm) 
        mass_lowrank5[i] = dot(xp, ψlowrank5p, Ωp) + dot(xm, ψlowrank5m, Ωm) 
    end

    Mp4 = EPMAfem.SpaceModels.assemble_bilinear(EPMAfem.SpaceModels.∫R_uv, EPMAfem.space_model(model4), EPMAfem.SpaceModels.even(EPMAfem.space_model(model4)), EPMAfem.SpaceModels.even(EPMAfem.space_model(model4)))
    Mm4 = EPMAfem.SpaceModels.assemble_bilinear(EPMAfem.SpaceModels.∫R_uv, EPMAfem.space_model(model4), EPMAfem.SpaceModels.odd(EPMAfem.space_model(model4)), EPMAfem.SpaceModels.odd(EPMAfem.space_model(model4)))
    for (i, (_, ψ_full4)) in enumerate(sol_full4)
        ψfull4p, ψfull4m = EPMAfem.pmview(ψ_full4, model4)
        energy_full4[i] = dot(ψfull4p, Mp4*ψfull4p) + dot(ψfull4m, Mm4*ψfull4m)
        Ω4p, Ω4m = EPMAfem.SphericalHarmonicsModels.eval_basis(EPMAfem.direction_model(model4), Ω -> 1.0/(4π))
        xp4, xm4 = EPMAfem.SpaceModels.eval_basis(EPMAfem.space_model(model4), x -> 1.0)
        mass_full4[i] = dot(xp4, ψfull4p, Ω4p) + dot(xm4, ψfull4m, Ω4m)
    end

    # # energy = EPMAfem.hquadrature(ϵ -> EPMAfem.SphericalHarmonicsModels.lebedev_quadrature_max()(Ω -> -Ω[1]*(EPMAfem.beam_energy_distribution(excitation, 1, ϵ)*EPMAfem.beam_direction_distribution(excitation, 1, Ω))^2), -1, 1)[1]

    # dir_mod = EPMAfem.SphericalHarmonicsModels.EOSphericalHarmonicsModel(47, 1)

    # EPMAfem.SphericalHarmonicsModels.lebedev_quadrature_max()(Ω -> abs(Ω[1])/(4π)*(EPMAfem.beam_direction_distribution(excitation, 1, Ω))^2)
    # dot(discrete_rhs.bΩp, discrete_rhs.bΩp)/4π/2

    # # Bbp = EPMAfem.SphericalHarmonicsModels.assemble_bilinear(EPMAfem.SphericalHarmonicsModels.∫S²_μuv(Ω -> (Ω[1] > 0)*Ω[1]), dir_mod, EPMAfem.SphericalHarmonicsModels.even(dir_mod), EPMAfem.SphericalHarmonicsModels.even(dir_mod))    
    # # dot(discrete_rhs.bΩp, Bbp*discrete_rhs.bΩp)/4π

    # bb = EPMAfem.SphericalHarmonicsModels.assemble_linear(EPMAfem.SphericalHarmonicsModels.∫S²_hv(Ω -> (Ω[1] < 0)*EPMAfem.beam_direction_distribution(excitation, 1, Ω)), dir_mod, EPMAfem.SphericalHarmonicsModels.even(dir_mod))
    # dot(bb, discrete_rhs.bΩp)/4π
    
    # EPMAfem.hquadrature(ϵ -> (EPMAfem.beam_energy_distribution(excitation, 1, ϵ))^2, 0, 2)[1]
    # ϵ_max = (sum(discrete_rhs.bϵ[2:end-1].^2)) * step(EPMAfem.energy_model(model))


    begin
        energy_bound = EPMAfem.SphericalHarmonicsModels.lebedev_quadrature_max()(Ω -> abs(Ω[1])/(4π)*(EPMAfem.beam_direction_distribution(excitation, 1, Ω))^2)*EPMAfem.hquadrature(ϵ -> (EPMAfem.beam_energy_distribution(excitation, 1, ϵ))^2, 0, 2)[1]
        t = 2.0 .- EPMAfem.energy_model(model) |> reverse
        t4 = 2.0 .- EPMAfem.energy_model(model4) |> reverse

        hline([4π*energy_bound], color=:gray, ls=:dash, label=L"\mathcal{E}_{\textrm{max}}")
        xlims!(0, 2)
        plot!(t, energy_full2, label=L"P_{7}", xflip=false, color=1)
        plot!(t, energy_full, label=L"P_{47}", color=2)
        plot!(t, energy_lowrank, label=L"P_{47}, r=2", color=4, ls=:dash)
        plot!(t, energy_lowrank5, label=L"P_{47}, r=5", ls=:dash, color=3)
        plot!(t4, energy_full4, label=L"P_{47}, (\textrm{highres})", ls=:dot, color=2)
        # plot!(t, energy_full3, label=L"P_{21}", color=6, ls=:dot)

        zoom_range = 120:-1:45
        zoom_datarange = (0.106, 0.108)
        plot!([t[first(zoom_range)], t[last(zoom_range)]], [zoom_datarange[1], zoom_datarange[1]], color=:black, label=nothing, linewidth=1)
        plot!([t[first(zoom_range)], t[last(zoom_range)]], [zoom_datarange[2], zoom_datarange[2]], color=:black, label=nothing, linewidth=1)
        plot!([t[first(zoom_range)], t[first(zoom_range)]], [zoom_datarange[1], zoom_datarange[2]], color=:black, label=nothing, linewidth=1)
        plot!([t[last(zoom_range)], t[last(zoom_range)]], [zoom_datarange[1], zoom_datarange[2]], color=:black, label=nothing, linewidth=1)
        xlabel!(L"t")
        ylabel!(L"\mathcal{E}(t) \times 4\pi")

        p_i = plot!([t[zoom_range] t[zoom_range] t[zoom_range] t[zoom_range] t[zoom_range] t4[3zoom_range]],
                    [fill(4π*energy_bound, length(zoom_range)) energy_full2[zoom_range] energy_full[zoom_range] energy_lowrank[zoom_range] energy_lowrank5[zoom_range] energy_full4[3zoom_range]], 
                    color=[:gray 1 2 4 3 2],
                    ls=[:dash :solid :solid :dash :dash :dot],
                    label=nothing,
                    inset=(1, bbox(0.38, 0.2, 0.5, 0.5, :bottom, :right)), subplot=2, ylims=zoom_datarange, xflip=false, framestyle=:box, tickfontsize=5)
        plot!(size=(400, 300), dpi=1000, fontfamily="Computer Modern", right_margin=2Plots.mm, legend=:topright)

        savefig(joinpath(figpath, "energy.png"))
    end


    begin
        mass_bound = EPMAfem.SphericalHarmonicsModels.lebedev_quadrature_max()(Ω -> abs(Ω[1])/(4π)*EPMAfem.beam_direction_distribution(excitation, 1, Ω))*EPMAfem.hquadrature(ϵ -> EPMAfem.beam_energy_distribution(excitation, 1, ϵ), 0, 2)[1]

        t = 2.0 .- EPMAfem.energy_model(model) |> reverse

        hline([4π*mass_bound], color=:gray, ls=:dash, label=L"M_{\textrm{max}}")
        xlims!(0, 2)
        plot!(t, 4π*mass_full2, label=L"P_{7}", xflip=false, color=1)
        plot!(t, 4π*mass_full, label=L"P_{47}", color=2)
        plot!(t, 4π*mass_lowrank, label=L"P_{47}, r=2", color=4, ls=:dash)
        plot!(t, 4π*mass_lowrank5, label=L"P_{47}, r=5", ls=:dash, color=3)
        plot!(t, 4π*mass_full3, label=L"P_{21}", color=6, ls=:dot)

        zoom_range = 120:-1:45
        zoom_datarange = (4π*0.0141, 4π*0.0145)
        plot!([t[first(zoom_range)], t[last(zoom_range)]], [zoom_datarange[1], zoom_datarange[1]], color=:black, label=nothing)
        plot!([t[first(zoom_range)], t[last(zoom_range)]], [zoom_datarange[2], zoom_datarange[2]], color=:black, label=nothing)
        plot!([t[first(zoom_range)], t[first(zoom_range)]], [zoom_datarange[1], zoom_datarange[2]], color=:black, label=nothing)
        plot!([t[last(zoom_range)], t[last(zoom_range)]], [zoom_datarange[1], zoom_datarange[2]], color=:black, label=nothing)
        xlabel!(L"t")
        ylabel!(L"M(t) \times 4\pi")

        p_i = plot!(t[zoom_range], [fill(4π*mass_bound, length(zoom_range)) 4π*mass_full2[zoom_range] 4π*mass_full[zoom_range] 4π*mass_lowrank[zoom_range] 4π*mass_lowrank5[zoom_range] 4π*mass_full3[zoom_range]], color=[:gray 1 2 4 3 6], ls=[:dash :solid :solid :dash :dash :dot], label=nothing,
            inset=(1, bbox(0.38, 0.2, 0.5, 0.5, :bottom, :right)), subplot=2, ylims=zoom_datarange, xflip=false, framestyle=:box, tickfontsize=5)
        plot!(size=(400, 300), dpi=1000, fontfamily="Computer Modern", right_margin=2Plots.mm, legend=:topright)

        savefig(joinpath(figpath, "mass.png"))
    end

    ## TIMINGS
    using BenchmarkTools
    b_sol = @benchmark for (ϵ, ψ) in $(sol) end # 18.495 s
    b_sol2 = @benchmark for (ϵ, ψ) in $(sol_full2) end # 1.006 s
    b_sol3 = @benchmark for (ϵ, ψ) in $(sol_full3) end # 7.263 s
    b_lr = @benchmark for (ϵ, ψ) in $(sol_lowrank) end # 709.695 ms
    b_lr2 = @benchmark for (ϵ, ψ) in $(sol_lowrank5) end # 1.614 s

    @show b_sol
    @show EPMAfem.n_basis(model)
    @show b_sol2
    @show EPMAfem.n_basis(model2)
    @show b_sol3
    @show EPMAfem.n_basis(model3)
    @show b_lr
    @show system_lowrank.max_rank
    @show b_lr2
    @show system_lowrank5.max_rank
# end
