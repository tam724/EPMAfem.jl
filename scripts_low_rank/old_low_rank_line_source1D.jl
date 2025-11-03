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

figpath = mkpath(joinpath(dirname(@__FILE__), "figures/1D_vacuum_linesource"))

struct PlaneSourceEquations{S} <: EPMAfem.AbstractPNEquations end
EPMAfem.number_of_elements(::PlaneSourceEquations) = 1
EPMAfem.number_of_scatterings(::PlaneSourceEquations) = 1
EPMAfem.stopping_power(::PlaneSourceEquations, e, ϵ) = 1.0
EPMAfem.absorption_coefficient(eq::PlaneSourceEquations, e, ϵ) = 0.0 #1.0
EPMAfem.scattering_coefficient(eq::PlaneSourceEquations, e, i, ϵ) = 0.0 #1.0
EPMAfem.mass_concentrations(::PlaneSourceEquations, e, x) = 1.0

EPMAfem.scattering_kernel(::PlaneSourceEquations{Inf}, e, i) = μ -> 1/(4π)
@generated μ₀(::PlaneSourceEquations{T}) where T = return :( $ (2π*hquadrature(μ -> exp(-T*(μ-1)^2), -1, 1)[1]))
EPMAfem.scattering_kernel(eq::PlaneSourceEquations{T}, e, i) where T = μ -> exp(-T*(μ-1)^2)/μ₀(eq)

energy_model = 0:0.01:1.0

plotdata = Dict()
Ts = [Inf] # [Inf, 1.0, 10.0, 100.0]
Ns = [3, 5, 15, 21] # [3, 7, 15, 21]
for T in Ts
    for N in Ns
        plotdata[(T, N)] = Dict()
        equations = PlaneSourceEquations{T}()
        space_model = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((-1.5, 1.5), (80)))
        direction_model = EPMAfem.SphericalHarmonicsModels.EOSphericalHarmonicsModel(N, 1)
        model = EPMAfem.DiscretePNModel(space_model, energy_model, direction_model)
        problem = EPMAfem.discretize_problem(equations, model, EPMAfem.cpu())

        @show N, EPMAfem.n_basis(problem)

        # source / boundary condition (here: zero)
        source = EPMAfem.Rank1DiscretePNVector(false, model, EPMAfem.cpu(), zeros(EPMAfem.n_basis(model).nϵ), zeros(EPMAfem.n_basis(model).nx.p), zeros(EPMAfem.n_basis(model).nΩ.p))

        # initial condition
        Mp = EPMAfem.SpaceModels.assemble_bilinear(EPMAfem.SpaceModels.∫R_uv, EPMAfem.space_model(model), EPMAfem.SpaceModels.even(EPMAfem.space_model(model)), EPMAfem.SpaceModels.even(EPMAfem.space_model(model))) |> EPMAfem.architecture(problem)
        Mm = EPMAfem.SpaceModels.assemble_bilinear(EPMAfem.SpaceModels.∫R_uv, EPMAfem.space_model(model), EPMAfem.SpaceModels.odd(EPMAfem.space_model(model)), EPMAfem.SpaceModels.odd(EPMAfem.space_model(model))) |> EPMAfem.architecture(problem)
        
        nb = EPMAfem.n_basis(model)
        basis_augmentation = (p=(U = EPMAfem.allocate_mat(EPMAfem.cpu(), nb.nx.p, 0),
                                 V = EPMAfem.allocate_mat(EPMAfem.cpu(), nb.nΩ.p, 1)),
                              m=(U = EPMAfem.allocate_mat(EPMAfem.cpu(), nb.nx.m, 0),
                                 V = EPMAfem.allocate_mat(EPMAfem.cpu(), nb.nΩ.m, 0)))
        Ωp, Ωm = EPMAfem.SphericalHarmonicsModels.eval_basis(EPMAfem.direction_model(model), Ω->1/(4π)) |> EPMAfem.architecture(problem)
        copy!(@view(basis_augmentation.p.V[:, 1]), Ωp)

        # system = EPMAfem.implicit_midpoint2(problem, A -> PNLazyMatrices.schur_complement(A, Krylov.minres, PNLazyMatrices.cache ∘ LinearAlgebra.inv!));
        system = EPMAfem.implicit_midpoint2(problem, LinearAlgebra.:\);
        system_lr3 = EPMAfem.implicit_midpoint_dlr5(problem, solver=LinearAlgebra.:\, max_ranks=(p=3, m=3));
        system_lr3_aug = EPMAfem.implicit_midpoint_dlr5(problem, solver=LinearAlgebra.:\, max_ranks=(p=3, m=3), basis_augmentation=basis_augmentation);
        system_lr6 = EPMAfem.implicit_midpoint_dlr5(problem, solver=LinearAlgebra.:\, max_ranks=(p=6, m=6));
        system_lr6_aug = EPMAfem.implicit_midpoint_dlr5(problem, solver=LinearAlgebra.:\, max_ranks=(p=6, m=6), basis_augmentation=basis_augmentation);
        system_lr20 = EPMAfem.implicit_midpoint_dlr5(problem, solver=LinearAlgebra.:\, max_ranks=(p=20, m=20));

        # σ = 0.03
        σ = 0.08
        using EPMAfem.HCubature
        init_x(x) = 1/(σ*sqrt(2π))*exp(-1/2*(x[1]-0.0)^2/σ^2)
        init_Ω(Ω) = 1.0 # pdf(VonMisesFisher([1, 0, 0], 2.0), [Ω...])
        bxp = collect(Mp) \ EPMAfem.SpaceModels.assemble_linear(EPMAfem.SpaceModels.∫R_μv(init_x), EPMAfem.space_model(model), EPMAfem.SpaceModels.even(EPMAfem.space_model(model)))
        bxm = collect(Mm) \ EPMAfem.SpaceModels.assemble_linear(EPMAfem.SpaceModels.∫R_μv(init_x), EPMAfem.space_model(model), EPMAfem.SpaceModels.odd(EPMAfem.space_model(model)))
        bΩp = EPMAfem.SphericalHarmonicsModels.assemble_linear(EPMAfem.SphericalHarmonicsModels.∫S²_hv(init_Ω), EPMAfem.direction_model(model), EPMAfem.SphericalHarmonicsModels.even(EPMAfem.direction_model(model)))
        bΩm = EPMAfem.SphericalHarmonicsModels.assemble_linear(EPMAfem.SphericalHarmonicsModels.∫S²_hv(init_Ω), EPMAfem.direction_model(model), EPMAfem.SphericalHarmonicsModels.odd(EPMAfem.direction_model(model)))
        # bΩp = EPMAfem.SphericalHarmonicsModels.assemble_linear(EPMAfem.SphericalHarmonicsModels.∫S²_hv(Ω -> 1/4π), EPMAfem.direction_model(model), EPMAfem.SphericalHarmonicsModels.even(EPMAfem.direction_model(model)))
        initial_condition = EPMAfem.allocate_solution_vector(system)
        ψ0p, ψ0m = EPMAfem.pmview(initial_condition, model)
        copy!(ψ0p, bxp .* bΩp')
        copy!(ψ0m, bxm .* bΩm')

        sol = EPMAfem.IterableDiscretePNSolution(system, source, initial_solution=initial_condition);
        sol_lr3 = EPMAfem.IterableDiscretePNSolution(system_lr3, source, initial_solution=initial_condition);
        sol_lr3_aug = EPMAfem.IterableDiscretePNSolution(system_lr3_aug, source, initial_solution=initial_condition);
        sol_lr6 = EPMAfem.IterableDiscretePNSolution(system_lr6, source, initial_solution=initial_condition);
        sol_lr6_aug = EPMAfem.IterableDiscretePNSolution(system_lr6_aug, source, initial_solution=initial_condition);
        sol_lr20 = EPMAfem.IterableDiscretePNSolution(system_lr20, source, initial_solution=initial_condition);

        #### GIF
        # Ωp, Ωm = EPMAfem.SphericalHarmonicsModels.eval_basis(EPMAfem.direction_model(model), Ω -> 1.0) |> EPMAfem.architecture(problem)
        # @gif for (ϵ, ψ) in sol
        #     ψp, ψm = EPMAfem.pmview(ψ, model)
        #     func = EPMAfem.SpaceModels.interpolable((p=collect(ψp*Ωp), m=collect(ψm*Ωm)), EPMAfem.space_model(model))
        #     plot(-1.5:0.01:1.5, x -> func(VectorValue(x)))
        #     # heatmap(-1.5:0.01:1.5, -1.5:0.01:1.5, (x, y) -> func(VectorValue(x, y)), aspect_ratio=:equal)
        # end

        probe = EPMAfem.PNProbe(model, EPMAfem.cpu(), Ω=Ω->1.0, ϵ=0.0)
        func = EPMAfem.interpolable(probe, sol)
        plotdata[(T, N)][("final", Inf)] = func
        if EPMAfem.n_basis(problem).nΩ.p > 2*6
            func_lr3 = EPMAfem.interpolable(probe, sol_lr3)
            plotdata[(T, N)][("final", 3)] = func_lr3
            func_lr3_aug = EPMAfem.interpolable(probe, sol_lr3_aug)
            plotdata[(T, N)][("final", (3, :aug))] = func_lr3
        end
        if EPMAfem.n_basis(problem).nΩ.p > 2*10
            func_lr6 = EPMAfem.interpolable(probe, sol_lr6)
            plotdata[(T, N)][("final", 6)] = func_lr6
            func_lr6_aug = EPMAfem.interpolable(probe, sol_lr6_aug)
            plotdata[(T, N)][("final", (6, :aug))] = func_lr6_aug
        end

        Ωp, Ωm = EPMAfem.SphericalHarmonicsModels.eval_basis(EPMAfem.direction_model(model), Ω->1/(4π)) |> EPMAfem.architecture(problem)
        xp, xm = EPMAfem.SpaceModels.eval_basis(EPMAfem.space_model(model), x -> 1.0) |> EPMAfem.architecture(problem)

        energy, energy_lr3, energy_lr6, energy_lr3_aug, energy_lr6_aug, energy_lr20 = zeros(length(sol)), zeros(length(sol)), zeros(length(sol)), zeros(length(sol)), zeros(length(sol)), zeros(length(sol))
        mass, mass_lr3, mass_lr6, mass_lr3_aug, mass_lr6_aug, mass_lr20 = zeros(length(sol)), zeros(length(sol)), zeros(length(sol)), zeros(length(sol)), zeros(length(sol)), zeros(length(sol))

        mass0 = dot(xp, ψ0p*Ωp) + dot(xm, ψ0m*Ωm)
        plotdata[(T, N)]["mass_init"] = mass0
        energy0 = dot(ψ0p, Mp*ψ0p) + dot(ψ0m, Mm*ψ0m)
        plotdata[(T, N)]["energy_init"] = energy0

        @show mass0, energy0

        for (i, (ϵ, ψ)) in enumerate(sol)
            ψp, ψm = EPMAfem.pmview(ψ, model)
            energy[i] = dot(ψp, Mp*ψp) + dot(ψm, Mm*ψm)
            mass[i] = dot(xp, ψp*Ωp) + dot(xm, ψm*Ωm)
            @show mass[i]
        end
        plotdata[(T, N)][("energy", Inf)] = energy
        plotdata[(T, N)][("mass", Inf)] = mass

        if EPMAfem.n_basis(problem).nΩ.p > 2*3
            for (i, (ϵ_lr3, ψ_lr3)) in enumerate(sol_lr3)
                ψp_lr3, ψm_lr3 = EPMAfem.pmview(ψ_lr3, model)
                energy_lr3[i] = dot(ψp_lr3, Mp*ψp_lr3) + dot(ψm_lr3, Mm*ψm_lr3)
                mass_lr3[i] = dot(xp, ψp_lr3*Ωp) + dot(xm, ψm_lr3*Ωm)
            end
            plotdata[(T, N)][("energy", 3)] = energy_lr3
            plotdata[(T, N)][("mass", 3)] = mass_lr3

            for (i, (ϵ_lr3, ψ_lr3)) in enumerate(sol_lr3_aug)
                ψp_lr3, ψm_lr3 = EPMAfem.pmview(ψ_lr3, model)
                energy_lr3_aug[i] = dot(ψp_lr3, Mp*ψp_lr3) + dot(ψm_lr3, Mm*ψm_lr3)
                mass_lr3_aug[i] = dot(xp, ψp_lr3*Ωp) + dot(xm, ψm_lr3*Ωm)
            end
            plotdata[(T, N)][("energy", (3, :aug))] = energy_lr3_aug
            plotdata[(T, N)][("mass", (3, :aug))] = mass_lr3_aug
        end

        if EPMAfem.n_basis(problem).nΩ.p > 2*6
            for (i, (ϵ_lr6, ψm_lr6)) in enumerate(sol_lr6)
                ψp_lr6, ψm_lr6 = EPMAfem.pmview(ψm_lr6, model)
                energy_lr6[i] = dot(ψp_lr6, Mp*ψp_lr6) + dot(ψm_lr6, Mm*ψm_lr6)
                mass_lr6[i] = dot(xp, ψp_lr6*Ωp) + dot(xm, ψm_lr6*Ωm)
            end
            plotdata[(T, N)][("energy", 6)] = energy_lr6
            plotdata[(T, N)][("mass", 6)] = mass_lr6

            for (i, (ϵ_lr6, ψm_lr6)) in enumerate(sol_lr6_aug)
                ψp_lr6, ψm_lr6 = EPMAfem.pmview(ψm_lr6, model)
                energy_lr6_aug[i] = dot(ψp_lr6, Mp*ψp_lr6) + dot(ψm_lr6, Mm*ψm_lr6)
                mass_lr6_aug[i] = dot(xp, ψp_lr6*Ωp) + dot(xm, ψm_lr6*Ωm)
            end
            plotdata[(T, N)][("energy", (6, :aug))] = energy_lr6_aug
            plotdata[(T, N)][("mass", (6, :aug))] = mass_lr6_aug
        end

        if EPMAfem.n_basis(problem).nΩ.p > 2*20
            for (i, (ϵ_lr20, ψm_lr20)) in enumerate(sol_lr20)
                ψp_lr20, ψm_lr20 = EPMAfem.pmview(ψm_lr20, model)
                energy_lr20[i] = dot(ψp_lr20, Mp*ψp_lr20) + dot(ψm_lr20, Mm*ψm_lr20)
                mass_lr20[i] = dot(xp, ψp_lr20*Ωp) + dot(xm, ψm_lr20*Ωm)
            end
            plotdata[(T, N)][("energy", 20)] = energy_lr20
            plotdata[(T, N)][("mass", 20)] = mass_lr20
        end
    end
end

using Serialization
serialize(joinpath(figpath, "plotdata.jls"), plotdata)
plotdata = deserialize(joinpath(figpath, "plotdata.jls"))

N_to_color = Dict(3 => 1, 5 => 2, 15 => 3, 21 => 4)
# FINAL
for T in Ts
    p1 = plot()
    for N in Ns
        plot!(p1, -1.5:0.01:1.5, x -> plotdata[(T, N)][("final", Inf)](VectorValue(x)), label=L"P_{%$(N)}", color=N_to_color[N])
        m_every = 15
        if haskey(plotdata[(T, N)], ("final", 3))
            plot!(p1, -1.5:0.01:1.5, x -> plotdata[(T, N)][("final", 3)](VectorValue(x)),  label=L"P_{%$(N)}, r=3", color=N_to_color[N], ls=:dot)
            plot!(p1, -1.5:0.01:1.5, x -> plotdata[(T, N)][("final", (3, :aug))](VectorValue(x)),  label=L"P_{%$(N)}, r=3 (aug)", color=N_to_color[N], ls=:dashdot)
        end
        if haskey(plotdata[(T, N)], ("final", 6))
            plot!(p1, -1.5:0.01:1.5, x -> plotdata[(T, N)][("final", 6)](VectorValue(x)), label=L"P_{%$(N)}, r=6", color=N_to_color[N], ls=:solid)
            plot!(p1, -1.5:0.01:1.5, x -> plotdata[(T, N)][("final", (6, :aug))](VectorValue(x)), label=L"P_{%$(N)}, r=6 (aug)", color=N_to_color[N], ls=:dashdot)
            # scatter!(p1, -1.5:0.15:1.5, x -> plotdata[(T, N)][("final", 6)](VectorValue(x)), label=nothing, color=N_to_color[N], marker=:x)
            # plot!(p1, [missing], [missing], label=L"P_{%$(N)}, r=6", color=N_to_color[N], marker=:x, ls=:solid)
        end
    end
    plot!(p1, size=(400, 300), dpi=1000, fontfamily="Computer Modern", right_margin=2Plots.mm, legend=:topright)
    xlabel!(p1, L"x")
    ylabel!(p1, L"\psi_0")
    savefig(p1, joinpath(figpath, "T_$(T)_final.png"))
end

# ENERGY
for T in Ts
    for N in Ns
        @show plotdata[(T, N)]["energy_init"]
    end
    E_init = plotdata[(T, maximum(Ns))]["energy_init"]/4π
    # E_init = hquadrature(x -> init_x(x)^2, -1.5, 1.5)[1]
    p2 = plot()
    for N in Ns
        plot!(p2, 1.0 .- energy_model, abs.(reverse(plotdata[(T, N)][("energy", Inf)]/(4π) .- E_init)), label=L"P_{%$(N)}", color=N_to_color[N])
        if haskey(plotdata[(T, N)], ("energy", 3))
            plot!(p2, 1.0 .- energy_model, abs.(reverse(plotdata[(T, N)][("energy", 3)]/(4π) .- E_init)), label=L"P_{%$(N)}, r=3", color=N_to_color[N], ls=:dot)
            plot!(p2, 1.0 .- energy_model, abs.(reverse(plotdata[(T, N)][("energy", (3, :aug))]/(4π) .- E_init)), label=L"P_{%$(N)}, r=3 (aug)", color=N_to_color[N], ls=:dot)
        end
        if haskey(plotdata[(T, N)], ("energy", 6))
            plot!(p2,1.0 .-  energy_model, abs.(reverse(plotdata[(T, N)][("energy", 6)]/(4π) .- E_init)), label=L"P_{%$(N)}, r=6", color=N_to_color[N], ls=:solid)
            plot!(p2,1.0 .-  energy_model, abs.(reverse(plotdata[(T, N)][("energy", (6, :aug))]/(4π) .- E_init)), label=L"P_{%$(N)}, r=6 (aug)", color=N_to_color[N], ls=:solid)
            # scatter!(p2, 1.0 .- energy_model[3:5:end], abs.(reverse(plotdata[(T, N)][("energy", 6)]/(4π) .- E_init))[3:5:end] , label=nothing, color=N_to_color[N], marker=:x)
            # plot!(p2, [missing], [missing], label=L"P_{%$(N)}, r=6", color=N_to_color[N], ls=:solid, marker=:x)
        end
        if haskey(plotdata[(T, N)], ("energy", 20))
           plot!(p2, 1.0 .- energy_model, abs.(reverse(plotdata[(T, N)][("energy", 20)]/(4π) .- E_init)), label=L"P_{%$(N)}, r=20", color=N_to_color[N], ls=:dash)
        end
    end
    ylims!(3e-16, 1e-0)
    plot!(p2, size=(400, 300), dpi=1000, fontfamily="Computer Modern", right_margin=2Plots.mm, legend=:topleft, yaxis=:log, yticks=[1e0, 1e-5, 1e-10, 1e-15])
    xlabel!(p2, L"t")
    ylabel!(p2, L"|{\mathcal{E}}(t)-\mathcal{E}_0\,\, |")
    savefig(p2, joinpath(figpath, "T_$(T)_energy.png"))
end

# MASS
for T in Ts
    for N in Ns
        @show plotdata[(T, N)]["mass_init"]
    end
    M_init = plotdata[(T, maximum(Ns))]["mass_init"]
    p3 = plot()
    for N in Ns
        plot!(p3, (1 .- energy_model), abs.(reverse(plotdata[(T, N)][("mass", Inf)]) .- M_init), label=L"P_{%$(N)}", color=N_to_color[N])
        if haskey(plotdata[(T, N)], ("mass", 3))
            plot!(p3, (1 .- energy_model), abs.(reverse(plotdata[(T, N)][("mass", 3)]) .- M_init), label=L"P_{%$(N)}, r=3", color=N_to_color[N], ls=:dot)
            plot!(p3, (1 .- energy_model), abs.(reverse(plotdata[(T, N)][("mass", (3, :aug))]) .- M_init), label=L"P_{%$(N)}, r=3 (aug)", color=N_to_color[N], ls=:dashdot)
        end
        if haskey(plotdata[(T, N)], ("mass", 6))
            plot!(p3, (1 .- energy_model), abs.(reverse(plotdata[(T, N)][("mass", 6)]) .- M_init), label=L"P_{%$(N)}, r=6", color=N_to_color[N], ls=:solid)
            plot!(p3, (1 .- energy_model), abs.(reverse(plotdata[(T, N)][("mass", (6, :aug))]) .- M_init), label=L"P_{%$(N)}, r=6 (aug)", color=N_to_color[N], ls=:dashdot)
            # plot!(p3, (1 .- energy_model), abs.(reverse(plotdata[(T, N)][("mass", (6, :aug))]) .- M_init), label=nothing, color=:red, ls=:dashdot)
            # scatter!(p3, (1 .- energy_model)[3:5:end], abs.(reverse(plotdata[(T, N)][("mass", 6)])[3:5:end] .- M_init), label=nothing, color=N_to_color[N], marker=:x)
            # plot!(p3, [missing], [missing], label=L"P_{%$(N)}, r=6", color=N_to_color[N], ls=:solid, marker=:x)
        end
        # if haskey(plotdata[(T, N)], ("mass", 20))
        #    plot!(p3, (1. .- energy_model), abs.(reverse(plotdata[(T, N)][("mass", 20)]) .- M_init), label=L"P_{%$(N)}, r=20", color=N_to_color[N], ls=:dash)
        # end
    end
    ylims!(3e-16, 1e-0)
    plot!(p3, size=(400, 300), dpi=1000, fontfamily="Computer Modern", right_margin=2Plots.mm, legend=:topleft, yaxis=:log, yticks=[1e0, 1e-5, 1e-10, 1e-15])
    xlabel!(p3, L"t")
    ylabel!(p3, L"|M(t) - M_0\, \, |")
    savefig(p3, joinpath(figpath, "T_$(T)_mass.png"))
end





plotdata[(Inf, 21)][("mass", (6, :aug))] = plotdata[(Inf, 21)][("mass", (10, :aug))]

    plot!(p3, size=(400, 300), dpi=1000, fontfamily="Computer Modern", right_margin=2Plots.mm, legend=:topleft, yaxis=:log, yticks=[1e0, 1e-5, 1e-10, 1e-15])
