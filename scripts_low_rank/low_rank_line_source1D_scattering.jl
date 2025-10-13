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

# figpath = mkpath(joinpath(dirname(@__FILE__), "figures/1D_vacuum_linesource"))

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

# plotdata = Dict()
# Ts = [Inf] # [Inf, 1.0, 10.0, 100.0]
# Ns = [3, 7, 15, 21]
# for T in Ts
#     for N in Ns
T = Inf
N = 15
        # plotdata[(T, N)] = Dict()
        equations = PlaneSourceEquations{T}()
        space_model = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((-1.5, 1.5), (70)))
        direction_model = EPMAfem.SphericalHarmonicsModels.EOSphericalHarmonicsModel(N, 1)
        model = EPMAfem.DiscretePNModel(space_model, energy_model, direction_model)
        problem = EPMAfem.discretize_problem(equations, model, EPMAfem.cpu())

        @show N, EPMAfem.n_basis(problem)

        # source / boundary condition (here: zero)
        source = EPMAfem.Rank1DiscretePNVector(false, model, EPMAfem.cpu(), zeros(EPMAfem.n_basis(model).nϵ), zeros(EPMAfem.n_basis(model).nx.p), zeros(EPMAfem.n_basis(model).nΩ.p))

        # initial condition
        Mp = EPMAfem.SpaceModels.assemble_bilinear(EPMAfem.SpaceModels.∫R_uv, EPMAfem.space_model(model), EPMAfem.SpaceModels.even(EPMAfem.space_model(model)), EPMAfem.SpaceModels.even(EPMAfem.space_model(model))) |> EPMAfem.architecture(problem)
        Mm = EPMAfem.SpaceModels.assemble_bilinear(EPMAfem.SpaceModels.∫R_uv, EPMAfem.space_model(model), EPMAfem.SpaceModels.odd(EPMAfem.space_model(model)), EPMAfem.SpaceModels.odd(EPMAfem.space_model(model))) |> EPMAfem.architecture(problem)
        
        # system = EPMAfem.implicit_midpoint2(problem, A -> PNLazyMatrices.schur_complement(A, Krylov.minres, PNLazyMatrices.cache ∘ LinearAlgebra.inv!));
        system = EPMAfem.implicit_midpoint2(problem, LinearAlgebra.:\);
        # system_lr3 = EPMAfem.implicit_midpoint_dlr5(problem, solver=LinearAlgebra.:\, max_ranks=(p=3, m=3));
        # system_lr20 = EPMAfem.implicit_midpoint_dlr5(problem, solver=LinearAlgebra.:\, max_ranks=(p=20, m=20));

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

        Ωp, Ωm = EPMAfem.SphericalHarmonicsModels.eval_basis(EPMAfem.direction_model(model), Ω->1/(4π)) |> EPMAfem.architecture(problem)
        xp, xm = EPMAfem.SpaceModels.eval_basis(EPMAfem.space_model(model), x -> 1.0) |> EPMAfem.architecture(problem)

        Ωp_b2 = EPMAfem.SphericalHarmonicsModels.assemble_linear(EPMAfem.SphericalHarmonicsModels.∫S²_hv(Ω -> abs(Ω[1])/(4π)), EPMAfem.direction_model(model), EPMAfem.SphericalHarmonicsModels.even(EPMAfem.direction_model(model)))
        xp_b2 = EPMAfem.SpaceModels.assemble_linear(EPMAfem.SpaceModels.∫∂R_ngv{EPMAfem.Dimensions.Z}(x -> 1.0), EPMAfem.space_model(model), EPMAfem.SpaceModels.even(EPMAfem.space_model(model))) .|> abs

        # this avoids numerical errors in the quadrature (for xp_b)
        xp_b = vec((Mp \ xp)' * problem.space_discretization.∂p[1])
        Ωp_b = vec(Ωp' * problem.direction_discretization.absΩp[1])

        mass0 = dot(xp, ψ0p*Ωp) + dot(xm, ψ0m*Ωm)
        @show mass0 - 1.0
        
        nb = EPMAfem.n_basis(model)
        basis_augmentation = (p=(U = EPMAfem.allocate_mat(EPMAfem.cpu(), nb.nx.p, 0),
                                 V = EPMAfem.allocate_mat(EPMAfem.cpu(), nb.nΩ.p, 1)),
                              m=(U = EPMAfem.allocate_mat(EPMAfem.cpu(), nb.nx.m, 0),
                                 V = EPMAfem.allocate_mat(EPMAfem.cpu(), nb.nΩ.m, 0)))
        
        copy!(@view(basis_augmentation.p.U[:, 1]), Mp \ xp)
        copy!(@view(basis_augmentation.p.U[:, 2]), Mp \ xp_b)
        copy!(@view(basis_augmentation.p.V[:, 1]), Ωp)
        copy!(@view(basis_augmentation.p.V[:, 2]), Ωp_b)
        basis_augmentation.m.U .= 1.0
        basis_augmentation.m.V .= 1.0

        basis_augmentation.p.U .= qr(basis_augmentation.p.U).Q |> Matrix
        basis_augmentation.p.V .= qr(basis_augmentation.p.V).Q |> Matrix
        basis_augmentation.m.U .= qr(basis_augmentation.m.U).Q |> Matrix
        basis_augmentation.m.V .= qr(basis_augmentation.m.V).Q |> Matrix

        conserved_quantities = (p=(U = EPMAfem.allocate_mat(EPMAfem.cpu(), nb.nx.p, 1),
                                   V = EPMAfem.allocate_mat(EPMAfem.cpu(), nb.nΩ.p, 1)),
                                m=(U = EPMAfem.allocate_mat(EPMAfem.cpu(), nb.nx.m, 0),
                                   V = EPMAfem.allocate_mat(EPMAfem.cpu(), nb.nΩ.m, 0)))

        copy!(@view(conserved_quantities.p.U[:, 1]), xp)
        copy!(@view(conserved_quantities.p.U[:, 2]), xp_b)
        copy!(@view(conserved_quantities.p.V[:, 1]), Ωp)
        copy!(@view(conserved_quantities.p.V[:, 2]), Ωp_b)
        
        system_lr10 = EPMAfem.implicit_midpoint_dlr5(problem, solver=LinearAlgebra.:\, max_ranks=(p=5, m=5));
        system_lr10_aug = EPMAfem.implicit_midpoint_dlr5(problem, solver=LinearAlgebra.:\, max_ranks=(p=5, m=5), basis_augmentation=basis_augmentation, conserved_quantities=conserved_quantities);

        # MAKE THE SYSTEMS FULLY MASS CONSERVATIVE BY A REFLECTIVE BOUNDARY CONDITION
        system.coeffs.γ[] = 0
        system_lr10.coeffs.γ[] = 0
        system_lr10_aug.coeffs.γ[] = 0

        # SWITCH OFF TRANSPORT
        # system.coeffs.δ[] = 0
        # system_lr10.coeffs.δ[] = 0
        # system_lr10_aug.coeffs.δ[] = 0

        sol = EPMAfem.IterableDiscretePNSolution(system, source, initial_solution=initial_condition);
        # sol_lr3 = EPMAfem.IterableDiscretePNSolution(system_lr3, source, initial_solution=initial_condition);
        sol_lr10 = EPMAfem.IterableDiscretePNSolution(system_lr10, source, initial_solution=initial_condition);
        sol_lr10_aug = EPMAfem.IterableDiscretePNSolution(system_lr10_aug, source, initial_solution=initial_condition);
        # sol_lr20 = EPMAfem.IterableDiscretePNSolution(system_lr20, source, initial_solution=initial_condition);


        temp, state = iterate(sol_lr10_aug)
        ((Up, Sp, Vtp), (Um, Sm, Vtm)) = EPMAfem.USVt(temp[2])

        Vtp' * Vtp * system_lr10_aug.basis_augmentation.p.V .- system_lr10_aug.basis_augmentation.p.V .|> abs |> maximum
        
        #### GIF
        # Ωp, Ωm = EPMAfem.SphericalHarmonicsModels.eval_basis(EPMAfem.direction_model(model), Ω -> 1.0) |> EPMAfem.architecture(problem)
        masses = zeros(length(sol), 3)
        energies = zeros(length(sol), 3)
        outflux = zeros(length(sol), 3)
        ranksp = zeros(length(sol), 2)
        ranksm = zeros(length(sol), 2)
        @gif for (i, ((ϵ, ψ), (ϵ1, ψ1), (ϵ2, ψ2))) in enumerate(zip(sol, sol_lr10, sol_lr10_aug))
        # @gif for ((ϵ1, ψ1), (ϵ2, ψ2)) in zip(sol_lr10, sol_lr10_aug)
        # @gif for (ϵ, ψ) in sol
            ψp, ψm = EPMAfem.pmview(ψ, model)
            @show dot(ψp, ψp) + dot(ψm, ψm), dot(ψp, ψp), dot(ψm, ψm)
            ψp1, ψm1 = EPMAfem.pmview(ψ1, model)
            ψp2, ψm2 = EPMAfem.pmview(ψ2, model)
            func = EPMAfem.SpaceModels.interpolable((p=collect(ψp*Ωp), m=collect(ψm*Ωm)), EPMAfem.space_model(model))
            func1 = EPMAfem.SpaceModels.interpolable((p=collect(ψp1*Ωp), m=collect(ψm1*Ωm)), EPMAfem.space_model(model))
            func2 = EPMAfem.SpaceModels.interpolable((p=collect(ψp2*Ωp), m=collect(ψm2*Ωm)), EPMAfem.space_model(model))
            plot(-1.5:0.01:1.5, x -> func(VectorValue(x)))
            plot!(-1.5:0.01:1.5, x -> func1(VectorValue(x)))
            plot!(-1.5:0.01:1.5, x -> func2(VectorValue(x)))
            # heatmap(-1.5:0.01:1.5, -1.5:0.01:1.5, (x, y) -> func(VectorValue(x, y)), aspect_ratio=:equal)

            masses[i, 1] = dot(xp, ψp*Ωp) + dot(xm, ψm*Ωm)
            energies[i, 1] = dot(ψp, Mp*ψp) + dot(ψm, Mm*ψm)
            outflux[i, 1] = dot(xp_b, ψp*Ωp_b)
            masses[i, 2] = dot(xp, ψp1*Ωp) + dot(xm, ψm1*Ωm)
            masses[i, 3] = dot(xp, ψp2*Ωp) + dot(xm, ψm2*Ωm)
            energies[i, 2] = dot(ψp1, Mp*ψp1) + dot(ψm1, Mm*ψm1)
            energies[i, 3] = dot(ψp2, Mp*ψp2) + dot(ψm2, Mm*ψm2)
            outflux[i, 2] = dot(xp_b, ψp1*Ωp_b)
            outflux[i, 3] = dot(xp_b, ψp2*Ωp_b)
            ranksp[i, 1] = ψ1.ranks.p[]
            ranksm[i, 1] = ψ1.ranks.m[]
            ranksp[i, 2] = ψ2.ranks.p[]
            ranksm[i, 2] = ψ2.ranks.m[]
        end

        for (i, ((ϵ, ψ), (ϵ2, ψ2))) in enumerate(zip(sol,  sol_lr10_aug))
            ψp, ψm = EPMAfem.pmview(ψ, model)
            @show dot(ψp, ψp) + dot(ψm, ψm), dot(ψp, ψp), dot(ψm, ψm)
        end




        plot(energies[:, 1])
        plot!(energies[:, 2])
        plot!(energies[:, 3])

        function cumtrapz(v, Δ)
            ∫v = similar(v)
            ∫v[1] = v[1]*Δ/2
            for i in 2:length(v)
                ∫v[i] = ∫v[i-1] + (v[i-1] + v[i])*Δ/2 
            end
            return ∫v
        end

        begin
            plot(abs.(masses[:, 1] .- 1.0), yaxis=:log)
            plot!(abs.(masses[:, 2] .- 1.0))
            plot!(abs.(masses[:, 3] .- 1.0))

            plot!(abs.(masses[:, 1] .+ cumtrapz(outflux[:, 1], step(energy_model)) .- 1.0), yaxis=:log, color=1, ls=:dash)
            plot!(abs.(masses[:, 2] .+ cumtrapz(outflux[:, 2], step(energy_model)) .- 1.0), color=2, ls=:dash)
            plot!(abs.(masses[:, 3] .+ cumtrapz(outflux[:, 3], step(energy_model)) .- 1.0), color=3, ls=:dash)
            ylims!(1e-16, 1.0)
            yticks!([1, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 1e-16])
        end
        
        plot!()
        
        begin
            # plot(masses[:, 1])
            # plot(masses[:, 1] .+ cumsum(outflux[:, 1] .* step(energy_model))/4π)
            plot(masses[:, 1] .+ cumtrapz(outflux[:, 1], step(energy_model))/4π)
            # plot!(1.0.+ cumsum(outflux[:, 1] .* step(energy_model))/4π)
        end
        plot(cumsum(outflux[:, 1] .* step(energy_model))/4π)

        plot(abs.(masses[:, 1] .- 1.0), yaxis=:log, color=1, ls=:dash)
        plot!(abs.(cumsum(outflux[:, 1]) * step(energy_model)/4π), yaxis=:log, ls=:dash)
        plot!(abs.(masses[:, 1] .- 1.0 .+ cumsum(outflux[:, 1]) * step(energy_model)/4π), yaxis=:log, ls=:dash)
        ylims!(1e-16, 1e-5)

        probe = EPMAfem.PNProbe(model, EPMAfem.cpu(), Ω=Ω->1.0, ϵ=0.0)
        func = EPMAfem.interpolable(probe, sol)
        plotdata[(T, N)][("final", Inf)] = func
        if EPMAfem.n_basis(problem).nΩ.p > 2*6
            func_lr3 = EPMAfem.interpolable(probe, sol_lr3)
            plotdata[(T, N)][("final", 3)] = func_lr3
        end
        if EPMAfem.n_basis(problem).nΩ.p > 2*10
            func_lr10 = EPMAfem.interpolable(probe, sol_lr10)
            plotdata[(T, N)][("final", 10)] = func_lr10
        end

        Ωp, Ωm = EPMAfem.SphericalHarmonicsModels.eval_basis(EPMAfem.direction_model(model), Ω->1/(4π)) |> EPMAfem.architecture(problem)
        xp, xm = EPMAfem.SpaceModels.eval_basis(EPMAfem.space_model(model), x -> 1.0) |> EPMAfem.architecture(problem)

        energy, energy_lr3, energy_lr10, energy_lr20 = zeros(length(sol)), zeros(length(sol)), zeros(length(sol)), zeros(length(sol))
        mass, mass_lr3, mass_lr10, mass_lr20 = zeros(length(sol)), zeros(length(sol)), zeros(length(sol)), zeros(length(sol))

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

        if EPMAfem.n_basis(problem).nΩ.p > 2*6
            for (i, (ϵ_lr3, ψ_lr3)) in enumerate(sol_lr3)
                ψp_lr3, ψm_lr3 = EPMAfem.pmview(ψ_lr3, model)
                energy_lr3[i] = dot(ψp_lr3, Mp*ψp_lr3) + dot(ψm_lr3, Mm*ψm_lr3)
                mass_lr3[i] = dot(xp, ψp_lr3*Ωp) + dot(xm, ψm_lr3*Ωm)
            end
            plotdata[(T, N)][("energy", 3)] = energy_lr3
            plotdata[(T, N)][("mass", 3)] = mass_lr3
        end

        if EPMAfem.n_basis(problem).nΩ.p > 2*10
            for (i, (ϵ_lr10, ψm_lr10)) in enumerate(sol_lr10)
                ψp_lr10, ψm_lr10 = EPMAfem.pmview(ψm_lr10, model)
                energy_lr10[i] = dot(ψp_lr10, Mp*ψp_lr10) + dot(ψm_lr10, Mm*ψm_lr10)
                mass_lr10[i] = dot(xp, ψp_lr10*Ωp) + dot(xm, ψm_lr10*Ωm)
            end
            plotdata[(T, N)][("energy", 10)] = energy_lr10
            plotdata[(T, N)][("mass", 10)] = mass_lr10
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

# dot(bΩp, bΩp) ./ 4π * dot(bxp, Mp*bxp)

using Serialization
serialize(joinpath(figpath, "plotdata.jls"), plotdata)
plotdata = deserialize(joinpath(figpath, "plotdata.jls"))

N_to_color = Dict(3 => 1, 7 => 2, 15 => 3, 21 => 4)
# FINAL
for T in Ts
    p1 = plot()
    for N in Ns
        plot!(p1, -1.5:0.01:1.5, x -> plotdata[(T, N)][("final", Inf)](VectorValue(x)), label=L"P_{%$(N)}", color=N_to_color[N])
        m_every = 15
        if haskey(plotdata[(T, N)], ("final", 3))
            plot!(p1, -1.5:0.01:1.5, x -> plotdata[(T, N)][("final", 3)](VectorValue(x)),  label=L"P_{%$(N)}, r=3", color=N_to_color[N], ls=:dot)
        end
        if haskey(plotdata[(T, N)], ("final", 10))
            plot!(p1, -1.5:0.01:1.5, x -> plotdata[(T, N)][("final", 10)](VectorValue(x)), label=nothing, color=N_to_color[N], ls=:solid)
            scatter!(p1, -1.5:0.15:1.5, x -> plotdata[(T, N)][("final", 10)](VectorValue(x)), label=nothing, color=N_to_color[N], marker=:x)
            plot!(p1, [missing], [missing], label=L"P_{%$(N)}, r=10", color=N_to_color[N], marker=:x, ls=:solid)
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
    E_init = plotdata[(T, 21)]["energy_init"]/4π
    # E_init = hquadrature(x -> init_x(x)^2, -1.5, 1.5)[1]
    p2 = plot()
    for N in Ns
        plot!(p2, 1.0 .- energy_model, abs.(reverse(plotdata[(T, N)][("energy", Inf)]/(4π) .- E_init)), label=L"P_{%$(N)}", color=N_to_color[N])
        if haskey(plotdata[(T, N)], ("energy", 3))
            plot!(p2, 1.0 .- energy_model, abs.(reverse(plotdata[(T, N)][("energy", 3)]/(4π) .- E_init)), label=L"P_{%$(N)}, r=3", color=N_to_color[N], ls=:dot)
        end
        if haskey(plotdata[(T, N)], ("energy", 10))
            plot!(p2,1.0 .-  energy_model, abs.(reverse(plotdata[(T, N)][("energy", 10)]/(4π) .- E_init)), label=nothing, color=N_to_color[N], ls=:solid)
            scatter!(p2, 1.0 .- energy_model[3:5:end], abs.(reverse(plotdata[(T, N)][("energy", 10)]/(4π) .- E_init))[3:5:end] , label=nothing, color=N_to_color[N], marker=:x)
            plot!(p2, [missing], [missing], label=L"P_{%$(N)}, r=10", color=N_to_color[N], ls=:solid, marker=:x)
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
    M_init = plotdata[(T, 21)]["mass_init"]
    p3 = plot()
    for N in Ns
        plot!(p3, (1 .- energy_model), abs.(reverse(plotdata[(T, N)][("mass", Inf)]) .- M_init), label=L"P_{%$(N)}", color=N_to_color[N])
        if haskey(plotdata[(T, N)], ("mass", 3))
            plot!(p3, (1 .- energy_model), abs.(reverse(plotdata[(T, N)][("mass", 3)]) .- M_init), label=L"P_{%$(N)}, r=3", color=N_to_color[N], ls=:dot)
        end
        if haskey(plotdata[(T, N)], ("mass", 10))
            plot!(p3, (1 .- energy_model), abs.(reverse(plotdata[(T, N)][("mass", 10)]) .- M_init), label=nothing, color=N_to_color[N], ls=:solid)
            scatter!(p3, (1 .- energy_model)[3:5:end], abs.(reverse(plotdata[(T, N)][("mass", 10)])[3:5:end] .- M_init), label=nothing, color=N_to_color[N], marker=:x)
            plot!(p3, [missing], [missing], label=L"P_{%$(N)}, r=10", color=N_to_color[N], ls=:solid, marker=:x)
        end
        if haskey(plotdata[(T, N)], ("mass", 20))
           plot!(p3, (1. .- energy_model), abs.(reverse(plotdata[(T, N)][("mass", 20)]) .- M_init), label=L"P_{%$(N)}, r=20", color=N_to_color[N], ls=:dash)
        end
    end
    ylims!(3e-16, 1e-0)
    plot!(p3, size=(400, 300), dpi=1000, fontfamily="Computer Modern", right_margin=2Plots.mm, legend=:topleft, yaxis=:log, yticks=[1e0, 1e-5, 1e-10, 1e-15])
    xlabel!(p3, L"t")
    ylabel!(p3, L"|M(t) - M_0\, \, |")
    savefig(p3, joinpath(figpath, "T_$(T)_mass.png"))
end

