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

nb = EPMAfem.n_basis(model)
basis_augmentation = (p=(U = EPMAfem.allocate_mat(EPMAfem.cpu(), nb.nx.p, 0),
                            V = EPMAfem.allocate_mat(EPMAfem.cpu(), nb.nΩ.p, 1)),
                        m=(U = EPMAfem.allocate_mat(EPMAfem.cpu(), nb.nx.m, 0),
                            V = EPMAfem.allocate_mat(EPMAfem.cpu(), nb.nΩ.m, 0)))


Ωp, Ωm = EPMAfem.SphericalHarmonicsModels.eval_basis(EPMAfem.direction_model(model), Ω->1/(4π)) |> EPMAfem.architecture(problem)
Ωp_b = vec(Ωp' * problem.direction_discretization.absΩp[1])

copy!(@view(basis_augmentation.p.V[:, 1]), Ωp)
copy!(@view(basis_augmentation.p.V[:, 2]), Ωp_b)
basis_augmentation.p.V .= qr(basis_augmentation.p.V).Q |> Matrix

system_full = EPMAfem.implicit_midpoint2(problem, Krylov.minres);
system_lowrank = EPMAfem.implicit_midpoint_dlr5(problem; max_ranks=(p=3, m=3));
system_lowrank_aug = EPMAfem.implicit_midpoint_dlr5(problem; max_ranks=(p=3, m=3), basis_augmentation=basis_augmentation);
system_lowrank5 = EPMAfem.implicit_midpoint_dlr5(problem; max_ranks=(p=5, m=5));
system_lowrank5_aug = EPMAfem.implicit_midpoint_dlr5(problem; max_ranks=(p=5, m=5), basis_augmentation=basis_augmentation);

direction_model2 = EPMAfem.SphericalHarmonicsModels.EOSphericalHarmonicsModel(7, 1)
model2 = EPMAfem.DiscretePNModel(space_model, range(0, 2, length=nϵ), direction_model2)
problem2 = EPMAfem.discretize_problem(equations, model2, EPMAfem.cpu())
discrete_rhs2 = EPMAfem.discretize_rhs(excitation, model2, EPMAfem.cpu())[1];
discrete_rhs2.bΩp .*= 2 # this should be fixed in the discretization...
system_full2 = EPMAfem.implicit_midpoint2(problem2, Krylov.minres);

space_model3 = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((-1, 0), (600,)))
model3 = EPMAfem.DiscretePNModel(space_model3, range(0, 2, length=3nϵ), direction_model)
problem3 = EPMAfem.discretize_problem(equations, model3, EPMAfem.cpu())
discrete_rhs3 = EPMAfem.discretize_rhs(excitation, model3, EPMAfem.cpu())[1];
discrete_rhs3.bΩp .*= 2 # this should be fixed in the discretization...
system_full3 = EPMAfem.implicit_midpoint2(problem3, Krylov.minres);

sol = system_full * discrete_rhs;
sol2 = system_full2 * discrete_rhs2;
sol3 = system_full3 * discrete_rhs3;
sol_lowrank = system_lowrank * discrete_rhs;
sol_lowrank_aug = system_lowrank_aug * discrete_rhs;
sol_lowrank5 = system_lowrank5 * discrete_rhs;
sol_lowrank5_aug = system_lowrank5_aug * discrete_rhs;

function exact_solution(ϵ, x, Ω)
    return EPMAfem.beam_direction_distribution(excitation, 1, Ω) * EPMAfem.beam_energy_distribution(excitation, 1, ϵ - x/abs(Ω[1]))
end

function exact_solution_average(ϵ, x)
    return EPMAfem.SphericalHarmonicsModels.lebedev_quadrature_max()(Ω -> exact_solution(ϵ, x, Ω))
end

begin
    probe0 = EPMAfem.PNProbe(model, EPMAfem.cpu(), Ω=Ω -> 1.0, ϵ=1.7)
    probe2_0 = EPMAfem.PNProbe(model2, EPMAfem.cpu(), Ω=Ω -> 1.0, ϵ=1.7)
    func_full_0 = EPMAfem.interpolable(probe0, sol)
    func_full2_0 = EPMAfem.interpolable(probe2_0, sol2)
    func_lowrank_0 = EPMAfem.interpolable(probe0, sol_lowrank)
    func_lowrank_aug_0 = EPMAfem.interpolable(probe0, sol_lowrank_aug)
    func_lowrank5_0 = EPMAfem.interpolable(probe0, sol_lowrank5)
    func_lowrank5_aug_0 = EPMAfem.interpolable(probe0, sol_lowrank5_aug)

    probe1 = EPMAfem.PNProbe(model, EPMAfem.cpu(), Ω=Ω -> 1.0, ϵ=1.3)
    probe2_1 = EPMAfem.PNProbe(model2, EPMAfem.cpu(), Ω=Ω -> 1.0, ϵ=1.3)
    func_full_1 = EPMAfem.interpolable(probe1, sol)
    func_full2_1 = EPMAfem.interpolable(probe2_1, sol2)
    func_lowrank_1 = EPMAfem.interpolable(probe1, sol_lowrank)
    func_lowrank_aug_1 = EPMAfem.interpolable(probe1, sol_lowrank_aug)
    func_lowrank5_1 = EPMAfem.interpolable(probe1, sol_lowrank5)
    func_lowrank5_aug_1 = EPMAfem.interpolable(probe1, sol_lowrank5_aug)

    probe2 = EPMAfem.PNProbe(model, EPMAfem.cpu(), Ω=Ω -> 1.0, ϵ=0.8)
    probe2_2 = EPMAfem.PNProbe(model2, EPMAfem.cpu(), Ω=Ω -> 1.0, ϵ=0.8)
    func_full_2 = EPMAfem.interpolable(probe2, sol)
    func_full2_2 = EPMAfem.interpolable(probe2_2, sol2)
    func_lowrank_2 = EPMAfem.interpolable(probe2, sol_lowrank)
    func_lowrank_aug_2 = EPMAfem.interpolable(probe2, sol_lowrank_aug)
    func_lowrank5_2 = EPMAfem.interpolable(probe2, sol_lowrank5)
    func_lowrank5_aug_2 = EPMAfem.interpolable(probe2, sol_lowrank5_aug)

    plot(-1:0.01:0, x -> exact_solution_average(1.7, x), color=:black, ls=:solid, label=L"\textrm{exact}", linewidth=2)
    plot!(size=(400, 300), dpi=1000, fontfamily="Computer Modern", right_margin=2Plots.mm, xflip=true, legend=:topright)

    plot!(-1:0.01:0, x -> func_full2_0(VectorValue(x)), color=5, ls=:solid, label=L"P_{7}")
    plot!(-1:0.01:0, x -> func_full_0(VectorValue(x)), color=1, ls=:solid, label=L"P_{47}")
    plot!(-1:0.01:0, x -> func_lowrank_0(VectorValue(x)), color=2, ls=:solid, label=L"P_{47}, \textrm{(3, default)}")
    plot!(-1:0.01:0, x -> func_lowrank_aug_0(VectorValue(x)), color=2, ls=:solid, label=nothing, alpha=0.5)
    plot!(-1:0.01:0, x -> func_lowrank5_0(VectorValue(x)), color=3, ls=:solid, label=L"P_{47}, \textrm{(5, default)}")
    plot!(-1:0.01:0, x -> func_lowrank5_aug_0(VectorValue(x)), color=3, ls=:solid, label=nothing, alpha=0.5)
    plot!([], [], color=:gray, alpha=0.5, label=L"\textrm{mass\, cons.}")
    annotate!(-0.12, 0.9, (L"t=0.3", 10))

    plot!(-1:0.01:0, x -> exact_solution_average(1.3, x), color=:black, ls=:dash, label=nothing, linewidth=2)
    plot!(-1:0.01:0, x -> func_full2_1(VectorValue(x)), color=5, label=nothing, ls=:dash)
    plot!(-1:0.01:0, x -> func_full_1(VectorValue(x)), color=1, label=nothing, ls=:dash)
    plot!(-1:0.01:0, x -> func_lowrank_1(VectorValue(x)), color=2, label=nothing, ls=:dash)
    plot!(-1:0.01:0, x -> func_lowrank_aug_1(VectorValue(x)), color=2, label=nothing, ls=:dash, alpha=0.5)
    plot!(-1:0.01:0, x -> func_lowrank5_1(VectorValue(x)), color=3, label=nothing, ls=:dash)
    plot!(-1:0.01:0, x -> func_lowrank5_aug_1(VectorValue(x)), color=3, label=nothing, ls=:dash, alpha=0.5)
    annotate!(-0.49, 0.8, (L"t=0.7", 10))

    plot!(-1:0.01:0, x -> exact_solution_average(0.8, x), color=:black, ls=:dashdot, label=nothing, linewidth=2)
    plot!(-1:0.01:0, x -> func_full2_2(VectorValue(x)), color=5, label=nothing, ls=:dashdot)
    plot!(-1:0.01:0, x -> func_full_2(VectorValue(x)), color=1, label=nothing, ls=:dashdot)
    plot!(-1:0.01:0, x -> func_lowrank_2(VectorValue(x)), color=2, label=nothing, ls=:dashdot)
    plot!(-1:0.01:0, x -> func_lowrank_aug_2(VectorValue(x)), color=2, label=nothing, ls=:dashdot, alpha=0.5)
    plot!(-1:0.01:0, x -> func_lowrank5_2(VectorValue(x)), color=3, label=nothing, ls=:dashdot)
    plot!(-1:0.01:0, x -> func_lowrank5_aug_2(VectorValue(x)), color=3, label=nothing, ls=:dashdot, alpha=0.5)
    annotate!(-0.78, 0.25, (L"t=1.2", 10))
    xlabel!(L"x")
    ylabel!(L"\psi_0")
    ylims!(-0.1, 1.25)
    plot!(legend_columns=2)
    savefig(joinpath(figpath, "mass_over_time_energy.png"))
end

energies = zeros(length(sol), 6)
mass = zeros(length(sol), 6)
Mp = EPMAfem.SpaceModels.assemble_bilinear(EPMAfem.SpaceModels.∫R_uv, EPMAfem.space_model(model), EPMAfem.SpaceModels.even(EPMAfem.space_model(model)), EPMAfem.SpaceModels.even(EPMAfem.space_model(model)))
Mm = EPMAfem.SpaceModels.assemble_bilinear(EPMAfem.SpaceModels.∫R_uv, EPMAfem.space_model(model), EPMAfem.SpaceModels.odd(EPMAfem.space_model(model)), EPMAfem.SpaceModels.odd(EPMAfem.space_model(model)))

Mp3 = EPMAfem.SpaceModels.assemble_bilinear(EPMAfem.SpaceModels.∫R_uv, EPMAfem.space_model(model3), EPMAfem.SpaceModels.even(EPMAfem.space_model(model3)), EPMAfem.SpaceModels.even(EPMAfem.space_model(model3)))
Mm3 = EPMAfem.SpaceModels.assemble_bilinear(EPMAfem.SpaceModels.∫R_uv, EPMAfem.space_model(model3), EPMAfem.SpaceModels.odd(EPMAfem.space_model(model3)), EPMAfem.SpaceModels.odd(EPMAfem.space_model(model3)))

for (i, ((_, ψfull), (_, ψfull2), (_, ψlr), (_, ψlr_aug), (_, ψlr5), (_, ψlr_aug5),)) in enumerate(zip(sol, sol2, sol_lowrank, sol_lowrank_aug, sol_lowrank5, sol_lowrank5_aug))
    ψfullp, ψfullm = EPMAfem.pmview(ψfull, model)
    ψfull2p, ψfull2m = EPMAfem.pmview(ψfull2, model2)
    ψlrp, ψlrm = EPMAfem.pmview(ψlr, model)
    ψlr_augp, ψlr_augm = EPMAfem.pmview(ψlr_aug, model)
    ψlr5p, ψlr5m = EPMAfem.pmview(ψlr5, model)
    ψlr5_augp, ψlr5_augm = EPMAfem.pmview(ψlr_aug5, model)
    energies[i, 1] = dot(ψfullp, Mp*ψfullp) + dot(ψfullm, Mm*ψfullm)
    energies[i, 2] = dot(ψfull2p, Mp*ψfull2p) + dot(ψfull2m, Mm*ψfull2m)
    energies[i, 3] = dot(ψlrp, Mp*ψlrp) + dot(ψlrm, Mm*ψlrm)
    energies[i, 4] = dot(ψlr_augp, Mp*ψlr_augp) + dot(ψlr_augm, Mm*ψlr_augm)
    energies[i, 5] = dot(ψlr5p, Mp*ψlr5p) + dot(ψlr5m, Mm*ψlr5m)
    energies[i, 6] = dot(ψlr5_augp, Mp*ψlr5_augp) + dot(ψlr5_augm, Mm*ψlr5_augm)

    xp, xm = EPMAfem.SpaceModels.eval_basis(EPMAfem.space_model(model), x -> 1.0)
    Ωp2, Ωm2 = EPMAfem.SphericalHarmonicsModels.eval_basis(EPMAfem.direction_model(model2), Ω -> 1.0/(4π))
    mass[i, 1] = dot(xp, ψfullp, Ωp) + dot(xm, ψfullm, Ωm)
    mass[i, 2] = dot(xp, ψfull2p, Ωp2) + dot(xm, ψfull2m, Ωm2)
    mass[i, 3] = dot(xp, ψlrp, Ωp) + dot(xm, ψlrm, Ωm)
    mass[i, 4] = dot(xp, ψlr_augp, Ωp) + dot(xm, ψlr_augm, Ωm)
    mass[i, 5] = dot(xp, ψlr5p, Ωp) + dot(xm, ψlr5m, Ωm)
    mass[i, 6] = dot(xp, ψlr5_augp, Ωp) + dot(xm, ψlr5_augm, Ωm)
end

energies3 = zeros(length(sol3), 1)
mass3 = zeros(length(sol3), 1)
for (i, (_, ψfull3)) in enumerate(sol3)
    ψfull3p, ψfull3m = EPMAfem.pmview(ψfull3, model3)
    xp3, xm3 = EPMAfem.SpaceModels.eval_basis(EPMAfem.space_model(model3), x -> 1.0)
    Ωp, Ωm = EPMAfem.SphericalHarmonicsModels.eval_basis(EPMAfem.direction_model(model), Ω -> 1.0/(4π))

    energies3[i, 1] = dot(ψfull3p, Mp3*ψfull3p) + dot(ψfull3m, Mm3*ψfull3m)
    mass3[i, 1] = dot(xp3, ψfull3p, Ωp) + dot(xm3, ψfull3m, Ωm)
end

begin
    mass_bound = EPMAfem.SphericalHarmonicsModels.lebedev_quadrature_max()(Ω -> abs(Ω[1])/(4π)*EPMAfem.beam_direction_distribution(excitation, 1, Ω))*EPMAfem.hquadrature(ϵ -> EPMAfem.beam_energy_distribution(excitation, 1, ϵ), 0, 2)[1]

    t = 2.0 .- EPMAfem.energy_model(model) |> reverse

    hline([mass_bound / mass_bound], color=:gray, ls=:dash, label=L"\mathcal{M}_{\textrm{max}}")
    xlims!(0, 2)
    plot!(t, mass[:, 2]./mass_bound, label=L"P_{7}", xflip=false, color=5)
    plot!(t, mass[:, 1]./mass_bound, label=L"P_{47}", color=1)
    plot!(t, mass[:, 3]./mass_bound, label=L"P_{47} \textrm{(3, default)}", color=2, ls=:solid)
    plot!(t, mass[:, 4]./mass_bound, label=L"P_{47} \textrm{(3, mass \, cons.)}", color=2, ls=:dash)
    plot!(t, mass[:, 5]./mass_bound, label=L"P_{47} \textrm{(5, default)}", color=3, ls=:solid)
    plot!(t, mass[:, 6]./mass_bound, label=L"P_{47} \textrm{(5, mass\, cons.)}", color=3, ls=:dash)

    zoom_range = 120:-1:45
    zoom_datarange = (0.998, 1.004)
    plot!([t[first(zoom_range)], t[last(zoom_range)]], [zoom_datarange[1], zoom_datarange[1]], color=:black, label=nothing)
    plot!([t[first(zoom_range)], t[last(zoom_range)]], [zoom_datarange[2], zoom_datarange[2]], color=:black, label=nothing)
    plot!([t[first(zoom_range)], t[first(zoom_range)]], [zoom_datarange[1], zoom_datarange[2]], color=:black, label=nothing)
    plot!([t[last(zoom_range)], t[last(zoom_range)]], [zoom_datarange[1], zoom_datarange[2]], color=:black, label=nothing)
    xlabel!(L"t")
    ylabel!(L"\mathcal{M}(t) / \mathcal{M}_{\textrm{max}}")
    ylims!(-0.01, 1.01)

    p_i = plot!(t[zoom_range],
                [fill(mass_bound/mass_bound, length(zoom_range)) mass[zoom_range, 2]./mass_bound mass[zoom_range, 1]./mass_bound mass[zoom_range, 3]./mass_bound mass[zoom_range, 4]./mass_bound mass[zoom_range, 5]./mass_bound mass[zoom_range, 6]./mass_bound],
                color=[:gray 5 1 2 2 3 3], ls=[:dash :solid :solid :solid :dash :solid :dash],
                label=nothing,
                inset=(1, bbox(0.02, 0.3, 0.45, 0.65, :bottom, :right)), subplot=2, ylims=zoom_datarange, xflip=false, framestyle=:box, tickfontsize=5)
    plot!(size=(400, 300), dpi=1000, fontfamily="Computer Modern", right_margin=2Plots.mm, legend=:bottomleft)

    savefig(joinpath(figpath, "mass.png"))
end

function cumtrapz(v, Δ)
    ∫v = similar(v)
    ∫v[1] = v[1]*Δ/2
    for i in 2:length(v)
        ∫v[i] = ∫v[i-1] + (v[i-1] + v[i])*Δ/2 
    end
    return ∫v
end

begin
    # energy_bound = 4π*EPMAfem.SphericalHarmonicsModels.lebedev_quadrature_max()(Ω -> abs(Ω[1])/(4π)*(EPMAfem.beam_direction_distribution(excitation, 1, Ω))^2)*EPMAfem.hquadrature(ϵ -> (EPMAfem.beam_energy_distribution(excitation, 1, ϵ))^2, 0, 2)[1]
    energy_bound = dot(discrete_rhs.bΩp, problem.direction_discretization.absΩp[1] \ discrete_rhs.bΩp)/2 * cumtrapz(discrete_rhs.bϵ.^2, step(model.energy_mdl))[end]


    t = 2.0 .- EPMAfem.energy_model(model) |> reverse
    t3 = 2.0 .- EPMAfem.energy_model(model3) |> reverse
    # t4 = 2.0 .- EPMAfem.energy_model(model4) |> reverse

    hline([energy_bound /energy_bound], color=:gray, ls=:dash, label=L"\mathcal{E}_{\textrm{max}}")
    xlims!(0, 2)
    plot!(t, energies[:, 2]./energy_bound, label=L"P_{7}", xflip=false, color=5)
    plot!(t, energies[:, 1]./energy_bound, label=L"P_{47}", color=1)
    plot!(t, energies[:, 3]./energy_bound, label=L"P_{47} \textrm{(3, default)}", color=2, ls=:solid)
    plot!(t, energies[:, 4]./energy_bound, label=L"P_{47} \textrm{(3, mass\, cons.)}", color=2, ls=:dash)
    plot!(t, energies[:, 5]./energy_bound, label=L"P_{47} \textrm{(5, default)}", color=3, ls=:solid)
    plot!(t, energies[:, 6]./energy_bound, label=L"P_{47} \textrm{(5, mass\, cons.)}", color=3, ls=:solid)
    plot!(t3, energies3[:, 1]./energy_bound, label=L"P_{47} \textrm{(highres)}", color=4, ls=:dash)

    zoom_range = 120:-1:45
    zoom_datarange = (0.996, 1.0001)
    plot!([t[first(zoom_range)], t[last(zoom_range)]], [zoom_datarange[1], zoom_datarange[1]], color=:black, label=nothing, linewidth=1)
    plot!([t[first(zoom_range)], t[last(zoom_range)]], [zoom_datarange[2], zoom_datarange[2]], color=:black, label=nothing, linewidth=1)
    plot!([t[first(zoom_range)], t[first(zoom_range)]], [zoom_datarange[1], zoom_datarange[2]], color=:black, label=nothing, linewidth=1)
    plot!([t[last(zoom_range)], t[last(zoom_range)]], [zoom_datarange[1], zoom_datarange[2]], color=:black, label=nothing, linewidth=1)
    xlabel!(L"t")
    ylabel!(L"\mathcal{E}(t) / \mathcal{E}_{\textrm{max}}")

    p_i = plot!([t[zoom_range] t[zoom_range] t[zoom_range] t[zoom_range] t[zoom_range] t[zoom_range] t[zoom_range] t3[3zoom_range]],
                [fill(energy_bound/energy_bound, length(zoom_range)) energies[zoom_range, 2]./energy_bound energies[zoom_range, 1]./energy_bound energies[zoom_range, 3]./energy_bound energies[zoom_range, 4]./energy_bound  energies[zoom_range, 5]./energy_bound energies[zoom_range, 6]./energy_bound energies3[3zoom_range, 1]./energy_bound], 
                color=[:gray 5 1 2 2 3 3 4],
                ls=[:dash :solid :solid :solid :dash :solid :dash :dash],
                label=nothing,
                inset=(1, bbox(0.02, 0.3, 0.45, 0.65, :bottom, :right)), subplot=2, ylims=zoom_datarange, xflip=false, framestyle=:box, tickfontsize=5)
    plot!(size=(400, 300), dpi=1000, fontfamily="Computer Modern", right_margin=2Plots.mm, legend=:bottomleft)

    savefig(joinpath(figpath, "energy.png"))
end

## TIMINGS
using BenchmarkTools
b1 = @benchmark for (ϵ, ψ) in $(sol) end # 35.332 s
b2 = @benchmark for (ϵ, ψ) in $(sol2) end # 1.347 s
b3 = @benchmark for (ϵ, ψ) in $(sol_lowrank) end # 1.498s
b4 = @benchmark for (ϵ, ψ) in $(sol_lowrank_aug) end # 1.604s
b5 = @benchmark for (ϵ, ψ) in $(sol_lowrank5) end # 1.984s
b6 = @benchmark for (ϵ, ψ) in $(sol_lowrank5_aug) end # 2.761s
b7 = @benchmark for (ϵ, ψ) in $(sol3) end

