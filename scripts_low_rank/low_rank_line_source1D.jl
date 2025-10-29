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
EPMAfem.absorption_coefficient(eq::PlaneSourceEquations, e, ϵ) = 0.0 # 1.0 
EPMAfem.scattering_coefficient(eq::PlaneSourceEquations, e, i, ϵ) = 0.0 #1.0
EPMAfem.mass_concentrations(::PlaneSourceEquations, e, x) = 1.0

EPMAfem.scattering_kernel(::PlaneSourceEquations{Inf}, e, i) = μ -> 1/(4π)
@generated μ₀(::PlaneSourceEquations{T}) where T = return :( $ (2π*hquadrature(μ -> exp(-T*(μ-1)^2), -1, 1)[1]))
EPMAfem.scattering_kernel(eq::PlaneSourceEquations{T}, e, i) where T = μ -> exp(-T*(μ-1)^2)/μ₀(eq)

energy_model = 0:0.01:1.0

T = Inf
N = 17
equations = PlaneSourceEquations{T}()
space_model = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((-1.5, 1.5), (200)))
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
Ωp_m, Ωm_m = EPMAfem.SphericalHarmonicsModels.eval_basis(EPMAfem.direction_model(model), Ω->EPMAfem.Dimensions.Ωz(Ω)/(4π)) |> EPMAfem.architecture(problem)
xp, xm = EPMAfem.SpaceModels.eval_basis(EPMAfem.space_model(model), x -> 1.0) |> EPMAfem.architecture(problem)

Ωp_b2 = EPMAfem.SphericalHarmonicsModels.assemble_linear(EPMAfem.SphericalHarmonicsModels.∫S²_hv(Ω -> abs(Ω[1])/(4π)), EPMAfem.direction_model(model), EPMAfem.SphericalHarmonicsModels.even(EPMAfem.direction_model(model)))
xp_b2 = EPMAfem.SpaceModels.assemble_linear(EPMAfem.SpaceModels.∫∂R_ngv{EPMAfem.Dimensions.Z}(x -> 1.0), EPMAfem.space_model(model), EPMAfem.SpaceModels.even(EPMAfem.space_model(model))) .|> abs

# this avoids numerical errors in the quadrature (for xp_b)
xp_b = vec((Mp \ xp)' * problem.space_discretization.∂p[1])
Ωp_b = vec(Ωp' * problem.direction_discretization.absΩp[1])

mass0 = dot(xp, ψ0p*Ωp) + dot(xm, ψ0m*Ωm)

energy0 = dot(ψ0p, Mp*ψ0p) + dot(ψ0m, Mm*ψ0m)

@show mass0 - 1.0

nb = EPMAfem.n_basis(model)
basis_augmentation = (p=(U = EPMAfem.allocate_mat(EPMAfem.cpu(), nb.nx.p, 0),
                            V = EPMAfem.allocate_mat(EPMAfem.cpu(), nb.nΩ.p, 1)),
                        m=(U = EPMAfem.allocate_mat(EPMAfem.cpu(), nb.nx.m, 0),
                            V = EPMAfem.allocate_mat(EPMAfem.cpu(), nb.nΩ.m, 0)))

basis_augmentation2 = (p=(U = EPMAfem.allocate_mat(EPMAfem.cpu(), nb.nx.p, 0),
                            V = EPMAfem.allocate_mat(EPMAfem.cpu(), nb.nΩ.p, 2)),
                        m=(U = EPMAfem.allocate_mat(EPMAfem.cpu(), nb.nx.m, 0),
                            V = EPMAfem.allocate_mat(EPMAfem.cpu(), nb.nΩ.m, 0)))

copy!(@view(basis_augmentation.p.V[:, 1]), Ωp)

copy!(@view(basis_augmentation2.p.V[:, 1]), Ωp)
copy!(@view(basis_augmentation2.p.V[:, 2]), Ωp_b)

basis_augmentation.p.V .= qr(basis_augmentation.p.V).Q |> Matrix
basis_augmentation2.p.V .= qr(basis_augmentation2.p.V).Q |> Matrix

system_lr10 = EPMAfem.implicit_midpoint_dlr5(problem, solver=LinearAlgebra.:\, max_ranks=(p=5, m=5));
system_lr10_aug = EPMAfem.implicit_midpoint_dlr5(problem, solver=LinearAlgebra.:\, max_ranks=(p=5, m=5), basis_augmentation=basis_augmentation);
system_lr10_aug2 = EPMAfem.implicit_midpoint_dlr5(problem, solver=LinearAlgebra.:\, max_ranks=(p=5, m=5), basis_augmentation=basis_augmentation2);

system_lr20_aug2 = EPMAfem.implicit_midpoint_dlr5(problem, solver=LinearAlgebra.:\, max_ranks=(p=10, m=10), basis_augmentation=basis_augmentation2);


sol = EPMAfem.IterableDiscretePNSolution(system, source, initial_solution=initial_condition);
sol_lr10 = EPMAfem.IterableDiscretePNSolution(system_lr10, source, initial_solution=initial_condition);
sol_lr10_aug = EPMAfem.IterableDiscretePNSolution(system_lr10_aug, source, initial_solution=initial_condition);
sol_lr10_aug2 = EPMAfem.IterableDiscretePNSolution(system_lr10_aug2, source, initial_solution=initial_condition);
sol_lr20_aug2 = EPMAfem.IterableDiscretePNSolution(system_lr20_aug2, source, initial_solution=initial_condition);

#### GIF
# Ωp, Ωm = EPMAfem.SphericalHarmonicsModels.eval_basis(EPMAfem.direction_model(model), Ω -> 1.0) |> EPMAfem.architecture(problem)
masses = zeros(length(sol), 5)
# momentum = zeros(length(sol), 5)
energies = zeros(length(sol), 5)
outflux = zeros(length(sol), 5)
ranksp = zeros(length(sol), 4)
ranksm = zeros(length(sol), 4)
@gif for (i, ((ϵ, ψ), (ϵ1, ψ1), (ϵ2, ψ2), (ϵ3, ψ3), (ϵ4, ψ4))) in enumerate(zip(sol, sol_lr10, sol_lr10_aug, sol_lr10_aug2, sol_lr20_aug2))
    ψp, ψm = EPMAfem.pmview(ψ, model)
    @show dot(ψp, ψp) + dot(ψm, ψm), dot(ψp, ψp), dot(ψm, ψm)
    ψp1, ψm1 = EPMAfem.pmview(ψ1, model)
    ψp2, ψm2 = EPMAfem.pmview(ψ2, model)
    ψp3, ψm3 = EPMAfem.pmview(ψ3, model)
    ψp4, ψm4 = EPMAfem.pmview(ψ4, model)
    func = EPMAfem.SpaceModels.interpolable((p=collect(ψp*Ωp), m=collect(ψm*Ωm)), EPMAfem.space_model(model))
    func1 = EPMAfem.SpaceModels.interpolable((p=collect(ψp1*Ωp), m=collect(ψm1*Ωm)), EPMAfem.space_model(model))
    func2 = EPMAfem.SpaceModels.interpolable((p=collect(ψp2*Ωp), m=collect(ψm2*Ωm)), EPMAfem.space_model(model))
    func3 = EPMAfem.SpaceModels.interpolable((p=collect(ψp3*Ωp), m=collect(ψm3*Ωm)), EPMAfem.space_model(model))
    func4 = EPMAfem.SpaceModels.interpolable((p=collect(ψp4*Ωp), m=collect(ψm4*Ωm)), EPMAfem.space_model(model))
    plot(-1.5:0.01:1.5, x -> func(VectorValue(x)), label=L"P_{%$(N)}")
    plot!(-1.5:0.01:1.5, x -> func1(VectorValue(x)), label=L"P_{%$(N)} \textrm{(5, default)}")
    plot!(-1.5:0.01:1.5, x -> func2(VectorValue(x)), label=L"P_{%$(N)} \textrm{(5, mass)}")
    plot!(-1.5:0.01:1.5, x -> func3(VectorValue(x)), label=L"P_{%$(N)} \textrm{(5, mass + outflux)}")
    plot!(-1.5:0.01:1.5, x -> func4(VectorValue(x)), label=L"P_{%$(N)} \textrm{(10, mass + outflux)}")
    plot!(size=(400, 300), fontfamily="Computer Modern", dpi=1000, legend=:bottom)
    xlabel!(L"x")
    ylabel!(L"\langle \psi \rangle")
    savefig(joinpath(figpath, "snapshots/t_$(ϵ).png"))
    # heatmap(-1.5:0.01:1.5, -1.5:0.01:1.5, (x, y) -> func(VectorValue(x, y)), aspect_ratio=:equal)

    masses[i, 1] = dot(xp, ψp*Ωp) + dot(xm, ψm*Ωm)
    # momentum[i, 1] = dot(xp, ψp*Ωp_m) + dot(xm, ψm*Ωm_m)
    energies[i, 1] = dot(ψp, Mp*ψp) + dot(ψm, Mm*ψm)
    outflux[i, 1] = dot(xp_b, ψp*Ωp_b)

    masses[i, 2] = dot(xp, ψp1*Ωp) + dot(xm, ψm1*Ωm)
    masses[i, 3] = dot(xp, ψp2*Ωp) + dot(xm, ψm2*Ωm)
    masses[i, 4] = dot(xp, ψp3*Ωp) + dot(xm, ψm3*Ωm)
    masses[i, 5] = dot(xp, ψp4*Ωp) + dot(xm, ψm4*Ωm)
    # momentum[i, 2] = dot(xp, ψp1*Ωp_m) + dot(xm, ψm1*Ωm_m)
    # momentum[i, 3] = dot(xp, ψp2*Ωp_m) + dot(xm, ψm2*Ωm_m)
    # momentum[i, 4] = dot(xp, ψp3*Ωp_m) + dot(xm, ψm3*Ωm_m)
    # momentum[i, 5] = dot(xp, ψp4*Ωp_m) + dot(xm, ψm4*Ωm_m)

    energies[i, 2] = dot(ψp1, Mp*ψp1) + dot(ψm1, Mm*ψm1)
    energies[i, 3] = dot(ψp2, Mp*ψp2) + dot(ψm2, Mm*ψm2)
    energies[i, 4] = dot(ψp3, Mp*ψp3) + dot(ψm3, Mm*ψm3)
    energies[i, 5] = dot(ψp4, Mp*ψp4) + dot(ψm4, Mm*ψm4)
    outflux[i, 2] = dot(xp_b, ψp1*Ωp_b)
    outflux[i, 3] = dot(xp_b, ψp2*Ωp_b)
    outflux[i, 4] = dot(xp_b, ψp3*Ωp_b)
    outflux[i, 5] = dot(xp_b, ψp4*Ωp_b)
end

using Serialization
serialize(joinpath(figpath, "data.jls"), (masses, energies, outflux))


begin
    t = energy_model
    plot(t, energies[:, 1], label=L"P_{%$(N)}")
    plot!(t, energies[:, 2], label=L"P_{%$(N)} \textrm{(default)}")
    plot!(t, energies[:, 3], label=L"P_{%$(N)} \textrm{(mass)}")
    plot!(t, energies[:, 4], label=L"P_{%$(N)} \textrm{(mass + bc)}")

    zoom_range = 20:1:55
    zoom_datarange = (44.30, 44.32)
    plot!([t[first(zoom_range)], t[last(zoom_range)]], [zoom_datarange[1], zoom_datarange[1]], color=:black, label=nothing, linewidth=1)
    plot!([t[first(zoom_range)], t[last(zoom_range)]], [zoom_datarange[2], zoom_datarange[2]], color=:black, label=nothing, linewidth=1)
    plot!([t[first(zoom_range)], t[first(zoom_range)]], [zoom_datarange[1], zoom_datarange[2]], color=:black, label=nothing, linewidth=1)
    plot!([t[last(zoom_range)], t[last(zoom_range)]], [zoom_datarange[1], zoom_datarange[2]], color=:black, label=nothing, linewidth=1)
    plot!(size=(400, 300), dpi=1000, fontfamily="Computer Modern", right_margin=2Plots.mm, legend=:bottomleft)
    xlabel!(L"t")
    ylabel!(L"\mathcal{E}(t)")
    p_i = plot!([t[zoom_range] t[zoom_range] t[zoom_range] t[zoom_range]],
            [energies[zoom_range, 1] energies[zoom_range, 2] energies[zoom_range, 3] energies[zoom_range, 4]], 
            color=[1 2 3 4],
            ls=[:solid, :solid, :solid, :solid],
            label=nothing,
            inset=(1, bbox(0.4, 0.4, 0.5, 0.35, :bottom, :right)), subplot=2, ylims=zoom_datarange, xflip=false, framestyle=:box, tickfontsize=5)
    savefig(joinpath(figpath, "energy.png"))
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
    t = energy_model

    plot(t, abs.(masses[:, 1] .- 1.0), yaxis=:log, label=L"P_{%$(N)}")
    plot!(t, abs.(masses[:, 2] .- 1.0), label=L"P_{%$(N)} \textrm{(default)}")
    plot!(t, abs.(masses[:, 3] .- 1.0), label=L"P_{%$(N)} \textrm{(mass)}")
    plot!(t, abs.(masses[:, 4] .- 1.0), label=L"P_{%$(N)} \textrm{(mass + bc)}")

    plot!(t, abs.(masses[:, 1] .+ cumtrapz(outflux[:, 1], step(energy_model)) .- 1.0), yaxis=:log, color=1, ls=:dash, label=nothing)
    plot!(t, abs.(masses[:, 2] .+ cumtrapz(outflux[:, 2], step(energy_model)) .- 1.0), color=2, ls=:dash, label=nothing)
    plot!(t, abs.(masses[:, 3] .+ cumtrapz(outflux[:, 3], step(energy_model)) .- 1.0), color=3, ls=:dash, label=nothing)
    plot!(t, abs.(masses[:, 4] .+ cumtrapz(outflux[:, 4], step(energy_model)) .- 1.0), color=4, ls=:dash, label=nothing)
    plot!([], [], color=:gray, ls=:dash, label="boundary flux corr.")
    ylims!(1e-16, 1.0)
    yticks!([1, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 1e-16])
    
    plot!(size=(400, 300), dpi=1000, fontfamily="Computer Modern", right_margin=2Plots.mm, legend=:topleft)
    xlabel!(L"t")
    ylabel!(L"|\mathcal{M}(t) - \mathcal{M}(0)|")
    savefig(joinpath(figpath, "mass.png"))
end
