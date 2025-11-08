using Revise
using EPMAfem

using EPMAfem.PNLazyMatrices
using EPMAfem.Krylov
using EPMAfem.Gridap
using LinearAlgebra
using Plots

struct LineSourceEquations <: EPMAfem.AbstractPNEquations end
EPMAfem.number_of_elements(::LineSourceEquations) = 1
EPMAfem.number_of_scatterings(::LineSourceEquations) = 1
EPMAfem.stopping_power(::LineSourceEquations, e, ϵ) = 1.0
EPMAfem.absorption_coefficient(eq::LineSourceEquations, e, ϵ) = 1.0
EPMAfem.scattering_coefficient(eq::LineSourceEquations, e, i, ϵ) = 1.0
EPMAfem.mass_concentrations(::LineSourceEquations, e, x) = 1.0
EPMAfem.scattering_kernel(::LineSourceEquations, e, i) = μ -> 1/(4π)
# equations = EPMAfem.filter_exp(LineSourceEquations(), 0.2, 4)
equations = LineSourceEquations()

space_model = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((-1.5, 1.5, -1.5, 1.5), (150, 150)))
N = 47
direction_model = EPMAfem.SphericalHarmonicsModels.EOSphericalHarmonicsModel(N, 2, :OE)
model = EPMAfem.DiscretePNModel(space_model, 0:0.01:1.0, direction_model)
arch = EPMAfem.cuda(Float64)
problem = EPMAfem.discretize_problem(equations, model, arch)

# source / boundary condition (here: zero)
nb = EPMAfem.n_basis(model)
source = EPMAfem.Rank1DiscretePNVector(false, model, arch, zeros(nb.nϵ), (p=zeros(nb.nx.p), m=zeros(nb.nx.m)) |> arch, (p=zeros(nb.nΩ.p), m=zeros(nb.nΩ.m)) |> arch)

# initial condition
σ = 0.03
# σ = 0.2
using EPMAfem.HCubature
# init_x(x) = 1/(2π*σ^2)*exp(-(x[1]*x[1]+x[2]*x[2])/(2*σ^2)) # normal gaussian
# init_x(x) = 1/(8π*σ^2)*exp(-(x[1]*x[1]+x[2]*x[2])/(2*σ^2)) # from (https://doi.org/10.1080/00411450.2014.910226)
init_x(x) = 1/(4π*σ^2)*exp(-(x[1]*x[1]+x[2]*x[2])/(4*σ^2)) # from (https://doi.org/10.1051/m2an/2022090)
init_Ω(_) = 1.0
Mp_cpu = EPMAfem.SpaceModels.assemble_bilinear(EPMAfem.SpaceModels.∫R_uv, EPMAfem.space_model(model), EPMAfem.SpaceModels.plus(EPMAfem.space_model(model)), EPMAfem.SpaceModels.plus(EPMAfem.space_model(model)))
# Mp = Mp_cpu |> EPMAfem.architecture(problem)
Mm_cpu = EPMAfem.SpaceModels.assemble_bilinear(EPMAfem.SpaceModels.∫R_uv, EPMAfem.space_model(model), EPMAfem.SpaceModels.minus(EPMAfem.space_model(model)), EPMAfem.SpaceModels.minus(EPMAfem.space_model(model))) |> EPMAfem.diag_if_diag
# Mm = Mm_cpu |> EPMAfem.architecture(problem)

bxp = Mp_cpu \ EPMAfem.SpaceModels.assemble_linear(EPMAfem.SpaceModels.∫R_μv(init_x), EPMAfem.space_model(model), EPMAfem.SpaceModels.plus(EPMAfem.space_model(model)))
bxm = Mm_cpu \ EPMAfem.SpaceModels.assemble_linear(EPMAfem.SpaceModels.∫R_μv(init_x), EPMAfem.space_model(model), EPMAfem.SpaceModels.minus(EPMAfem.space_model(model)))
bΩp = EPMAfem.SphericalHarmonicsModels.assemble_linear(EPMAfem.SphericalHarmonicsModels.∫S²_hv(init_Ω), EPMAfem.direction_model(model), EPMAfem.SphericalHarmonicsModels.plus(EPMAfem.direction_model(model)))
bΩm = EPMAfem.SphericalHarmonicsModels.assemble_linear(EPMAfem.SphericalHarmonicsModels.∫S²_hv(init_Ω), EPMAfem.direction_model(model), EPMAfem.SphericalHarmonicsModels.minus(EPMAfem.direction_model(model)))

nb = EPMAfem.n_basis(problem)
initial_condition = EPMAfem.allocate_vec(arch, nb.nx.p*nb.nΩ.p + nb.nx.m*nb.nΩ.m)
ψ0p, ψ0m = EPMAfem.pmview(initial_condition, model)
copy!(ψ0p, bxp .* bΩp')
copy!(ψ0m, bxm .* bΩm')

# #### GIF
# Ωp, Ωm = EPMAfem.SphericalHarmonicsModels.eval_basis(EPMAfem.direction_model(model), Ω -> 1.0) |> EPMAfem.architecture(problem)
# @gif for (ϵ, ψ) in sol
#     ψp, ψm = EPMAfem.pmview(ψ, model)
#     func = EPMAfem.SpaceModels.interpolable((p=collect(ψp*Ωp), m=collect(ψm*Ωm)), EPMAfem.space_model(model))
#     heatmap(-1.5:0.01:1.5, -1.5:0.01:1.5, (x, y) -> func(VectorValue(x, y)), aspect_ratio=:equal, cmap=:jet)
# end

using Serialization
figpath = mkpath(joinpath(dirname(@__FILE__), "figures/2D_linesource/"))
mkpath(joinpath(dirname(@__FILE__), "figures/2D_linesource/solutions/$(N)"))

function compute_sol_and_rank_evolution(sol)
    data = Dict()
    data["ranks"] = (p=zeros(length(sol)), m=zeros(length(sol)))
    data["mass"] = zeros(length(sol))
    Ωp, Ωm = EPMAfem.SphericalHarmonicsModels.eval_basis(EPMAfem.direction_model(sol.system.problem.model), Ω -> 1.0) |> EPMAfem.architecture(sol.system.problem)
    xp, xm = EPMAfem.SpaceModels.eval_basis(EPMAfem.space_model(sol.system.problem.model), x -> 1.0) |> EPMAfem.architecture(sol.system.problem)

    # pbl = sol.system.problem
    # xp_b = unlazy(pbl.space_discretization.∂p[1], vec_size -> EPMAfem.allocate_vec(EPMAfem.architecture(pbl), vec_size))*(EPMAfem.cu(Mp_cpu \ collect(xp)))
    # Ωp_b = vec(Ωp' * pbl.direction_discretization.absΩp[1])

    for (i, (ϵ, ψ)) in enumerate(sol)
        ψp, ψm = EPMAfem.pmview(ψ, sol.system.problem.model)
        data["mass"][i] = dot(xp, ψp*Ωp) + dot(xm, ψm*Ωm)
        if hasproperty(ψ, :ranks)
            data["ranks"].p[i], data["ranks"].m[i] = ψ.ranks.p[], ψ.ranks.m[]
        end
        @show data["ranks"].p[i], data["ranks"].m[i]
        if i == length(sol)
            ψp, ψm = EPMAfem.pmview(ψ, sol.system.problem.model)
            data["final"] = EPMAfem.SpaceModels.interpolable((p=collect(ψp*Ωp), m=collect(ψm*Ωm)), EPMAfem.space_model(sol.system.problem.model))
        end
    end
    return data
end

system = EPMAfem.implicit_midpoint2(problem, A -> PNLazyMatrices.schur_complement(A, Krylov.minres, PNLazyMatrices.cache ∘ LinearAlgebra.inv!));
sol = EPMAfem.IterableDiscretePNSolution(system, source, initial_solution=initial_condition);
serialize(joinpath(figpath, "solutions/$N/full.jls"), compute_sol_and_rank_evolution(sol))

for (i, (rmax, tol)) in collect(enumerate([(100, 0.05), (200, 0.025), (200, 0.0125), (310, 0.00625)]))
    let
        system_lr = EPMAfem.implicit_midpoint_dlr5(problem, max_ranks=(p=rmax, m=rmax), tolerance=tol);
        sol_lr = EPMAfem.IterableDiscretePNSolution(system_lr, source, initial_solution=initial_condition);
        serialize(joinpath(figpath, "solutions/$(N)/lr_$(i).jls"), compute_sol_and_rank_evolution(sol_lr))
    end
    let
        system_lr_cons = EPMAfem.implicit_midpoint_dlr5(problem, max_ranks=(p=rmax, m=rmax), tolerance=tol, basis_augmentation=:mass);
        sol_lr_cons = EPMAfem.IterableDiscretePNSolution(system_lr_cons, source, initial_solution=initial_condition);
        serialize(joinpath(figpath, "solutions/$(N)/lr_$(i)_cons.jls"), compute_sol_and_rank_evolution(sol_lr_cons))
    end
end

using CSV
using LaTeXStrings

ref_sol = CSV.File(joinpath(figpath, "refPhiFull.txt"), header=0) |> CSV.Tables.matrix
heatmap(range(-1.5, 1.5, 250), range(-1.5, 1.5, 250), ref_sol, aspect_ratio=:equal, cmap=:viridis, clims=(0, 0.4))
plot!(size=(300, 288), fontfamily="Computer Modern", margin=2Plots.mm, dpi=1000)
xlabel!(L"x")
ylabel!(L"y")
savefig(joinpath(figpath, "ref.png"))

path(N, t, ::Val{:def}) = joinpath(figpath, "solutions", "$(N)", t==-1 ? "full.jls" : "lr_$(t).jls")
path(N, t, ::Val{:cons}) = joinpath(figpath, "solutions", "$(N)", t==-1 ? "full.jls" : "lr_$(t)_cons.jls")
tol(t) = Dict(1=>L"\vartheta = 5 \times 10^{-2}", 2=>L"\vartheta = 2.5 \times 10^{-2}", 3=>L"\vartheta = 1.25 \times 10^{-2}", 4=>L"\vartheta = 6.25 \times 10^{-3}")[t]
for N in [47]
    ts =  [-1, 1, 2, 3, 4]
    # xy plots
    for t in ts
        if t == -1
            data = deserialize(path(N, t, Val(:def)))
            heatmap(-1.5:0.01:1.5, -1.5:0.01:1.5, (x, y) -> data["final"](VectorValue(x, y))/4π, aspect_ratio=:equal, cmap=:viridis, clims=(0, 0.4))
            plot!(size=(300, 288), fontfamily="Computer Modern", margin=2Plots.mm, dpi=1000)
            xlabel!(L"x")
            ylabel!(L"y")
            savefig(joinpath(figpath, "final_$(N)_full.png"))
        else
            for cons in [Val(:def), Val(:cons)]
                data = deserialize(path(N, t, cons))
                heatmap(-1.5:0.01:1.5, -1.5:0.01:1.5, (x, y) -> data["final"](VectorValue(x, y))/4π, aspect_ratio=:equal, cmap=:viridis, clims=(0, 0.4))
                plot!(size=(300, 288), fontfamily="Computer Modern", margin=2Plots.mm, dpi=1000)
                xlabel!(L"x")
                ylabel!(L"y")
                savefig(joinpath(figpath, "final_$(N)_$(t)_$(cons).png"))
            end
        end
    end

    # lineouts
    for cons in [Val(:def), Val(:cons)]
        plot(range(-1.5, 1.5, 250), ref_sol[:, 250÷2], color=:gray, ls=:dash, label="reference")
        for t in [-1, 2, 3, 4]
            if t == -1
                data = deserialize(path(N, -1, Val(:def)))
                plot!(range(-1.5, 1.5, 150), x -> data["final"](VectorValue(x, 0.0))/4π, label=L"P_{47}", color=:black)
            else
                data = deserialize(path(N, t, cons))
                plot!(range(-1.5, 1.5, 150), x -> data["final"](VectorValue(x, 0.0))/4π, label=tol(t), color=t-1)
            end
        end
        plot!(size=(400, 300), fontfamily="Computer Modern", dpi=1000)
        savefig(joinpath(figpath, "lineout_$(N)_$(cons).png"))
    end

    # mass evolution
    let
        plot()
        data = deserialize(path(N, -1, Val(:def)))
        plot!(EPMAfem.energy_model(model), data["mass"]./4π, color=:black, label=L"P_{47}", ls=:solid, linewidth=2)
        # default
        # data = deserialize(path(N, 1, Val(:def)))
        # plot!(EPMAfem.energy_model(model), data["mass"], label=L"r^+ (\vartheta=5 \times 10^{-2})", color=1, ls=:solid)

        data = deserialize(path(N, 2, Val(:def)))
        plot!(EPMAfem.energy_model(model), data["mass"]./4π, label=L"\vartheta=2.5 \times 10^{-2}", color=1, ls=:solid)

        data = deserialize(path(N, 3, Val(:def)))
        plot!(EPMAfem.energy_model(model), data["mass"]./4π, label=L"\vartheta=1.25 \times 10^{-2}", color=2, ls=:solid)

        data = deserialize(path(N, 4, Val(:def)))
        plot!(EPMAfem.energy_model(model), data["mass"]./4π, label=L"\vartheta=6.25 \times 10^{-3}", color=3, ls=:solid)

        # conservative
        # data = deserialize(path(N, 1, Val(:cons)))
        # plot!(EPMAfem.energy_model(model), data["mass"], label=nothing, color=1, ls=:dash)

        data = deserialize(path(N, 2, Val(:cons)))
        plot!(EPMAfem.energy_model(model), data["mass"]./4π, label=nothing, color=1, ls=:dash)

        data = deserialize(path(N, 3, Val(:cons)))
        plot!(EPMAfem.energy_model(model), data["mass"]./4π, label=nothing, color=2, ls=:dash)

        data = deserialize(path(N, 4, Val(:cons)))
        plot!(EPMAfem.energy_model(model), data["mass"]./4π, label=nothing, color=3, ls=:dash)

        plot!([], [], ls=:dash, color=:gray, label=L"(\textrm{mass \, conservative})")
        xlabel!(L"t")
        ylabel!("mass")
        ylims!(0.98, 1.001)
        plot!(fontfamily="Computer Modern", size=(400, 300), dpi=1000)
        savefig(joinpath(figpath, "mass.png"))
    end

    # rank evolution
    let
        plot()
        # default
        data = deserialize(path(N, 1, Val(:def)))
        # plot!(EPMAfem.energy_model(model), data["ranks"].m, label=nothing, color=1, ls=:solid, alpha=0.5)
        # plot!(EPMAfem.energy_model(model), data["ranks"].p, label=L"r^+ (\vartheta=5 \times 10^{-2})", color=1, ls=:solid)

        data = deserialize(path(N, 2, Val(:def)))
        plot!(EPMAfem.energy_model(model), data["ranks"].m, label=nothing, color=1, ls=:solid, alpha=0.5)
        plot!(EPMAfem.energy_model(model), data["ranks"].p, label=L"r^+ (\vartheta=2.5 \times 10^{-2})", color=1, ls=:solid)

        data = deserialize(path(N, 3, Val(:def)))
        plot!(EPMAfem.energy_model(model), data["ranks"].m, label=nothing, color=2, ls=:solid, alpha=0.5)
        plot!(EPMAfem.energy_model(model), data["ranks"].p, label=L"r^+ (\vartheta=1.25 \times 10^{-2})", color=2, ls=:solid)

        data = deserialize(path(N, 4, Val(:def)))
        plot!(EPMAfem.energy_model(model), data["ranks"].m, label=nothing, color=3, ls=:solid, alpha=0.5)
        plot!(EPMAfem.energy_model(model), data["ranks"].p, label=L"r^+ (\vartheta=6.25 \times 10^{-3})", color=3, ls=:solid)

        # conservative
        data = deserialize(path(N, 1, Val(:cons)))
        # plot!(EPMAfem.energy_model(model), data["ranks"].m, label=nothing, color=1, ls=:dash, alpha=0.5)
        # plot!(EPMAfem.energy_model(model), data["ranks"].p, label=nothing, color=1, ls=:dash)

        data = deserialize(path(N, 2, Val(:cons)))
        plot!(EPMAfem.energy_model(model), data["ranks"].m, label=nothing, color=1, ls=:dash, alpha=0.5)
        plot!(EPMAfem.energy_model(model), data["ranks"].p, label=nothing, color=1, ls=:dash)

        data = deserialize(path(N, 3, Val(:cons)))
        plot!(EPMAfem.energy_model(model), data["ranks"].m, label=nothing, color=2, ls=:dash, alpha=0.5)
        plot!(EPMAfem.energy_model(model), data["ranks"].p, label=nothing, color=2, ls=:dash)

        data = deserialize(path(N, 4, Val(:cons)))
        plot!(EPMAfem.energy_model(model), data["ranks"].m, label=nothing, color=3, ls=:dash, alpha=0.5)
        plot!(EPMAfem.energy_model(model), data["ranks"].p, label=nothing, color=3, ls=:dash)

        plot!([], [], ls=:solid, alpha=0.5, color=:gray, label=L"r^- (\textrm{transparent})")
        plot!([], [], ls=:dash, color=:gray, label=L"(\textrm{mass \, conservative})")
        xlabel!(L"t")
        ylabel!("rank")
        plot!(fontfamily="Computer Modern", size=(400, 300), dpi=1000)
        savefig(joinpath(figpath, "ranks.png"))
    end

end
