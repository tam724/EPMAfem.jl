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

space_model = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((-1.5, 1.5, -1.5, 1.5), (200, 200)))
N = 47
direction_model = EPMAfem.SphericalHarmonicsModels.EOSphericalHarmonicsModel(N, 2)
model = EPMAfem.DiscretePNModel(space_model, 0:0.01:1.0, direction_model)
problem = EPMAfem.discretize_problem(equations, model, EPMAfem.cuda())

# source / boundary condition (here: zero)
source = EPMAfem.Rank1DiscretePNVector(false, model, EPMAfem.cuda(), zeros(EPMAfem.n_basis(model).nϵ), zeros(EPMAfem.n_basis(model).nx.p), zeros(EPMAfem.n_basis(model).nΩ.p))

# initial condition
σ = 0.03
# σ = 0.2
using EPMAfem.HCubature
# init_x(x) = 1/(2π*σ^2)*exp(-(x[1]*x[1]+x[2]*x[2])/(2*σ^2)) #normal gaussian
# init_x(x) = 1/(8π*σ^2)*exp(-(x[1]*x[1]+x[2]*x[2])/(2*σ^2)) # from (https://doi.org/10.1080/00411450.2014.910226)
init_x(x) = 1/(4π*σ^2)*exp(-(x[1]*x[1]+x[2]*x[2])/(4*σ^2)) # from (https://doi.org/10.1051/m2an/2022090)
init_Ω(_) = 1.0
Mp_cpu = EPMAfem.SpaceModels.assemble_bilinear(EPMAfem.SpaceModels.∫R_uv, EPMAfem.space_model(model), EPMAfem.SpaceModels.even(EPMAfem.space_model(model)), EPMAfem.SpaceModels.even(EPMAfem.space_model(model)))
# Mp = Mp_cpu |> EPMAfem.architecture(problem)
Mm_cpu = EPMAfem.SpaceModels.assemble_bilinear(EPMAfem.SpaceModels.∫R_uv, EPMAfem.space_model(model), EPMAfem.SpaceModels.odd(EPMAfem.space_model(model)), EPMAfem.SpaceModels.odd(EPMAfem.space_model(model))) |> EPMAfem.diag_if_diag
# Mm = Mm_cpu |> EPMAfem.architecture(problem)

bxp = Mp_cpu \ EPMAfem.SpaceModels.assemble_linear(EPMAfem.SpaceModels.∫R_μv(init_x), EPMAfem.space_model(model), EPMAfem.SpaceModels.even(EPMAfem.space_model(model)))
bxm = Mm_cpu \ EPMAfem.SpaceModels.assemble_linear(EPMAfem.SpaceModels.∫R_μv(init_x), EPMAfem.space_model(model), EPMAfem.SpaceModels.odd(EPMAfem.space_model(model)))
bΩp = EPMAfem.SphericalHarmonicsModels.assemble_linear(EPMAfem.SphericalHarmonicsModels.∫S²_hv(init_Ω), EPMAfem.direction_model(model), EPMAfem.SphericalHarmonicsModels.even(EPMAfem.direction_model(model)))
bΩm = EPMAfem.SphericalHarmonicsModels.assemble_linear(EPMAfem.SphericalHarmonicsModels.∫S²_hv(init_Ω), EPMAfem.direction_model(model), EPMAfem.SphericalHarmonicsModels.odd(EPMAfem.direction_model(model)))

nb = EPMAfem.n_basis(problem)
initial_condition = EPMAfem.allocate_vec(EPMAfem.cuda(), nb.nx.p*nb.nΩ.p + nb.nx.m*nb.nΩ.m)
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

probe = EPMAfem.PNProbe(model, EPMAfem.cuda(), ϵ=0.0, Ω=Ω->1.0)

system = EPMAfem.implicit_midpoint2(problem, A -> PNLazyMatrices.schur_complement(A, Krylov.minres, PNLazyMatrices.cache ∘ LinearAlgebra.inv!));
sol = EPMAfem.IterableDiscretePNSolution(system, source, initial_solution=initial_condition);
serialize(joinpath(figpath, "solutions/$N/full.jls"), compute_sol_and_rank_evolution(sol))

function compute_sol_and_rank_evolution(sol)
    data = Dict()
    data["ranks"] = (p=zeros(length(sol)), m=zeros(length(sol)))
    data["mass"] = zeros(length(sol))
    Ωp, Ωm = EPMAfem.SphericalHarmonicsModels.eval_basis(EPMAfem.direction_model(sol.system.problem.model), Ω -> 1.0) |> EPMAfem.architecture(sol.system.problem)
    xp, xm = EPMAfem.SpaceModels.eval_basis(EPMAfem.space_model(sol.system.problem.model), x -> 1.0) |> EPMAfem.architecture(sol.system.problem)
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
        GC.gc(true)
    end
    return data
end

for (i, (rmax, tol)) in collect(enumerate([(100, 0.25), (200, 0.025), (200, 0.0125), (400, 0.00625)]))
    system_lr = EPMAfem.implicit_midpoint_dlr5(problem, max_ranks=(p=300, m=300), tolerance=tol);
    sol_lr = EPMAfem.IterableDiscretePNSolution(system_lr, source, initial_solution=initial_condition);
    serialize(joinpath(figpath, "solutions/$(N)/lr_$(i).jls"), compute_sol_and_rank_evolution(sol_lr))

    system_lr_cons = EPMAfem.implicit_midpoint_dlr5(problem, max_ranks=(p=300, m=300), tolerance=tol, basis_augmentation=:mass);
    sol_lr_cons = EPMAfem.IterableDiscretePNSolution(system_lr_cons, source, initial_solution=initial_condition);
    serialize(joinpath(figpath, "solutions/$(N)/lr_$(i)_cons.jls"), compute_sol_and_rank_evolution(sol_lr_cons))
end

using CSV
ref_sol = CSV.File(joinpath(figpath, "refPhiFull.txt"), header=0) |> CSV.Tables.matrix
heatmap(range(-1.5, 1.5, 250), range(-1.5, 1.5, 250), ref_sol, aspect_ratio=:equal, cmap=:viridis, clims=(0, 0.4))
plot!(size=(300, 288), fontfamily="Computer Modern", margin=2Plots.mm, dpi=1000)
xlabel!(L"x")
ylabel!(L"y")
savefig(joinpath(figpath, "ref.png"))

using LaTeXStrings
path(N, t, ::Val{:def}) = joinpath(figpath, "solutions", "$(N)", t==-1 ? "full.jls" : "lr_$(t).jls")
path(N, t, ::Val{:cons}) = joinpath(figpath, "solutions", "$(N)", t==-1 ? "full.jls" : "lr_$(t)_cons.jls")
tol(t) = Dict(2=>L"5 \times 10^{-2}", 3=>L"2.5 \times 10^{-2}", 4=>L"1.25 \times 10^{-2}", 5=>L"6.25 \times 10^{-3}")[t]
for N in [47]
    ts =  [-1, 2, 3, 4]
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

    # plot()
    # for t in ts
    #     if t == -1
    #         data = deserialize(path(N, t, Val(:cons)))
    #         plot!(data["mass"], label="$(t)_full", color=:black)
    #     else
    #         for cons in [Val(:def), Val(:cons)]
    #             data = deserialize(path(N, t, cons))
    #             plot!(data["mass"], label="$(t)_$(cons)", color=t, ls=(cons == Val(:def)) ? :solid : :dash)
    #         end
    #     end
    # end

    plot()
    for t in ts
        if t == -1
        else
            for cons in [Val(:def), Val(:cons)]
                data = deserialize(path(N, t, cons))
                plot!(data["ranks"].p, label="$(t)_$(cons)", color=t, ls=(cons == Val(:def)) ? :solid : :dash)
                plot!(data["ranks"].m, label="$(t)_$(cons)", color=t, ls=(cons == Val(:def)) ? :dot : :dot)
            end
        end
    end





    # plot(range(-1.5, 1.5, 250), ref_sol[250÷2, :], color=:black, label="reference", linewidth=2, ls=:dot)
    # for (i, t) in enumerate(ts)
    #     data = deserialize(path(N, t))
    #     plot!(-1.5:0.01:1.5, x -> data["final"](VectorValue(x, 0.0))/4π, color=i, label=t==-1 ? L"\textrm{full}" : L"\textrm{tol}="*"$(tol(t))")
    # end
    # ylabel!(L"\langle \psi \rangle")
    # xlabel!(L"x")
    # plot!(size=(400, 300), fontfamily="Computer Modern", margin=2Plots.mm,  dpi=1000, legend=:topright)
    # savefig(joinpath(figpath, "lineout_$(N).png"))

    # plot()
    # for (i, t) in enumerate(ts)
    #     if t == -1 continue end
    #     data = deserialize(path(N, t))
    #     plot!(range(0, 1, 101), data["ranks"].p, color=i, label=t==-1 ? L"\textrm{full}" : L"\textrm{tol}="*"$(tol(t))")
    #     plot!(range(0, 1, 101), data["ranks"].m, ls=:dash, color=i, label=nothing)
    # end
    # ylabel!(L"\textrm{ranks}")
    # xlabel!(L"t")
    # plot!(size=(400, 300), fontfamily="Computer Modern", margin=2Plots.mm,  dpi=1000)
    # savefig(joinpath(figpath, "ranks_$(N).png"))
end
