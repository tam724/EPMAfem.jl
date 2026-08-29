using Revise
using EPMAfem
using Distributions
using SpecialFunctions
using EPMAfem.Gridap
using Plots
using LinearAlgebra
using EPMAfem.PNLazyMatrices
using EPMAfem.Krylov
using LaTeXStrings
include("scripts/grid_gen.jl")

struct TestEquations <: EPMAfem.AbstractPNEquations end
EPMAfem.number_of_elements(::TestEquations) = 1
EPMAfem.number_of_scatterings(::TestEquations) = 1
EPMAfem.stopping_power(::TestEquations, e, ϵ) = 1.0
EPMAfem.absorption_coefficient(eq::TestEquations, e, ϵ) = 0.0
EPMAfem.scattering_coefficient(eq::TestEquations, e, i, ϵ) = 0.0
function EPMAfem.mass_concentrations(::TestEquations, e, x)
    1.0
end

vmf_normalization(p, κ) = κ^(p/2-1)/((2π)^(p/2)*besseli(p/2-1, κ))
EPMAfem.scattering_kernel(::TestEquations, e, i) = μ -> 0.0 # vmf_normalization(2, 5.0)*exp(5.0*μ)

# forward_plots
energy_model = range(0, 1, 300)
equations = TestEquations()

using SpecialFunctions
σ = 0.1
init_x(x) = 1/(σ^2*2π)*exp(-1/2*((x[1])^2 + (x[2])^2)/σ^2)
init_Ω(Ω) = 1.0 # pdf(VonMisesFisher([1, 0, 0], 2.0), [Ω...])

function analytic_solution(x, t)
    r2 = x[1]^2 + x[2]^2
    return 2π/(σ^2*2π) *
           exp(-(r2+t^2)/(2σ^2)) *
           besseli(0, sqrt(r2)*t/σ^2)
end
heatmap(-1.5:0.01:1.5, -1.5:0.01:1.5, (x, y) -> analytic_solution(VectorValue(x, y), 1.0), aspect_ratio=:equal)

L2 = Dict()
L2_rel = Dict()
n_dof = Dict()
n_cells = Dict()
sols = Dict()
energy = Dict()

problem_cache = nothing
sol_cache = nothing

# grid_gen_2D((-1.5, 1.5, -1.5, 1.5); min_res=0.02, max_res=0.02, filepath="/tmp/tmp_msh.msh")
for mesh in [:structured, :unstructured][2:2]
    for EO in [:EO, :OE][2:2]
        L2[(mesh, EO)] = zeros(4)
        N_structured = [20, 40, 80, 160]
        res_unstructured = [0.25, 0.12, 0.06, 0.026]
        n_cells[(mesh, EO)] = zeros(Int64, 4)
        n_dof[(mesh, EO)] = zeros(Int64, 4)
        for (i, n_res) in collect(enumerate(mesh == :structured ? N_structured : res_unstructured))[1:1]
            if mesh == :structured
                space_model = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((-1.5, 1.5, -1.5, 1.5), (n_res, n_res)))
            else
                grid_gen_2D((-1.5, 1.5, -1.5, 1.5); min_res=n_res, max_res=n_res, filepath="/tmp/tmp_msh.msh")
                space_model = EPMAfem.SpaceModels.GridapSpaceModel(DiscreteModelFromFile("/tmp/tmp_msh.msh"))
            end
            n_cells[(mesh, EO)][i] = Gridap.num_cells(space_model.discrete_model.grid)
            if EO == :EO
                n_dof[(mesh, EO)][i] = EPMAfem.SpaceModels.n_basis(space_model).p
            else
                n_dof[(mesh, EO)][i] = EPMAfem.SpaceModels.n_basis(space_model).m
            end
            direction_model = EPMAfem.SphericalHarmonicsModels.EOCircularHarmonicsModel(3, 2, EO)
            model = EPMAfem.DiscretePNModel(space_model, energy_model, direction_model)

            problem = EPMAfem.discretize_problem(equations, model, EPMAfem.cpu())


            rhs = EPMAfem.Rank1DiscretePNVector(false, model, EPMAfem.cpu(), zeros(EPMAfem.n_basis(model).nϵ), (p=zeros(EPMAfem.n_basis(model).nx.p), m=zeros(EPMAfem.n_basis(model).nx.m)), (p=zeros(EPMAfem.n_basis(model).nΩ.p), m=zeros(EPMAfem.n_basis(model).nΩ.m)))
            # system = EPMAfem.implicit_midpoint2(problem, A -> PNLazyMatrices.schur_complement(A, Krylov.minres(; atol=1e-12, rtol=1e-12), PNLazyMatrices.cache ∘ LinearAlgebra.inv!));
            system = EPMAfem.implicit_midpoint2(problem, Krylov.minres(; atol=1e-15, rtol=1e-15));

            # initial condition
            Mp = EPMAfem.SpaceModels.assemble_bilinear(EPMAfem.SpaceModels.∫R_uv, EPMAfem.space_model(model), EPMAfem.SpaceModels.plus(EPMAfem.space_model(model)), EPMAfem.SpaceModels.plus(EPMAfem.space_model(model))) |> EPMAfem.architecture(problem)
            Mm = EPMAfem.SpaceModels.assemble_bilinear(EPMAfem.SpaceModels.∫R_uv, EPMAfem.space_model(model), EPMAfem.SpaceModels.minus(EPMAfem.space_model(model)), EPMAfem.SpaceModels.minus(EPMAfem.space_model(model))) |> EPMAfem.architecture(problem)
            bxp = Mp \ EPMAfem.SpaceModels.assemble_linear(EPMAfem.SpaceModels.∫R_μv(init_x), EPMAfem.space_model(model), EPMAfem.SpaceModels.plus(EPMAfem.space_model(model)))
            bxm = Mm \ EPMAfem.SpaceModels.assemble_linear(EPMAfem.SpaceModels.∫R_μv(init_x), EPMAfem.space_model(model), EPMAfem.SpaceModels.minus(EPMAfem.space_model(model)))
            bΩp = EPMAfem.SphericalHarmonicsModels.assemble_linear(EPMAfem.SphericalHarmonicsModels.∫S²_hv(init_Ω), EPMAfem.direction_model(model), EPMAfem.SphericalHarmonicsModels.plus(EPMAfem.direction_model(model)))
            bΩm = EPMAfem.SphericalHarmonicsModels.assemble_linear(EPMAfem.SphericalHarmonicsModels.∫S²_hv(init_Ω), EPMAfem.direction_model(model), EPMAfem.SphericalHarmonicsModels.minus(EPMAfem.direction_model(model)))

            if EO == :EO
                bΩm .= 0.0
                bxm .= 0.0
                bΩp[1] = sqrt(2π)
                bΩp[2:end] .= 0.0
            else
                bΩp .= 0.0
                bxp .= 0.0
                bΩm[1] = sqrt(2π)
                bΩm[2:end] .= 0.0
            end
            # @show rhs.bΩ
            # @show bΩm
            # continue

            initial_condition = EPMAfem.allocate_solution_vector(system)
            ψ0p, ψ0m = EPMAfem.pmview(initial_condition, model)
            copy!(ψ0p, bxp .* bΩp')
            copy!(ψ0m, bxm .* bΩm')

            problem_cache = (system, initial_condition)

            # problem_cache = (Mm, bxp, bxm, bΩp, bΩm, initial_condition)
            # continue


            sol = EPMAfem.IterableDiscretePNSolution(system, rhs, initial_solution=initial_condition);

            # energy[(mesh, EO, n_res)] = zeros(length(energy_model))
            # for (i, (ϵ, ψ)) in enumerate(sol)
            #     ψp, ψm = EPMAfem.pmview(ψ, model)
            #     energy[(mesh, EO, n_res)][i] = dot(Mp*ψp, ψp) + dot(Mm*ψm, ψm)
            # end
            # for l2 error ...
            # probe = EPMAfem.PNProbe(model, EPMAfem.cpu(); ϵ=0.0, Ω=Ω->1.0)
            
            
            # b_integr = EPMAfem.initialize_integration(probe)
            # b_integr.cache.bases.Ω.p .= 0.0
            # b_integr.cache.bases.Ω.m[1] = sqrt(2π)
            # b_integr.cache.bases.Ω.m[2:end] .= 0.0
            # for (idx, ψ) in sol
            #     if EPMAfem.is_first(idx) continue end # (where ψ is initialized to 0 anyways..)
            #     b_integr(idx, ψ)
            # end
            # sol_integral = EPMAfem.finalize_integration(b_integr)
            sol_final = nothing
            # i = 0
            for (ϵ, ψ) in sol
                sol_final = ψ
                # break
                # ψp, ψm = EPMAfem.pmview(sol_final, model)
                # i += 1
                # if i == 3
                #     break
                # end
                # xx = EPMAfem.SpaceModels.interpolable((p=ψp*bΩp.*0.0, m=ψm*bΩm), EPMAfem.space_model(model))
                # p1 = heatmap(-1.5:0.01:1.5, -1.5:0.01:1.5, (x, y) -> xx(VectorValue(x, y)), aspect_ratio=:equal, clims=(-4, 4), cmap=:jet)
                # display(p1)
            end
            ψp, ψm = EPMAfem.pmview(sol_final, model)
            sol_cache = sol_final, model
            xx = EPMAfem.SpaceModels.interpolable((p=ψp*bΩp.*0.0, m=ψm*bΩm), EPMAfem.space_model(model))

            dx = space_model.args[2]
            sols[(mesh, EO, n_res)] = xx
            L2[(mesh, EO)][i] = sqrt(sum(∫((xx.interp.uh - (x -> analytic_solution(x, 1.0)))*(xx.interp.uh - (x -> analytic_solution(x, 1.0))))*dx))
        end
    end
end

begin
    plot()
    for (mesh, EO, n_res) in keys(energy)
        plot!(energy_model, energy[(mesh, EO, n_res)] ./ energy[(mesh, EO, n_res)][1])
    end
    ylims!(0.99, 1.01)
    plot!()
end

# [0.25, 0.12, 0.06, 0.026]
# grid_gen_2D((-1.5, 1.5, -1.5, 1.5); min_res=0.06, max_res=0.06, filepath="/tmp/tmp_msh.msh")

begin
    for (mesh, EO, n_res) in keys(sols)
        # if n_res != 80 && n_res != 0.06 continue end
        # if n_res == 80 || n_res == 0.06
            xx = sols[(mesh, EO, n_res)]
            plot!(size=(315, 300), dpi=1000, fontfamily="Computer Modern")
            display(plot!())
            #savefig("convergence_EO_mesh/solution_$(mesh)_$(EO)_$(n_res).png")
        # end
    end
end

ψp, ψm = EPMAfem.pmview(sol_cache[1], sol_cache[2])
xx = EPMAfem.SpaceModels.interpolable((p=ψp[:, 4]*0, m=ψm[:, 4]), EPMAfem.space_model(sol_cache[2]))
p1 = heatmap(-1.5:0.01:1.5, -1.5:0.01:1.5, (x, y) -> xx(VectorValue(x, y)), aspect_ratio=:equal, clims=(-4, 4), cmap=:jet)

begin
    L2_ref = sqrt(hcubature(((x, y), ) -> init_x(VectorValue(x, y))^2, [-1.5, -1.5], [1.5, 1.5])[1])
    plot(xaxis=:log, yaxis=:log)
    plot!(n_cells[(:structured, :OE)], L2[(:structured, :OE)]/L2_ref, color=1, marker=false, label=L"structured, $L^2-H^1$")
    scatter!(n_cells[(:structured, :OE)], L2[(:structured, :OE)]/L2_ref, color=1, marker=:o, label=nothing)
    plot!(n_cells[(:structured, :EO)], L2[(:structured, :EO)]/L2_ref, color=1, ls=:dash, marker=false, label=L"structured, $H^1-L^2$")
    scatter!(n_cells[(:structured, :EO)], L2[(:structured, :EO)]/L2_ref, color=1, ls=:dash, marker=:o, label=nothing)
    plot!(n_cells[(:unstructured, :OE)], L2[(:unstructured, :OE)]/L2_ref, color=2, marker=false, label=L"unstructured, $L^2-H^1$")
    scatter!(n_cells[(:unstructured, :OE)], L2[(:unstructured, :OE)]/L2_ref, color=2, marker=:o, label=nothing)
    plot!(n_cells[(:unstructured, :EO)], L2[(:unstructured, :EO)]/L2_ref, color=2, ls=:dash, marker=:false, label=L"unstructured, $H^1-L^2$")
    scatter!(n_cells[(:unstructured, :EO)], L2[(:unstructured, :EO)]/L2_ref, color=2, ls=:dash, marker=:o, label=nothing)

    plot!(3e2:5e4, x->200/x, ls=:dash, color=:gray, label=nothing)
    annotate!(2e3, 5.2e-2, Plots.text(L"\mathcal{O}(1/N)", 9, :gray), color=:gray)
    plot!(3e2:5e4, x->50/(x)^(1/2), ls=:dash, color=:gray, label=nothing)
    annotate!(8e3, 1, Plots.text(L"\mathcal{O}(1/\sqrt{N})", 9, :gray), color=:gray)
    plot!(size=(400, 300), dpi=1000, fontfamily="Computer Modern", legend=:bottomleft)
    xlabel!("number of grid cells")
    ylabel!(L"relative $L^2$ error")
    savefig("convergence_EO_mesh/rel_L2_error_n_cells.png")
end

begin
    L2_ref = sqrt(hcubature(((x, y), ) -> init_x(VectorValue(x, y))^2, [-1.5, -1.5], [1.5, 1.5])[1])
    plot(xaxis=:log, yaxis=:log)
    plot!(n_dof[(:structured, :OE)], L2[(:structured, :OE)]/L2_ref, color=1, marker=false, label=L"structured, $L^2-H^1$")
    scatter!(n_dof[(:structured, :OE)], L2[(:structured, :OE)]/L2_ref, color=1, marker=:o, label=nothing)
    plot!(n_dof[(:structured, :EO)], L2[(:structured, :EO)]/L2_ref, color=1, ls=:dash, marker=false, label=L"structured, $H^1-L^2$")
    scatter!(n_dof[(:structured, :EO)], L2[(:structured, :EO)]/L2_ref, color=1, ls=:dash, marker=:o, label=nothing)
    plot!(n_dof[(:unstructured, :OE)], L2[(:unstructured, :OE)]/L2_ref, color=2, marker=false, label=L"unstructured, $L^2-H^1$")
    scatter!(n_dof[(:unstructured, :OE)], L2[(:unstructured, :OE)]/L2_ref, color=2, marker=:o, label=nothing)
    plot!(n_dof[(:unstructured, :EO)], L2[(:unstructured, :EO)]/L2_ref, color=2, ls=:dash, marker=:false, label=L"unstructured, $H^1-L^2$")
    scatter!(n_dof[(:unstructured, :EO)], L2[(:unstructured, :EO)]/L2_ref, color=2, ls=:dash, marker=:o, label=nothing)

    plot!(3e2:5e4, x->200/x, ls=:dash, color=:gray, label=nothing)
    annotate!(2e3, 5.2e-2, Plots.text(L"\mathcal{O}(1/N)", 9, :gray), color=:gray)
    plot!(3e2:5e4, x->50/(x)^(1/2), ls=:dash, color=:gray, label=nothing)
    annotate!(8e3, 1, Plots.text(L"\mathcal{O}(1/\sqrt{N})", 9, :gray), color=:gray)
    plot!(size=(400, 300), dpi=1000, fontfamily="Computer Modern", legend=:bottomleft)
    xlabel!("number of dof")
    ylabel!(L"relative $L^2$ error")
    savefig("convergence_EO_mesh/rel_L2_error_n_dof.png")
end
