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
EPMAfem.scattering_kernel(::TestEquations, e, i) = μ -> vmf_normalization(2, 5.0)*exp(5.0*μ)

# forward_plots
equations = TestEquations()

N = [2, 4, 8, 16, 32, 64, 128, 265, 512]
scalar_function = zeros(2, 2, length(N))
for (i, n_x) in collect(enumerate(N))[end:end], (j, eo) in enumerate([:EO, :OE]), (k, ord) in enumerate([(p=1, m=0), (p=2, m=1)])
    energy_model = range(0, 1, 2000)
    space_model = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((0.0, 1.0), (n_x)), plus=(name=lagrangian, order=ord.p, conformity=:H1), minus=(name=lagrangian, order=ord.m, conformity=:L2))
    direction_model = EPMAfem.SphericalHarmonicsModels.EOCircularHarmonicsModel(39, 1, eo)
    model = EPMAfem.DiscretePNModel(space_model, energy_model, direction_model)

    # source computation
    rhs = let
        T = EPMAfem.base_type(EPMAfem.cpu())
        SM = EPMAfem.SpaceModels
        SH = EPMAfem.SphericalHarmonicsModels
        μϵ = Vector{T}([exp(-500*(ϵ-0.8)^2) for ϵ ∈ EPMAfem.energy_model(model)])
        μΩ = (p=SH.assemble_linear(SH.∫S²_hv(Ω -> -1.0), direction_model, SH.plus(direction_model)), m=SH.assemble_linear(SH.∫S²_hv(Ω -> -1.0), direction_model, SH.minus(direction_model))) |> EPMAfem.cpu()
        μx = (p=SM.assemble_linear(SM.∫R_μv(x -> exp(-500*(x[1]-0.7)^2)), space_model, SM.plus(space_model)), m=SM.assemble_linear(SM.∫R_μv(x -> exp(-500*(x[1]-0.7)^2)), space_model, SM.minus(space_model))) |> EPMAfem.cpu()
        [EPMAfem.Rank1DiscretePNVector(false, model, EPMAfem.cpu(), μϵ, μx, μΩ)]
    end

    # extraction computation
    extr = let
        T = EPMAfem.base_type(EPMAfem.cpu())
        SM = EPMAfem.SpaceModels
        SH = EPMAfem.SphericalHarmonicsModels
        μϵ = Vector{T}([exp(-500*(ϵ-0.2)^2) for ϵ ∈ EPMAfem.energy_model(model)])
        μΩ = (p=SH.assemble_linear(SH.∫S²_hv(Ω -> 1.0), direction_model, SH.plus(direction_model)), m=SH.assemble_linear(SH.∫S²_hv(Ω -> 1.0), direction_model, SH.minus(direction_model))) |> EPMAfem.cpu()
        μx = (p=SM.assemble_linear(SM.∫R_μv(x -> exp(-500*(x[1]-0.3)^2)), space_model, SM.plus(space_model)), m=SM.assemble_linear(SM.∫R_μv(x -> exp(-500*(x[1]-0.3)^2)), space_model, SM.minus(space_model))) |> EPMAfem.cpu()
        [EPMAfem.Rank1DiscretePNVector(true, model, EPMAfem.cpu(), μϵ, μx, μΩ)]
    end

    problem = EPMAfem.discretize_problem(equations, model, EPMAfem.cpu())
    # system = EPMAfem.implicit_midpoint2(problem, A -> PNLazyMatrices.schur_complement(A, Krylov.minres(; rtol=1e-13, atol=1e-13), PNLazyMatrices.cache ∘ LinearAlgebra.inv!));
    system = EPMAfem.implicit_midpoint2(problem, Krylov.minres(; rtol=1e-13, atol=1e-13));

    scalar_function[j, k, i] = (extr*system*rhs)[1]
    if j == 2
        scalar_function[j, k, i] *= -1
    end
end

scalar_function[2, :, :] *= -1 # weird...

begin
    ref = scalar_function[:, :, end]
    ref .= scalar_function[2, 2, end]
    xx = 1
    plot(N[1:end-xx], abs.((scalar_function[1, 1, 1:end-xx] .- ref[1, 1]) ./ ref[1, 1]), xaxis=:log, yaxis=:log, label=L"(e: $H^1$, p=1), (o: $L^2$, p=0)", marker=:o)
    plot!(N[1:end-xx], abs.((scalar_function[2, 1, 1:end-xx] .- ref[2, 1]) ./ ref[2, 1]), xaxis=:log, yaxis=:log, label=L"(o: $H^1$, p=1), (e: $L^2$, p=0)", marker=:o)
    plot!(N[1:end-xx], abs.((scalar_function[1, 2, 1:end-xx] .- ref[1, 2]) ./ ref[1, 2]), xaxis=:log, yaxis=:log, label=L"(e: $H^1$, p=2), (o: $L^2$, p=1)", marker=:o)
    plot!(N[1:end-xx], abs.((scalar_function[2, 2, 1:end-xx] .- ref[2, 2]) ./ ref[2, 2]), xaxis=:log, yaxis=:log, label=L"(o: $H^1$, p=2), (e: $L^2$, p=1)", marker=:o)
    # plot!(N[1:end-1], 1.0./N[1:end-1], color=:gray, label="first order")
    plot!(N[1:end-1], 30.0./(N[1:end-1]).^2, color=:lightgray, ls=:solid, label=nothing)
    annotate!(100, 1e-2, Plots.text(L"\mathcal{O}(1/N^2)", :black, 8))
    plot!(N[1:end-1], 300.0./(N[1:end-1]).^4, color=:lightgray, ls=:dash, label=nothing)
    annotate!(100, 3e-5, Plots.text(L"\mathcal{O}(1/N^4)", :black, 8))
    yticks!([1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7])
    ylabel!("relative functional error")
    xlabel!("number of cells")
    ylims!(1e-8, 1e1)
    plot!(size=(400, 300), dpi=1000, fontfamily="Computer Modern", legend=:bottomleft)
    savefig("scripts_2D_comparison_convergence/1D/convergence.png")
    plot!()
    # xlims!()
end

# vizualization

energy_model = range(0, 1, 200)
space_model = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((0.0, 1.0), (50)), plus=(name=lagrangian, order=2, conformity=:H1), minus=(name=lagrangian, order=1, conformity=:L2))
direction_model = EPMAfem.SphericalHarmonicsModels.EOCircularHarmonicsModel(39, 1, :OE)
model = EPMAfem.DiscretePNModel(space_model, energy_model, direction_model)

# source computation
rhs = let
    T = EPMAfem.base_type(EPMAfem.cpu())
    SM = EPMAfem.SpaceModels
    SH = EPMAfem.SphericalHarmonicsModels
    μϵ = Vector{T}([exp(-500*(ϵ-0.8)^2) for ϵ ∈ EPMAfem.energy_model(model)])
    μΩ = (p=SH.assemble_linear(SH.∫S²_hv(Ω -> -1.0), direction_model, SH.plus(direction_model)), m=SH.assemble_linear(SH.∫S²_hv(Ω -> -1.0), direction_model, SH.minus(direction_model))) |> EPMAfem.cpu()
    μx = (p=SM.assemble_linear(SM.∫R_μv(x -> exp(-500*(x[1]-0.7)^2)), space_model, SM.plus(space_model)), m=SM.assemble_linear(SM.∫R_μv(x -> exp(-500*(x[1]-0.7)^2)), space_model, SM.minus(space_model))) |> EPMAfem.cpu()
    [EPMAfem.Rank1DiscretePNVector(false, model, EPMAfem.cpu(), μϵ, μx, μΩ)]
end

# extraction computation
extr = let
    T = EPMAfem.base_type(EPMAfem.cpu())
    SM = EPMAfem.SpaceModels
    SH = EPMAfem.SphericalHarmonicsModels
    μϵ = Vector{T}([exp(-500*(ϵ-0.2)^2) for ϵ ∈ EPMAfem.energy_model(model)])
    μΩ = (p=SH.assemble_linear(SH.∫S²_hv(Ω -> 1.0), direction_model, SH.plus(direction_model)), m=SH.assemble_linear(SH.∫S²_hv(Ω -> 1.0), direction_model, SH.minus(direction_model))) |> EPMAfem.cpu()
    μx = (p=SM.assemble_linear(SM.∫R_μv(x -> exp(-500*(x[1]-0.3)^2)), space_model, SM.plus(space_model)), m=SM.assemble_linear(SM.∫R_μv(x -> exp(-500*(x[1]-0.3)^2)), space_model, SM.minus(space_model))) |> EPMAfem.cpu()
    [EPMAfem.Rank1DiscretePNVector(true, model, EPMAfem.cpu(), μϵ, μx, μΩ)]
end

problem = EPMAfem.discretize_problem(equations, model, EPMAfem.cpu())
system = EPMAfem.implicit_midpoint2(problem, A -> PNLazyMatrices.schur_complement(A, Krylov.minres(; rtol=1e-13, atol=1e-13), PNLazyMatrices.cache ∘ LinearAlgebra.inv!));

probe1 = EPMAfem.PNProbe(model, EPMAfem.cpu(); ϵ=0.8, Ω=Ω->1.0)
f1 = EPMAfem.interpolable(probe1, system*rhs[1])
probe2 = EPMAfem.PNProbe(model, EPMAfem.cpu(); ϵ=0.5, Ω=Ω->1.0)
f2 = EPMAfem.interpolable(probe2, system*rhs[1])
probe3 = EPMAfem.PNProbe(model, EPMAfem.cpu(); ϵ=0.2, Ω=Ω->1.0)
f3 = EPMAfem.interpolable(probe3, system*rhs[1])
begin
    plot(0:0.002:1, x -> -f1(VectorValue(x)), color=1, ls=:dot, label=L"sol at $t = 0.2$")
    plot!(0:0.002:1, x -> -f2(VectorValue(x)), color=1, ls=:dash, label=L"sol at $t = 0.5$")
    plot!(0:0.002:1, x -> -f3(VectorValue(x)), color=1, ls=:solid, label=L"sol at $t = 0.8$")
    plot!(0:0.002:1, x -> 0.1*exp(-500*(x-0.3)^2)*exp(-500*(0.2-0.2)^2), color=:gray, ls=:solid, label=L"$h^{\textrm{vol}}(t=0.8)$ (scaled)")
    plot!(size=(400, 300), dpi=1000, fontfamily="Computer Modern", legend=:topleft)
    xlabel!("z")
    savefig("scripts_2D_comparison_convergence/1D/sol_visualization.png")
end


number_of_minres_iterations = Dict()
time_of_iterative_solver = Dict()
val = Dict()
total_time = Dict()
setting = "equal_error"
setting = "equal_dof"


for (i, ord) in enumerate([(p=1, m=0), (p=2, m=1)])
    energy_model = range(0, 1, 200)
    if setting  == "equal_error"
        # to equalize the errors for this testcase (goal: 4e-4), got p=1/0: 3.6e-4, and with p=2/1: 3.9e-4
        # the number of basis functions is: p=1/0: (p=501, m=500), p=2/1: (p=101, m=100)
        n_x = i == 1 ? 500 : 50
    else
        @assert setting == "equal_dof"
        # to equalize the ndof for this testcase, got p=1/0: 1e-3, and with p=2/1: 3.9e-4
        # the number of basis functions is: p=1/0: (p=101, m=100), p=2/1: (p=101, m=100)
        n_x = i == 1 ? 100 : 50
    end
    space_model = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((0.0, 1.0), (n_x)), plus=(name=lagrangian, order=ord.p, conformity=:H1), minus=(name=lagrangian, order=ord.m, conformity=:L2))
    direction_model = EPMAfem.SphericalHarmonicsModels.EOCircularHarmonicsModel(39, 1, :OE)
    model = EPMAfem.DiscretePNModel(space_model, energy_model, direction_model)

    # source computation
    rhs_ = let
        T = EPMAfem.base_type(EPMAfem.cpu())
        SM = EPMAfem.SpaceModels
        SH = EPMAfem.SphericalHarmonicsModels
        μϵ = Vector{T}([exp(-500*(ϵ-0.8)^2) for ϵ ∈ EPMAfem.energy_model(model)])
        μΩ = (p=SH.assemble_linear(SH.∫S²_hv(Ω -> -1.0), direction_model, SH.plus(direction_model)), m=SH.assemble_linear(SH.∫S²_hv(Ω -> -1.0), direction_model, SH.minus(direction_model))) |> EPMAfem.cpu()
        μx = (p=SM.assemble_linear(SM.∫R_μv(x -> exp(-500*(x[1]-0.7)^2)), space_model, SM.plus(space_model)), m=SM.assemble_linear(SM.∫R_μv(x -> exp(-500*(x[1]-0.7)^2)), space_model, SM.minus(space_model))) |> EPMAfem.cpu()
        [EPMAfem.Rank1DiscretePNVector(false, model, EPMAfem.cpu(), μϵ, μx, μΩ)]
    end

    # extraction computation
    extr_ = let
        T = EPMAfem.base_type(EPMAfem.cpu())
        SM = EPMAfem.SpaceModels
        SH = EPMAfem.SphericalHarmonicsModels
        μϵ = Vector{T}([exp(-500*(ϵ-0.2)^2) for ϵ ∈ EPMAfem.energy_model(model)])
        μΩ = (p=SH.assemble_linear(SH.∫S²_hv(Ω -> 1.0), direction_model, SH.plus(direction_model)), m=SH.assemble_linear(SH.∫S²_hv(Ω -> 1.0), direction_model, SH.minus(direction_model))) |> EPMAfem.cpu()
        μx = (p=SM.assemble_linear(SM.∫R_μv(x -> exp(-500*(x[1]-0.3)^2)), space_model, SM.plus(space_model)), m=SM.assemble_linear(SM.∫R_μv(x -> exp(-500*(x[1]-0.3)^2)), space_model, SM.minus(space_model))) |> EPMAfem.cpu()
        [EPMAfem.Rank1DiscretePNVector(true, model, EPMAfem.cpu(), μϵ, μx, μΩ)]
    end

    number_of_minres_iterations[(i, 1)] = zeros(length(energy_model)-1)
    time_of_iterative_solver[(i, 1)] = zeros(length(energy_model)-1)
    number_of_minres_iterations[(i, 2)] = zeros(length(energy_model)-1)
    time_of_iterative_solver[(i, 2)] = zeros(length(energy_model)-1)

    problem = EPMAfem.discretize_problem(equations, model, EPMAfem.cpu())

    system1 = EPMAfem.implicit_midpoint2(problem, A -> PNLazyMatrices.schur_complement(A, Krylov.minres(; rtol=1e-13, atol=1e-13), PNLazyMatrices.cache ∘ LinearAlgebra.inv!));
    empty!(EPMAfem.PNLazyMatrices.global_stats)
    sol = EPMAfem.IterableDiscretePNSolution(system1, rhs_[1]);
    GC.gc()
    total_time[(i, 1)] = @elapsed begin val[(i, 1)] = extr_*(sol) end
    GC.gc()
    for (k, stat) in enumerate(EPMAfem.PNLazyMatrices.global_stats)
        number_of_minres_iterations[(i, 1)][k] = stat.niter
        time_of_iterative_solver[(i, 1)][k] = stat.timer
    end

    # system 2
    system2 = EPMAfem.implicit_midpoint2(problem, Krylov.minres(; rtol=1e-13, atol=1e-13));
    empty!(EPMAfem.PNLazyMatrices.global_stats)
    sol = EPMAfem.IterableDiscretePNSolution(system2, rhs_[1]);
    GC.gc()
    total_time[(i, 2)] = @elapsed begin val[(i, 2)] = extr_*(sol) end
    GC.gc()
    for (k, stat) in enumerate(EPMAfem.PNLazyMatrices.global_stats)
        number_of_minres_iterations[(i, 2)][k] = stat.niter
        time_of_iterative_solver[(i, 2)][k] = stat.timer
    end
end

reference_result = 0.00018605290332009911
for (key, res) in val
    @show key
    @show abs(((-res[1]) - reference_result)/reference_result)
end

begin
    plot()
    plot!(number_of_minres_iterations[(1, 2)], label="(p=1/0)", color=1, ls=:dot)
    plot!(number_of_minres_iterations[(1, 1)], label="(p=1/0), schur", color=1, ls=:solid)
    plot!(number_of_minres_iterations[(2, 2)], label="(p=2/1)", color=2, ls=:dot)
    plot!(number_of_minres_iterations[(2, 1)], label="(p=2/1), schur", color=2, ls=:solid)
    xlabel!("time steps")
    ylabel!("n minres iter. per step")
    plot!(size=(400, 300), fontfamily="Computer Modern", dpi=1000, legend=:topleft)
    ylims!(-1, 80)
    savefig("scripts_2D_comparison_convergence/1D/$(setting)/n_iter_per_step.png")
    plot!()
end

begin
    plot()
    plot!(time_of_iterative_solver[(1, 2)], label="(p=1/0)", color=1, ls=:dot)
    plot!(time_of_iterative_solver[(1, 1)], label="(p=1/0), schur", color=1, ls=:solid)
    plot!(time_of_iterative_solver[(2, 2)], label="(p=2/1)", color=2, ls=:dot)
    plot!(time_of_iterative_solver[(2, 1)], label="(p=2/1), schur", color=2, ls=:solid)
    xlabel!("time steps")
    ylabel!("time per step [s]")
    plot!(size=(400, 300), fontfamily="Computer Modern", dpi=1000, legend=:topleft)
    ylims!(-0.001, 0.08)
    savefig("scripts_2D_comparison_convergence/1D/$(setting)/time_per_step.png")
    plot!()
end
