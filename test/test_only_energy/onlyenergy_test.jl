
module OnlyEnergyTests

using EPMAfem

using EPMAfem: PNArchitecture, energy_model, mass_concentrations, stopping_power, absorption_coefficient, scattering_coefficient, number_of_scatterings, number_of_elements, base_type
using EPMAfem: DiscretePNProblem, Rank1DiscretePNVector, discretize_rhs, discretize_problem, implicit_midpoint, implicit_midpoint2, implicit_midpoint_dlr

using SpecialFunctions
using Plots
using LinearAlgebra
using Interpolations
using HCubature
using Test
using OrdinaryDiffEq
using ConcreteStructs
using Distributions

include("onlyenergy_model.jl")

function compute(N, eq, solver, use_adjoint, arch)
    model = OnlyEnergyModel(range(0.0, 1.0, length=N))
    discrete_problem = discretize_problem(eq, model, arch)
    if solver == "schur_old"
        discrete_system = implicit_midpoint(discrete_problem, EPMAfem.PNSchurSolver)
    elseif solver == "full_old"
        discrete_system = implicit_midpoint(discrete_problem, EPMAfem.PNKrylovMinresSolver)
    elseif solver == "schur"
        discrete_system = implicit_midpoint2(discrete_problem, (EPMAfem.PNLazyMatrices.schur_complement, EPMAfem.Krylov.minres))
    elseif solver == "full"
        discrete_system = implicit_midpoint2(discrete_problem, EPMAfem.Krylov.minres)
    elseif solver == "dlr"
        discrete_system = implicit_midpoint_dlr(discrete_problem; max_rank=1)
    else
        throw(ArgumentError("solver must be 'schur' or 'full' or schur_old or full_old"))
    end
    if !use_adjoint
        discrete_ext = discretize_rhs(eq, model, arch)
    else
        discrete_ext = discretize_adjoint_rhs(eq, model, arch)
    end

    if !use_adjoint
        A = discrete_system * discrete_ext
    else
        A = adjoint(discrete_system) * discrete_ext
    end

    sol = zeros(length(energy_model(model)))
    ϵs = zeros(length(energy_model(model)))

    for (idx, ψ) in A
        ψp, ψm = EPMAfem.pmview(ψ, model)
        full_sol = ψp |> collect
        sol[idx.i] = only(ψp |> collect)
        @assert isapprox(only(ψm |> collect), 0.0; atol=1e-4)
        ϵs[idx.i] = EPMAfem.ϵ(idx)
    end
    return ϵs, sol
end

function extend_solution(ϵs, sol)
    # extend the solutions and energy interval to the full range
    sol = [sol..., sol[end]]
    Δϵ = ϵs[2] - ϵs[1]
    ϵs = [ϵs..., ϵs[end] + Δϵ]
    return ϵs, sol
end

function compute_L2(N, eq, solver, use_adjoint, arch)
    ϵs, sol = compute(N, eq, solver, use_adjoint, arch)
    if use_adjoint
        ϵs, sol = extend_solution(ϵs, sol)
    end
    interpol_sol = Interpolations.interpolate((ϵs,), sol, Gridded(Linear()))
    L2_error = hquadrature(ϵ-> (interpol_sol(ϵ) - exact_solution(eq, use_adjoint)(ϵ))^2, 0.0, 1.0)[1]
    return L2_error
end

function subsample(ϵs, fac::Integer)
    return range(ϵs[1], ϵs[end], length=length(ϵs)*fac)
end

function rand_eq(n_elem, n_scat, k_is_one=false)
    ρ = rand(Dirichlet(n_elem, 1.0))
    return OnlyEnergyEquations((α = rand()*300+100, β=rand()*0.8+0.1, ρ=ρ, s=rand(n_elem), τ=10*randn(n_elem), τ2=rand(n_elem), σ=randn(n_elem, n_scat), σ2=rand(n_elem, n_scat), k=k_is_one ? ones(n_elem, n_scat) : rand(n_elem, n_scat)))
end

function rand_eq_with_analytic_solution()
    n_elem = 1
    n_scat = 1
    return OnlyEnergyEquations((α=rand()*300+100, β=rand()*0.8+0.1, ρ=ones(n_elem), s=zeros(n_elem), τ=zeros(n_elem), τ2=10*rand(n_elem), σ=zeros(n_elem, n_scat), σ2=zeros(n_elem, n_scat), k=ones(n_elem, n_scat)))
end

function test_against_analytic_solution(produce_plots::Bool, plotpath=nothing)
    n_sols = 10
    equations = [rand_eq_with_analytic_solution() for _ in 1:n_sols]
    for use_adjoint in [true, false]
        for solver in ["full", "schur", "dlr", "full_old", "schur_old"]
            if solver == "dlr" && use_adjoint == true continue end # skip for now
            for arch in [EPMAfem.cpu(Float64), EPMAfem.cpu(Float32), EPMAfem.cuda(Float64), EPMAfem.cuda(Float32)]
                if produce_plots plot() end
                for i_sols in 1:n_sols
                    eq = equations[i_sols]
                    ϵs, sol = compute(100, eq, solver, use_adjoint, arch)
                    exact_sol = exact_solution(eq, use_adjoint)
                    if produce_plots 
                        scatter!(ϵs, sol, color=i_sols, label=nothing)
                        plot!(subsample(ϵs, 3), exact_sol, label=nothing, color=i_sols)
                    end
                    
                    N_test = 10
                    test_is = rand(1:length(ϵs), N_test)
                    test_ϵs = ϵs[test_is]
                    @test all([isapprox(sol[test_is[i]], exact_sol(test_ϵs[i]); atol=1e-2, rtol=1e-2) for i in 1:N_test])
                    for i in 1:N_test
                        if !isapprox(sol[test_is[i]], exact_sol(test_ϵs[i]); atol=1e-2, rtol=1e-2)
                            diff = sol[test_is[i]] - exact_sol(test_ϵs[i])
                            @warn "Equation: $(eq), Adjoint: $use_adjoint, Solver: $solver, Architecture: $arch - Solution at index $(test_is[i]) (ϵ = $(test_ϵs[i])) is not correct: difference = $diff"
                        end
                    end
                end
                if produce_plots savefig(joinpath(plotpath, "only_energy_comparison_analytic_$(use_adjoint)_$(solver)_$(arch).png")) end
            end
        end
    end
end

function test_against_diffeq(produce_plots, plotpath=nothing)
    n_sols = 10
    equations = [rand_eq(5, 5) for _ in 1:n_sols]
    for use_adjoint in [true, false]
        for solver in ["full", "schur", "dlr", "full_old", "schur_old"]
            if solver == "dlr" && use_adjoint == true continue end # skip for now
            for arch in [EPMAfem.cpu(Float64), EPMAfem.cpu(Float32), EPMAfem.cuda(Float64), EPMAfem.cuda(Float32)]
                if produce_plots plot() end
                for i_sols in 1:n_sols
                    eq = equations[i_sols]
                    ϵs, sol = compute(100, eq, solver, use_adjoint, arch)
                    diffeq_sol = diffeq_solution(eq, use_adjoint)
                    if produce_plots 
                        scatter!(ϵs, sol, color=i_sols, label=nothing)
                        plot!(subsample(ϵs, 3), diffeq_sol, label=nothing, color=i_sols)
                    end
                    
                    N_test = 10
                    test_is = rand(1:length(ϵs), N_test)
                    test_ϵs = ϵs[test_is]
                    @test all([isapprox(sol[test_is[i]], diffeq_sol(test_ϵs[i]); atol=1e-3, rtol=1e-2) for i in 1:N_test])
                    for i in 1:N_test
                        if !isapprox(sol[test_is[i]], diffeq_sol(test_ϵs[i]); atol=1e-3, rtol=1e-2)
                            diff = sol[test_is[i]] - diffeq_sol(test_ϵs[i])
                            @warn "Equation: $(eq), Adjoint: $use_adjoint, Solver: $solver, Architecture: $arch - Solution at index $(test_is[i]) (ϵ = $(test_ϵs[i])) is not correct: difference = $diff"
                        end
                    end
                end
                if produce_plots savefig(joinpath(plotpath, "only_energy_comparison_diffeq_$(use_adjoint)_$(solver)_$(arch).png")) end
            end
        end
    end
    return nothing
end

plotpath = joinpath(@__DIR__, "EPMAfem_testplots/only_energy/")

test_against_analytic_solution(false, mkpath(joinpath(plotpath, "analytic/")))
test_against_diffeq(false, mkpath(joinpath(plotpath, "diffeq/")))

### to play with this: 
# use_adjoint = false
# eqs = [rand_eq_with_analytic_solution() for _ in 1:10]
# plot()
# for i in 1:10
#     eq = eqs[i]
#     ϵs, sol = compute(100, eq, "schur", use_adjoint, EPMAfem.cpu())
#     plot!(subsample(ϵs, 10), exact_solution(eq, use_adjoint), color=i)
#     scatter!(ϵs, sol, color=i, ls=:dot)
# end
# plot!()

# use_adjoint = false
# eqs = [rand_eq(5, 5) for _ in 1:10]
# plot()
# for i in 1:10
#     eq = eqs[i]
#     ϵs, sol = compute(100, eq, "schur", use_adjoint, EPMAfem.cpu())
#     diffeq_sol = diffeq_solution(eq, use_adjoint)

#     plot!(subsample(ϵs, 10), diffeq_sol, color=i, label="$(i)")
#     plot!(ϵs, sol, color=i, ls=:dot, label="$(i)")
# end
# plot!()


### convergence plots
# eq = rand_eq_with_analytic_solution()
# use_adjoint = false
# ϵs, sol = compute(50, eq, "full", use_adjoint, EPMAfem.cpu())
# sol_diffeq = diffeq_solution(eq, use_adjoint)
# sol_exact = exact_solution(eq, use_adjoint)
# scatter(ϵs, sol)
# plot!(subsample(ϵs, 10), sol_diffeq)
# plot!(subsample(ϵs, 10), sol_exact)

# Ns = [8, 16, 32, 64, 128, 256, 512, 1024]
# L2_schur = map(N -> compute_L2(N, eq, "schur", use_adjoint, EPMAfem.cpu()), Ns)
# L2_full = map(N -> compute_L2(N, eq, "full", use_adjoint, EPMAfem.cpu()), Ns)

# scatter(Ns, L2_full, xaxis=:log, yaxis=:log, label="L2 error (full solver)")
# scatter!(Ns, L2_schur, xaxis=:log, yaxis=:log, label="L2 error (schur solver)")
# plot!(Ns, 1 ./ Ns .^ 3, color=:gray, ls=:dot, label="N^-3")
# plot!(Ns, 1 ./ Ns .^ 4, color=:gray, ls=:dash, label="N^-4")
# plot!(Ns, 1 ./ Ns .^ 5, color=:gray, ls=:dashdot, label="N^-5")
# xlabel!("N")
# ylabel!("L2_error")
end
