using Revise
using EPMAfem

using EPMAfem.PNLazyMatrices
using EPMAfem.Krylov
using EPMAfem.Gridap
using LinearAlgebra
using Plots
using LaTeXStrings
using BenchmarkTools
# include("plot_overloads.jl")
using NeXLCore
using Unitful
using Plots
NExt = Base.get_extension(EPMAfem, :NeXLCoreExt)

figpath = mkpath(joinpath(dirname(@__FILE__), "figures/2D_epma"))

equations = NExt.epma_equations(
    [n"Al", n"Cr"],
    [NExt.EPMADetector(n"Al K-L2", VectorValue(1.0, 0.0, 0.0)), NExt.EPMADetector(n"Cr K-L2", VectorValue(1.0, 0.0, 0.0))],
    range(50u"eV", 20u"keV", length=100), 33)
timings = []
plotdata = Dict()
Ns = [1, 3, 5, 7, 9, 11, 13, 15, 21, 27]
ranks = [3, 5, 10, 15]
excitation = EPMAfem.pn_excitation([(x=0.0, y=0.0)], [NExt.dimless(15u"keV", equations.dim_basis)], [VectorValue(-1.0, 0.0, 0.0)], beam_position_σ=0.02, beam_energy_σ=0.05, beam_direction_κ=50)

function mass_concentrations(e, x_)
    z = NExt.dimful(x_[1], u"nm", equations.dim_basis)
    x = NExt.dimful(x_[2], u"nm", equations.dim_basis)

    if (x - 80u"nm")^2 + (z - (-200u"nm"))^2 < (80u"nm")^2
        return e == 1 ? 0.0 : NExt.dimless(n"Cr".density, equations.dim_basis)
    else
        return e == 1 ? NExt.dimless(n"Al".density, equations.dim_basis) : 0.0
    end
end

arch = EPMAfem.cuda(Float64)
step_callback(ϵ, ψ) = @show ϵ
get_model(N) = NExt.epma_model(equations, (-1500u"nm", 0.0u"nm", -1000u"nm", 1000u"nm"), (150, 200), N)
for N in Ns
    model = get_model(N)
    problem = EPMAfem.discretize_problem(equations, model, arch, updatable=true)

    discrete_rhs = EPMAfem.discretize_rhs(excitation, model, arch)[1]

    ρs = EPMAfem.discretize_mass_concentrations([x -> mass_concentrations(e, x) for e in 1:EPMAfem.number_of_elements(equations)], model)
    EPMAfem.update_problem!(problem, ρs)

    probe = EPMAfem.PNProbe(model, arch, Ω=Ω -> 1.0, ϵ=ϵ -> 1.0)
    system_full = EPMAfem.implicit_midpoint2(problem.problem, A -> PNLazyMatrices.schur_complement(A, Krylov.minres, PNLazyMatrices.cache ∘ LinearAlgebra.inv!))
    sol_full = EPMAfem.IterableDiscretePNSolution(system_full, discrete_rhs, step_callback=step_callback)
    # run first 2 iterations (warmup)
    iterate(sol_full, iterate(sol_full)[2])
    # run the full simulation
    time_full = @elapsed func_full = EPMAfem.interpolable(probe, sol_full)

    @show time_full
    push!(timings, ((N, Inf), time_full))
    plotdata[(N, Inf)] = func_full

    for rank in ranks
        @show (N, rank)
        if 2 * rank > EPMAfem.n_basis(problem.problem).nΩ.p || rank > EPMAfem.n_basis(problem.problem).nΩ.p
            continue
        end
        system_lowrank = EPMAfem.implicit_midpoint_dlr5(problem.problem, max_ranks=(p=rank, m=rank), basis_augmentation=:mass)
        sol_lowrank = EPMAfem.IterableDiscretePNSolution(system_lowrank, discrete_rhs, step_callback=step_callback)
        # run first 2 iterations (warmup)
        iterate(sol_lowrank, iterate(sol_lowrank)[2])
        # run the full simulation
        time_lowrank = @elapsed func_lowrank = EPMAfem.interpolable(probe, EPMAfem.IterableDiscretePNSolution(system_lowrank, discrete_rhs, step_callback=step_callback))
        @show N, rank, time_lowrank
        push!(timings, ((N, rank), time_lowrank))
        plotdata[(N, rank)] = func_lowrank
    end
end

timings_noschur = []
for N in Ns
    model = get_model(N)
    problem = EPMAfem.discretize_problem(equations, model, arch, updatable=true)
    discrete_rhs = EPMAfem.discretize_rhs(excitation, model, arch)[1]
    ρs = EPMAfem.discretize_mass_concentrations([x -> mass_concentrations(e, x) for e in 1:EPMAfem.number_of_elements(equations)], model)
    EPMAfem.update_problem!(problem, ρs)

    probe = EPMAfem.PNProbe(model, arch, Ω=Ω -> 1.0, ϵ=ϵ -> 1.0)
    system_full_noschur = EPMAfem.implicit_midpoint2(problem.problem, Krylov.minres)
    sol_full = EPMAfem.IterableDiscretePNSolution(system_full_noschur, discrete_rhs, step_callback=step_callback)
    # run first 2 iterations (warmup)
    iterate(sol_full, iterate(sol_full)[2])
    # run the full simulation
    time_full = @elapsed EPMAfem.interpolable(probe, sol_full)
    @show time_full
    push!(timings_noschur, ((N, Inf), time_full))
end

using Serialization
serialize(joinpath(figpath, "plotdata.jls"), plotdata)
serialize(joinpath(figpath, "timings.jls"), timings)
serialize(joinpath(figpath, "timings_noschur.jls"), timings_noschur)

plotdata = deserialize(joinpath(figpath, "plotdata.jls"))
timings = deserialize(joinpath(figpath, "timings.jls"))
timings_noschur = deserialize(joinpath(figpath, "timings_noschur.jls"))

neg_to_nan(x) = x < 0 ? NaN : x
for N in Ns
    contourf(
        -1000u"nm":1u"nm":1000u"nm",
        -1500u"nm":1u"nm":0u"nm",
        (x, z) -> plotdata[(N, Inf)](Gridap.Point(NExt.dimless.((z, x), Ref(equations.dim_basis)...))) |> neg_to_nan, aspect_ratio=:equal, linewidth=0, clims=(0.0, 0.08), colorbar=nothing)
        xlabel!("")
        ylabel!("")
        plot!(size=(400, 300), dpi=1000, fontfamily="Computer Modern", right_margin=2Plots.mm)
        savefig(joinpath(figpath, "N$(N)_rankinf.png"))
    for rank in ranks
        if haskey(plotdata, (N, rank))
            contourf(
                -1000u"nm":1u"nm":1000u"nm",
                -1500u"nm":1u"nm":0u"nm",
                (x, z) -> plotdata[(N, rank)](Gridap.Point(NExt.dimless.((z, x), Ref(equations.dim_basis))...)) |> neg_to_nan, aspect_ratio=:equal, linewidth=0, clims=(0.0, 0.08), colorbar=nothing)
            xlabel!("")
            ylabel!("")
            plot!(size=(400, 300), dpi=1000, fontfamily="Computer Modern", right_margin=2Plots.mm)
            savefig(joinpath(figpath, "N$(N)_rank$(rank).png"))
        end
    end
end

begin
    # data = [plotdata[(27, Inf)](Gridap.Point(NExt.dimless.((z, x), Ref(equations.dim_basis)...))) for z in -1500u"nm":1u"nm":0u"nm", x in -1000u"nm":1u"nm":1000u"nm"]
    contourf(
        -1000:1:1000,
        -1500:1:0,
        data / maximum(data), aspect_ratio=:equal, linewidth=0)
    xlabel!(L"x \, \textrm{[nm]}")
    ylabel!(L"z \, \textrm{[nm]}")
    plot!(1e7.*(8e-6 .+ 8e-6*sin.(0:0.01:2π)), 1e7.*(-2e-5 .+ 8e-6*cos.(0:0.01:2π)), color=:darkgray, label=nothing, ls=:dot, linewidth=1)
    plot!(size=(400, 300), dpi=1000, fontfamily="Computer Modern", right_margin=2Plots.mm)
    savefig(joinpath(figpath, "PN_results.png"))
end


function compute_rel_L(p1, p_ref, p)
    @assert maximum(abs.((p1.interp.args[1].free_values))) <= 1e-15 
    values_1 = p1.interp.args[2].free_values
    values_ref = p_ref.interp.args[2].free_values
    return norm(values_1 .- values_ref, p) / norm(values_ref, p)
end

model = get_model(3)
ρs = EPMAfem.discretize_mass_concentrations([x -> mass_concentrations(e, x) for e in 1:EPMAfem.number_of_elements(equations)], model)

function compute_rel_M(p1, p_ref)
    @assert maximum(abs.((p1.interp.args[1].free_values))) <= 1e-15
    values_1 = p1.interp.args[2].free_values .* ρs[2, :]
    values_ref = p_ref.interp.args[2].free_values .* ρs[2, :]
    return (abs(sum(values_1) - sum(values_ref))) / sum(values_ref)
end

function memory_req(N, r)
    model = get_model(N)
    (ne, (nxp, nxm), (nΩp, nΩm)) = EPMAfem.n_basis(model)
    if isinf(r)
        return ne*(nxp*nΩp + nxm*nΩm)
    else
        return ne*(nxp*r + r*r + r*nΩp + nxm*r + r*r + r*nΩm)
    end
end

timings2 = Dict()
for (x, t) in timings timings2[x] = t end
for ((N, _), t) in timings_noschur timings2[(N, -1)] = t end

L1_norm = Dict()
Linf_norm = Dict()
L2_norm = Dict()
M_norm = Dict()
for k in keys(plotdata)
    L1_norm[k] = compute_rel_L(plotdata[k], plotdata[(27, Inf)], 1)
    Linf_norm[k] = compute_rel_L(plotdata[k], plotdata[(27, Inf)], Inf)
    L2_norm[k] = compute_rel_L(plotdata[k], plotdata[(27, Inf)], 2)
    M_norm[k] = compute_rel_M(plotdata[k], plotdata[(27, Inf)])
end
Ns2 = Ns[1:end-1]

m_size(N) = 5*N^(1/4)

begin
    plot()
    for (i, n) in enumerate(Ns)
        if !(n == 27)
            scatter!([timings2[(n, Inf)]], [L2_norm[(n, Inf)]], markersize=[m_size(n)], color=1, label=(i==length(Ns2) ? L"P_N" : nothing), text= Plots.text(L"P_{%$(n)}", 4), alpha=0.8)
            if !(n==21)
                scatter!([timings2[(n, -1)]], [L2_norm[(n, Inf)]], markersize=[m_size(n)], color=1, label=(i==length(Ns2)-1 ? L"P_N (noschur)" : nothing), text= Plots.text(L"P_{%$(n)}", 4), alpha=0.3)
            end
        end
        if haskey(timings2, (n, 3))
            scatter!([get(timings2, (n, 3), NaN)], [get(L2_norm, (n, 3), NaN)], markersize=[m_size(n)], color=2, label=(i==length(Ns2) ? L"r=3" : nothing), text=Plots.text(L"P_{%$(n)}", 4), alpha=0.8)
        end
        if haskey(timings2, (n, 5))
            scatter!([get(timings2, (n, 5), NaN)], [get(L2_norm, (n, 5), NaN)], markersize=[m_size(n)], color=3, label=(i==length(Ns2) ? L"r=5" : nothing), text=Plots.text(L"P_{%$(n)}", 4), alpha=0.8)
        end
        if haskey(timings2, (n, 10))
            scatter!([get(timings2, (n, 10), NaN)], [get(L2_norm, (n, 10), NaN)], markersize=[m_size(n)], color=4, label=(i==length(Ns2) ? L"r=10" : nothing), text=Plots.text(L"P_{%$(n)}", 4), alpha=0.8)
        end
        if haskey(timings2, (n, 15))
            scatter!([get(timings2, (n, 15), NaN)], [get(L2_norm, (n, 15), NaN)], markersize=[m_size(n)], color=5, label=(i==length(Ns2) ? L"r=15" : nothing), text=Plots.text(L"P_{%$(n)}", 4), alpha=0.8)
        end
    end
    xlabel!("Runtime (s)", xaxis=:log)
    ylabel!(L"L_2 \textrm{\, error\, (rel.)} ", yaxis=:log)
    plot!(size=(400, 300), fontfamily="Computer Modern", dpi=1000, legend=:bottomleft)
    # savefig(joinpath(figpath, "L2_error.png"))
end

begin
    plot()
    for (i, n) in enumerate(Ns)
        if !(n == 27)
            scatter!([memory_req(n, Inf)], [L2_norm[(n, Inf)]], markersize=[m_size(n)], color=1, label=(i==length(Ns2) ? L"P_N" : nothing), text= Plots.text(L"P_{%$(n)}", 4), alpha=0.8)
        end
        if haskey(timings2, (n, 3))
            scatter!([memory_req(n, 3)], [get(L2_norm, (n, 3), NaN)], markersize=[m_size(n)], color=2, label=(i==length(Ns2) ? L"r=3" : nothing), text=Plots.text(L"P_{%$(n)}", 4), alpha=0.8)
        end
        if haskey(timings2, (n, 5))
            scatter!([memory_req(n, 5)], [get(L2_norm, (n, 5), NaN)], markersize=[m_size(n)], color=3, label=(i==length(Ns2) ? L"r=5" : nothing), text=Plots.text(L"P_{%$(n)}", 4), alpha=0.8)
        end
        if haskey(timings2, (n, 10))
            scatter!([memory_req(n, 10)], [get(L2_norm, (n, 10), NaN)], markersize=[m_size(n)], color=4, label=(i==length(Ns2) ? L"r=10" : nothing), text=Plots.text(L"P_{%$(n)}", 4), alpha=0.8)
        end
        if haskey(timings2, (n, 15))
            scatter!([memory_req(n, 15)], [get(L2_norm, (n, 15), NaN)], markersize=[m_size(n)], color=5, label=(i==length(Ns2) ? L"r=15" : nothing), text=Plots.text(L"P_{%$(n)}", 4), alpha=0.8)
        end
    end
    xlabel!("DOF", xaxis=:log)
    ylabel!(L"L_2 \textrm{\, error\, (rel.)} ", yaxis=:log)
    plot!(size=(400, 300), fontfamily="Computer Modern", dpi=1000)
    savefig(joinpath(figpath, "L2_error_memory.png"))
end

begin
    plot()
    for (i, n) in enumerate(Ns)
        if !(n == 27)
            scatter!([timings2[(n, Inf)]], [L1_norm[(n, Inf)]], markersize=[m_size(n)], color=1, label=(i==length(Ns2) ? L"P_N" : nothing), text=Plots.text(L"P_{%$(n)}", 4), alpha=0.8)
        end
        if haskey(timings2, (n, 3))
            scatter!([get(timings2, (n, 3), NaN)], [get(L1_norm, (n, 3), NaN)], markersize=[m_size(n)], color=2, label=(i==length(Ns2) ? L"r=3" : nothing), text=Plots.text(L"P_{%$(n)}", 4), alpha=0.8)
        end
        if haskey(timings2, (n, 5))
            scatter!([get(timings2, (n, 5), NaN)], [get(L1_norm, (n, 5), NaN)], markersize=[m_size(n)], color=3, label=(i==length(Ns2) ? L"r=5" : nothing), text=Plots.text(L"P_{%$(n)}", 4), alpha=0.8)
        end
        if haskey(timings2, (n, 10))
            scatter!([get(timings2, (n, 10), NaN)], [get(L1_norm, (n, 10), NaN)], markersize=[m_size(n)], color=4, label=(i==length(Ns2) ? L"r=10" : nothing), text=Plots.text(L"P_{%$(n)}", 4), alpha=0.8)
        end
        if haskey(timings2, (n, 15))
            scatter!([get(timings2, (n, 15), NaN)], [get(L1_norm, (n, 15), NaN)], markersize=[m_size(n)], color=5, label=(i==length(Ns2) ? L"r=15" : nothing), text=Plots.text(L"P_{%$(n)}", 4), alpha=0.8)
        end
    end
    xlabel!("Runtime (s)", xaxis=:log)
    ylabel!(L"L_1 \textrm{\, error\, (rel.)} ", yaxis=:log)
    plot!(size=(400, 300), fontfamily="Computer Modern", dpi=1000)
    savefig(joinpath(figpath, "L1_error.png"))
end

begin
    plot()
    for (i, n) in enumerate(Ns)
        if !(n == 27)
            scatter!([memory_req(n, Inf)], [L1_norm[(n, Inf)]], markersize=[m_size(n)], color=1, label=(i==length(Ns2) ? L"P_N" : nothing), text=Plots.text(L"P_{%$(n)}", 4), alpha=0.8)
        end
        if haskey(timings2, (n, 3))
            scatter!([memory_req(n, 3)], [get(L1_norm, (n, 3), NaN)], markersize=[m_size(n)], color=2, label=(i==length(Ns2) ? L"r=3" : nothing), text=Plots.text(L"P_{%$(n)}", 4), alpha=0.8)
        end
        if haskey(timings2, (n, 5))
            scatter!([memory_req(n, 5)], [get(L1_norm, (n, 5), NaN)], markersize=[m_size(n)], color=3, label=(i==length(Ns2) ? L"r=5" : nothing), text=Plots.text(L"P_{%$(n)}", 4), alpha=0.8)
        end
        if haskey(timings2, (n, 10))
            scatter!([memory_req(n, 10)], [get(L1_norm, (n, 10), NaN)], markersize=[m_size(n)], color=4, label=(i==length(Ns2) ? L"r=10" : nothing), text=Plots.text(L"P_{%$(n)}", 4), alpha=0.8)
        end
        if haskey(timings2, (n, 15))
            scatter!([memory_req(n, 15)], [get(L1_norm, (n, 15), NaN)], markersize=[m_size(n)], color=5, label=(i==length(Ns2) ? L"r=15" : nothing), text=Plots.text(L"P_{%$(n)}", 4), alpha=0.8)
        end
    end
    xlabel!("DOF", xaxis=:log)
    ylabel!(L"L_1 \textrm{\, error\, (rel.)} ", yaxis=:log)
    plot!(size=(400, 300), fontfamily="Computer Modern", dpi=1000)
    savefig(joinpath(figpath, "L1_error_memory.png"))
end

begin
    plot()
    for (i, n) in enumerate(Ns)
        if !(n==27)
            scatter!([timings2[(n, Inf)]], [Linf_norm[(n, Inf)]], markersize=[m_size(n)], color=1, label=(i==length(Ns2) ? L"P_N" : nothing), text=Plots.text(L"P_{%$(n)}", 4), alpha=0.8)
        end
        if haskey(timings2, (n, 3))
            scatter!([get(timings2, (n, 3), NaN)], [get(Linf_norm, (n, 3), NaN)], markersize=[m_size(n)], color=2, label=(i==length(Ns2) ? L"r=3" : nothing), text=Plots.text(L"P_{%$(n)}", 4), alpha=0.8)
        end
        if haskey(timings2, (n, 5))
            scatter!([get(timings2, (n, 5), NaN)], [get(Linf_norm, (n, 5), NaN)], markersize=[m_size(n)], color=3, label=(i==length(Ns2) ? L"r=5" : nothing), text=Plots.text(L"P_{%$(n)}", 4), alpha=0.8)
        end
        if haskey(timings2, (n, 10))
            scatter!([get(timings2, (n, 10), NaN)], [get(Linf_norm, (n, 10), NaN)], markersize=[m_size(n)], color=4, label=(i==length(Ns2) ? L"r=10" : nothing), text=Plots.text(L"P_{%$(n)}", 4), alpha=0.8)
        end
        if haskey(timings2, (n, 15))
            scatter!([get(timings2, (n, 15), NaN)], [get(Linf_norm, (n, 15), NaN)], markersize=[m_size(n)], color=5, label=(i==length(Ns2) ? L"r=15" : nothing), text=Plots.text(L"P_{%$(n)}", 4), alpha=0.8)
        end
    end
    xlabel!("Runtime (s)", xaxis=:log)
    ylabel!(L"L_\infty \textrm{\, error\, (rel.)} ", yaxis=:log)
    plot!(size=(400, 300), fontfamily="Computer Modern", dpi=1000)
    savefig(joinpath(figpath, "Linf_error.png"))
end

begin
    plot()
    for (i, n) in enumerate(Ns)
        if !(n==27)
            scatter!([memory_req(n, Inf)], [Linf_norm[(n, Inf)]], markersize=[m_size(n)], color=1, label=(i==length(Ns2) ? L"P_N" : nothing), text=Plots.text(L"P_{%$(n)}", 4), alpha=0.8)
        end
        if haskey(timings2, (n, 3))
            scatter!([memory_req(n, 3)], [get(Linf_norm, (n, 3), NaN)], markersize=[m_size(n)], color=2, label=(i==length(Ns2) ? L"r=3" : nothing), text=Plots.text(L"P_{%$(n)}", 4), alpha=0.8)
        end
        if haskey(timings2, (n, 5))
            scatter!([memory_req(n, 5)], [get(Linf_norm, (n, 5), NaN)], markersize=[m_size(n)], color=3, label=(i==length(Ns2) ? L"r=5" : nothing), text=Plots.text(L"P_{%$(n)}", 4), alpha=0.8)
        end
        if haskey(timings2, (n, 10))
            scatter!([memory_req(n, 10)], [get(Linf_norm, (n, 10), NaN)], markersize=[m_size(n)], color=4, label=(i==length(Ns2) ? L"r=10" : nothing), text=Plots.text(L"P_{%$(n)}", 4), alpha=0.8)
        end
        if haskey(timings2, (n, 15))
            scatter!([memory_req(n, 15)], [get(Linf_norm, (n, 15), NaN)], markersize=[m_size(n)], color=5, label=(i==length(Ns2) ? L"r=15" : nothing), text=Plots.text(L"P_{%$(n)}", 4), alpha=0.8)
        end
    end
    xlabel!("DOF", xaxis=:log)
    ylabel!(L"L_\infty \textrm{\, error\, (rel.)} ", yaxis=:log)
    plot!(size=(400, 300), fontfamily="Computer Modern", dpi=1000)
    savefig(joinpath(figpath, "Linf_error_memory.png"))
end

begin
    plot()
    for (i, n) in enumerate(Ns)
        if !(n==27)
            scatter!([timings2[(n, Inf)]], [M_norm[(n, Inf)]], markersize=[m_size(n)], color=1, label=(i==length(Ns2) ? L"P_N" : nothing), text=Plots.text(L"P_{%$(n)}", 4), alpha=0.8)
        end
        if haskey(timings2, (n, 3))
            scatter!([get(timings2, (n, 3), NaN)], [get(M_norm, (n, 3), NaN)], markersize=[m_size(n)], color=2, label=(i==length(Ns2) ? L"r=3" : nothing), text=Plots.text(L"P_{%$(n)}", 4), alpha=0.8)
        end
        if haskey(timings2, (n, 5))
            scatter!([get(timings2, (n, 5), NaN)], [get(M_norm, (n, 5), NaN)], markersize=[m_size(n)], color=3, label=(i==length(Ns2) ? L"r=5" : nothing), text=Plots.text(L"P_{%$(n)}", 4), alpha=0.8)
        end
        if haskey(timings2, (n, 10))
            scatter!([get(timings2, (n, 10), NaN)], [get(M_norm, (n, 10), NaN)], markersize=[m_size(n)], color=4, label=(i==length(Ns2) ? L"r=10" : nothing), text=Plots.text(L"P_{%$(n)}", 4), alpha=0.8)
        end
        if haskey(timings2, (n, 15))
            scatter!([get(timings2, (n, 15), NaN)], [get(M_norm, (n, 15), NaN)], markersize=[m_size(n)], color=5, label=(i==length(Ns2) ? L"r=15" : nothing), text=Plots.text(L"P_{%$(n)}", 4), alpha=0.8)
        end
    end
    xlabel!("Runtime (s)", xaxis=:log)
    ylabel!("Measurement Error (rel.)", yaxis=:log)
    plot!(size=(400, 300), fontfamily="Computer Modern", dpi=1000)
    savefig(joinpath(figpath, "M_error.png"))
end

begin
    plot()
    for (i, n) in enumerate(Ns)
        if !(n==27)
            scatter!([memory_req(n, Inf)], [M_norm[(n, Inf)]], markersize=[m_size(n)], color=1, label=(i==length(Ns2) ? L"P_N" : nothing), text=Plots.text(L"P_{%$(n)}", 4), alpha=0.8)
        end
        if haskey(timings2, (n, 3))
            scatter!([memory_req(n, 3)], [get(M_norm, (n, 3), NaN)], markersize=[m_size(n)], color=2, label=(i==length(Ns2) ? L"r=3" : nothing), text=Plots.text(L"P_{%$(n)}", 4), alpha=0.8)
        end
        if haskey(timings2, (n, 5))
            scatter!([memory_req(n, 5)], [get(M_norm, (n, 5), NaN)], markersize=[m_size(n)], color=3, label=(i==length(Ns2) ? L"r=5" : nothing), text=Plots.text(L"P_{%$(n)}", 4), alpha=0.8)
        end
        if haskey(timings2, (n, 10))
            scatter!([memory_req(n, 10)], [get(M_norm, (n, 10), NaN)], markersize=[m_size(n)], color=4, label=(i==length(Ns2) ? L"r=10" : nothing), text=Plots.text(L"P_{%$(n)}", 4), alpha=0.8)
        end
        if haskey(timings2, (n, 15))
            scatter!([memory_req(n, 15)], [get(M_norm, (n, 15), NaN)], markersize=[m_size(n)], color=5, label=(i==length(Ns2) ? L"r=15" : nothing), text=Plots.text(L"P_{%$(n)}", 4), alpha=0.8)
        end
    end
    xlabel!("DOF", xaxis=:log)
    ylabel!("Measurement Error (rel.)", yaxis=:log)
    plot!(size=(400, 300), fontfamily="Computer Modern", dpi=1000)
    savefig(joinpath(figpath, "M_error_memory.png"))
end

time_map = zeros(Union{Float64,Missing}, 5, length(Ns))
for (i, N) in enumerate(Ns)
    for (j, r) in enumerate([Inf, 15, 10, 5, 3])
        idx = findfirst((((N_, r_), _),) -> (N_ == N && r_ == r), timings)
        if !isnothing(idx)
            time_map[j, i] = timings[idx][2]
        else
            time_map[j, i] = missing
        end
    end
end

begin
    heatmap(log.(10, time_map), yflip=true, colorbar=false, cmap=cgrad([:green, :white, :red]))
    for (i, N) in enumerate(Ns)
        for (j, r) in enumerate([Inf, 15, 10, 5, 3])
            if !ismissing(time_map[j, i])
                annotate!(i, j, Plots.text("$(round(time_map[j, i], digits=2))s", "Computer Modern", 5), :black)
            end
        end
    end
    plot!(xticks=(1:length(Ns), [L"P_{%$(i)}" for i in Ns]))
    plot!(yticks=(1:5, [L"r=\infty", L"r=15", L"r=10", L"r=5", L"r=3"]))
    plot!(size=(400, 100), dpi=1000, fontfamily="Computer Modern", right_margin=2Plots.mm)

    savefig(joinpath(figpath, "timings_2d.png"))
end

begin
    direction_model = EPMAfem.SphericalHarmonicsModels.EOSphericalHarmonicsModel(27, 3)
    moments = sort(direction_model.moments)
    x = EPMAfem.SphericalHarmonicsModels.assemble_linear(EPMAfem.SphericalHarmonicsModels.∫S²_hv(Ω -> EPMAfem.beam_direction_distribution(excitation, 1, Ω)), direction_model, moments)
    plot(abs.(x), label=nothing)
    n_mom = [EPMAfem.SphericalHarmonicsModels.EOSphericalHarmonicsModel(N, 3).moments |> length for N in Ns]
    for i in 1:length(n_mom) plot!([n_mom[i], n_mom[i]], [-0.1, 0.73], color=:black, label=nothing) end
    [annotate!([n_mom[i] + (i==1 ? -6.0 : 0.0)], [0.8], Plots.text(L"P_{%$(Ns[i])}", "Computer Modern", 5)) for i in 1:length(Ns)]
    plot!(size=(400, 100), fontfamily="Computer Modern", dpi=1000)
    ylims!(-0.05, 0.9)
    savefig(joinpath(figpath, "boundary_moments.png"))
end
