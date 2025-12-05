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

figpath = mkpath(joinpath(dirname(@__FILE__), "figures/2D_epma_adjoint"))

equations = NExt.epma_equations(
    [n"Al", n"Cr"],
    [NExt.EPMADetector(n"Al K-L2", VectorValue(1.0, 0.0, 0.0)), NExt.EPMADetector(n"Cr K-L2", VectorValue(1.0, 0.0, 0.0))],
    range(50u"eV", 20u"keV", length=100), 27)

meas1, meas2, ranks1, ranks2, timings = Dict(), Dict(), Dict(), Dict(), Dict()
using Serialization
try
    (meas1, meas2, ranks1, ranks2, timings) = deserialize(joinpath(figpath, "data.jls"))
catch
    @warn "cannot load results"
end

Ns = [1, 3, 5, 7, 9, 11, 13, 15, 21, 27]

get_model(N) = NExt.epma_model(equations, (-2000u"nm", 0.0u"nm", -2000u"nm", 2000u"nm"), (150, 300), N)

for N in Ns
    @show N
    model = get_model(N)
    arch = EPMAfem.cuda(Float64)
    problem = EPMAfem.discretize_problem(equations, model, arch, updatable=true)

    excitation = EPMAfem.pn_excitation([(x=NExt.dimless(x_, equations.dim_basis), y=0.0) for x_ in range(-500u"nm", 500u"nm", 100)], [NExt.dimless(15u"keV", equations.dim_basis)], [VectorValue(-1.0, 0.0, 0.0)], beam_position_σ=0.02, beam_energy_σ=0.05, beam_direction_κ=50)
    discrete_rhs = EPMAfem.discretize_rhs(excitation, model, arch)
    discrete_ext = NExt.discretize_detectors(equations, model, arch, absorption=false)
    discrete_ext[1].vector.bϵ .*= 0.01/maximum(discrete_ext[1].vector.bϵ) # (normalize TODO: there should be a general way to normalize coeffs)
    discrete_ext[2].vector.bϵ .*= 0.01/maximum(discrete_ext[2].vector.bϵ)

    function mass_concentrations(e, x_)
        # return e == 1 ? NExt.dimless(n"Al".density, equations.dim_basis) : 0.0
        z = NExt.dimful(x_[1], u"nm", equations.dim_basis)
        x = NExt.dimful(x_[2], u"nm", equations.dim_basis)

        if (x - 80u"nm")^2 + (z - (-200u"nm"))^2 < (80u"nm")^2
            return e == 1 ? 0.0 : NExt.dimless(n"Cr".density, equations.dim_basis)
        else
            return e == 1 ? NExt.dimless(n"Al".density, equations.dim_basis) : 0.0
        end
    end
    ρs = EPMAfem.discretize_mass_concentrations([x -> mass_concentrations(e, x) for e in 1:EPMAfem.number_of_elements(equations)], model)

    EPMAfem.update_problem!(problem, ρs)
    EPMAfem.update_vector!(discrete_ext[1], ρs)
    EPMAfem.update_vector!(discrete_ext[2], ρs)

    system_full = EPMAfem.implicit_midpoint2(problem.problem, A -> PNLazyMatrices.schur_complement(A, Krylov.minres, PNLazyMatrices.cache ∘ LinearAlgebra.inv!));

    sol = EPMAfem.IterableDiscretePNSolution(adjoint(system_full), discrete_ext[1].vector, step_callback=(ϵ, ψ) -> @show "1", N, ϵ)
    for _ in Iterators.take(sol, 2) end # warmup
    timings[(N, 1, "full")] = @elapsed meas1[(N, "full")] = ((sol)*discrete_rhs)[:]
    
    sol = EPMAfem.IterableDiscretePNSolution(adjoint(system_full), discrete_ext[2].vector, step_callback=(ϵ, ψ) -> @show "2", N, ϵ)
    for _ in Iterators.take(sol, 2) end # warmup
    timings[(N, 2, "full")] = @elapsed meas2[(N, "full")] = ((sol)*discrete_rhs)[:]

    serialize(joinpath(figpath, "data.jls"), (meas1, meas2, ranks1, ranks2, timings))

    system_full_noschur = EPMAfem.implicit_midpoint2(problem.problem, Krylov.minres);

    sol = EPMAfem.IterableDiscretePNSolution(adjoint(system_full_noschur), discrete_ext[1].vector, step_callback=(ϵ, ψ) -> @show "1", N, ϵ)
    for _ in Iterators.take(sol, 2) end # warmup
    timings[(N, 1, "full_noschur")] = @elapsed meas1[(N, "full_noschur")] = ((sol)*discrete_rhs)[:]
    
    sol = EPMAfem.IterableDiscretePNSolution(adjoint(system_full_noschur), discrete_ext[2].vector, step_callback=(ϵ, ψ) -> @show "2", N, ϵ)
    for _ in Iterators.take(sol, 2) end # warmup
    timings[(N, 2, "full_noschur")] = @elapsed meas2[(N, "full_noschur")] = ((sol)*discrete_rhs)[:]

    serialize(joinpath(figpath, "data.jls"), (meas1, meas2, ranks1, ranks2, timings))

    if N > 1
        for (i_r, tol_r) in collect(enumerate([0.025, 0.0125, 0.00625]))
            begin # default lr
                system_lowrank = EPMAfem.implicit_midpoint_dlr5(problem.problem, max_ranks=(p=30, m=30), tolerance=tol_r);
                ranks1[(N, :noaug, tol_r)] = (p=zeros(length(EPMAfem.energy_model(model))), m=zeros(length(EPMAfem.energy_model(model))))
                sol = EPMAfem.IterableDiscretePNSolution(adjoint(system_lowrank), discrete_ext[1].vector; step_callback=(ϵ, ψ) -> begin
                    @show "1non", N, ϵ
                    ranks1[(N, :noaug, tol_r)].p[EPMAfem.plus½(ϵ)] = ψ.ranks.p[]
                    ranks1[(N, :noaug, tol_r)].m[EPMAfem.plus½(ϵ)] = ψ.ranks.m[]
                end)
                for _ in Iterators.take(sol, 2) end# warmup
                timings[(N, 1, :noaug, tol_r)] = @elapsed meas1[(N, :noaug, tol_r)] = (sol*discrete_rhs)[:]

                ranks2[(N, :noaug, tol_r)] = (p=zeros(length(EPMAfem.energy_model(model))), m=zeros(length(EPMAfem.energy_model(model))))
                sol = EPMAfem.IterableDiscretePNSolution(adjoint(system_lowrank), discrete_ext[2].vector; step_callback=(ϵ, ψ) -> begin
                    @show "2non", N, ϵ
                    ranks2[(N, :noaug, tol_r)].p[EPMAfem.plus½(ϵ)] = ψ.ranks.p[]
                    ranks2[(N, :noaug, tol_r)].m[EPMAfem.plus½(ϵ)] = ψ.ranks.m[]
                end)
                for _ in Iterators.take(sol, 2) end # warmup
                timings[(N, 2, :noaug, tol_r)] = @elapsed meas2[(N, :noaug, tol_r)] = (sol*discrete_rhs)[:]
                serialize(joinpath(figpath, "data.jls"), (meas1, meas2, ranks1, ranks2, timings))
            end

            begin # mass conservative
                system_lowrank_aug = EPMAfem.implicit_midpoint_dlr5(problem.problem, max_ranks=(p=30, m=30), tolerance=tol_r, basis_augmentation=:mass);
                ranks1[(N, :aug, tol_r)] = (p=zeros(length(EPMAfem.energy_model(model))), m=zeros(length(EPMAfem.energy_model(model))))
                sol = EPMAfem.IterableDiscretePNSolution(adjoint(system_lowrank_aug), discrete_ext[1].vector; step_callback=(ϵ, ψ) -> begin
                    @show "1mass", N, ϵ
                    ranks1[(N, :aug, tol_r)].p[EPMAfem.plus½(ϵ)] = ψ.ranks.p[]
                    ranks1[(N, :aug, tol_r)].m[EPMAfem.plus½(ϵ)] = ψ.ranks.m[]
                end)
                for _ in Iterators.take(sol, 2) end# warmup
                timings[(N, 1, :aug, tol_r)] = @elapsed meas1[(N, :aug, tol_r)] = (sol*discrete_rhs)[:]

                ranks2[(N, :aug, tol_r)] = (p=zeros(length(EPMAfem.energy_model(model))), m=zeros(length(EPMAfem.energy_model(model))))
                sol = EPMAfem.IterableDiscretePNSolution(adjoint(system_lowrank_aug), discrete_ext[2].vector; step_callback=(ϵ, ψ) -> begin
                    @show "2mass", N, ϵ
                    ranks2[(N, :aug, tol_r)].p[EPMAfem.plus½(ϵ)] = ψ.ranks.p[]
                    ranks2[(N, :aug, tol_r)].m[EPMAfem.plus½(ϵ)] = ψ.ranks.m[]
                end)
                for _ in Iterators.take(sol, 2) end # warmup
                timings[(N, 2, :aug, tol_r)] = @elapsed meas2[(N, :aug, tol_r)] = (sol*discrete_rhs)[:]
                serialize(joinpath(figpath, "data.jls"), (meas1, meas2, ranks1, ranks2, timings))
            end

            begin # mass and "extraction" conservative
                nb = EPMAfem.n_basis(model)
                basis_augmentation = (  p=(V = EPMAfem.allocate_mat(EPMAfem.architecture(problem.problem), nb.nΩ.p, 1), ),
                                        m=(V = EPMAfem.allocate_mat(EPMAfem.architecture(problem.problem), nb.nΩ.m, 1), ))
                Ωp, Ωm = EPMAfem.SphericalHarmonicsModels.eval_basis(EPMAfem.direction_model(model), one)
                basis_augmentation.m.V[:, 1] .= Ωm |> normalize |> arch
                basis_augmentation.p.V[:, 1] .= collect(discrete_rhs[1].bΩ.p) |> normalize |> arch
                system_lowrank_aug2 = EPMAfem.implicit_midpoint_dlr5(problem.problem, max_ranks=(p=30, m=30), tolerance=tol_r, basis_augmentation=basis_augmentation);
                ranks1[(N, :aug2, tol_r)] = (p=zeros(length(EPMAfem.energy_model(model))), m=zeros(length(EPMAfem.energy_model(model))))
                sol = EPMAfem.IterableDiscretePNSolution(adjoint(system_lowrank_aug2), discrete_ext[1].vector; step_callback=(ϵ, ψ) -> begin
                    @show "1mass", N, ϵ
                    ranks1[(N, :aug2, tol_r)].p[EPMAfem.plus½(ϵ)] = ψ.ranks.p[]
                    ranks1[(N, :aug2, tol_r)].m[EPMAfem.plus½(ϵ)] = ψ.ranks.m[]
                end)
                for _ in Iterators.take(sol, 2) end# warmup
                timings[(N, 1, :aug2, tol_r)] = @elapsed meas1[(N, :aug2, tol_r)] = (sol*discrete_rhs)[:]

                ranks2[(N, :aug2, tol_r)] = (p=zeros(length(EPMAfem.energy_model(model))), m=zeros(length(EPMAfem.energy_model(model))))
                sol = EPMAfem.IterableDiscretePNSolution(adjoint(system_lowrank_aug2), discrete_ext[2].vector; step_callback=(ϵ, ψ) -> begin
                    @show "2mass", N, ϵ
                    ranks2[(N, :aug2, tol_r)].p[EPMAfem.plus½(ϵ)] = ψ.ranks.p[]
                    ranks2[(N, :aug2, tol_r)].m[EPMAfem.plus½(ϵ)] = ψ.ranks.m[]
                end)
                for _ in Iterators.take(sol, 2) end # warmup
                timings[(N, 2, :aug2, tol_r)] = @elapsed meas2[(N, :aug2, tol_r)] = (sol*discrete_rhs)[:]
                serialize(joinpath(figpath, "data.jls"), (meas1, meas2, ranks1, ranks2, timings))
            end
        end
    end
end

meas1

(meas1, meas2, ranks1, ranks2, timings) = deserialize(joinpath(figpath, "data.jls"))

begin
    x = range(-500u"nm", 500u"nm", 100)
    plot()
    plot!(x, meas1[(7, "full")], label=L"P_{7}", color=:black, ls=:dashdot, linewidth=1)
    plot!(x, meas1[(13, "full")], label=L"P_{13}", color=:black, ls=:dash, linewidth=1)
    
    N3 = 27
    plot!(x, meas1[(N3, "full")], label=L"P_{%$(N3)}", color=:black, linewidth=1)

    plot!(x, meas1[(N3, :noaug, 0.025)], label=nothing, ls=:solid, color=2)
    plot!([], [], label="BUG", ls=:solid, color=:gray)
    plot!(x, meas1[(N3, :aug, 0.025)], label=nothing, ls=:dash, color=2)
    plot!([], [], label="mass cons.", ls=:dash, color=:gray)
    plot!(x, meas1[(N3, :aug2, 0.025)], label=nothing, ls=:dot, color=2)
    plot!([], [], label="mass cons., beam", ls=:dot, color=:gray)

    plot!(x, meas1[(N3, :noaug, 0.0125)], label=nothing, color=3, ls=:solid)
    plot!(x, meas1[(N3, :aug, 0.0125)], label=nothing, color=3, ls=:dash)
    plot!(x, meas1[(N3, :aug2, 0.0125)], label=nothing, color=3, ls=:dot)

    plot!(x, meas1[(N3, :noaug, 0.00625)], label=nothing, color=4, ls=:solid)
    plot!(x, meas1[(N3, :aug, 0.00625)], label=nothing, color=4, ls=:dash)
    plot!(x, meas1[(N3, :aug2, 0.00625)], label=nothing, color=4, ls=:dot)

    scatter!([], [], color=:2, ls=:solid, label=L"\vartheta=0.025", markerstrokewidth=0)
    scatter!([], [], color=:3, ls=:solid, label=L"\vartheta=0.0125", markerstrokewidth=0)
    scatter!([], [], color=:4, ls=:solid, label=L"\vartheta=0.00625", markerstrokewidth=0)

    plot!(size=(400, 300), fontfamily="Computer Modern", dpi=1000, legend=:bottomleft)
    ylims!(1.0, 1.65)
    savefig(joinpath(figpath, "measurements_Al.png"))
end

begin
    x = range(-500u"nm", 500u"nm", 100)
    plot()
    plot!(x, meas2[(3, "full")], label=L"P_{3}", color=:black, ls=:dashdot, linewidth=1)
    plot!(x, meas2[(5, "full")], label=L"P_{5}", color=:black, ls=:dash, linewidth=1)
    
    N3 = 27
    plot!(x, meas2[(N3, "full")], label=L"P_{%$(N3)}", color=:black, linewidth=1)

    plot!(x, meas2[(N3, :noaug, 0.025)], label=nothing, ls=:solid, color=2)
    plot!([], [], label="BUG", ls=:solid, color=:gray)
    plot!(x, meas2[(N3, :aug, 0.025)], label=nothing, ls=:dash, color=2)
    plot!([], [], label="mass cons.", ls=:dash, color=:gray)
    plot!(x, meas2[(N3, :aug2, 0.025)], label=nothing, ls=:dot, color=2)
    plot!([], [], label="mass cons., beam", ls=:dot, color=:gray)

    plot!(x, meas2[(N3, :noaug, 0.0125)], label=nothing, color=3, ls=:solid)
    plot!(x, meas2[(N3, :aug, 0.0125)], label=nothing, color=3, ls=:dash)
    plot!(x, meas2[(N3, :aug2, 0.0125)], label=nothing, color=3, ls=:dot)

    plot!(x, meas2[(N3, :noaug, 0.00625)], label=nothing, color=4, ls=:solid)
    plot!(x, meas2[(N3, :aug, 0.00625)], label=nothing, color=4, ls=:dash)
    plot!(x, meas2[(N3, :aug2, 0.00625)], label=nothing, color=4, ls=:dot)

    scatter!([], [], color=:2, ls=:solid, label=L"\vartheta=0.025", markerstrokewidth=0)
    scatter!([], [], color=:3, ls=:solid, label=L"\vartheta=0.0125", markerstrokewidth=0)
    scatter!([], [], color=:4, ls=:solid, label=L"\vartheta=0.00625", markerstrokewidth=0)

    plot!(size=(400, 300), fontfamily="Computer Modern", dpi=1000, legend=:topleft)
    # ylims!(1.0, 1.65)
    savefig(joinpath(figpath, "measurements_Cr.png"))
end


function memory_req(N, ranks)
    model = get_model(N)
    (ne, (nxp, nxm), (nΩp, nΩm)) = EPMAfem.n_basis(model)
    if ranks isa Number && isinf(ranks)
        return ne*(nxp*nΩp + nxm*nΩm)
    else
        nn = 0
        for (rp, rm) in zip(ranks.p, ranks.m)
            rp = Int(rp)
            rm = Int(rm)
            nn += nxp*rp + rp*rp + rp*nΩp + nxm*rm + rm*rm + rm*nΩm
        end
        return nn
    end
end

# max abs error
rel_error(ref, meas, p) = norm(ref .- meas, p) / norm(ref, p)

m_size(N) = 5*N^(1/4)

begin
    N_ref = 27

    for (norm_p_name, norm_p) in [("\\infty", Inf), ("1", 1), ("2", 2)]
        for (m_name, (meas_, ranks_)) in [("Al", (meas1, ranks1)), ("Cr", (meas2, ranks2))]
            for (aug_name, aug) in [("aug", :aug), ("noaug", :noaug), ("aug2", :aug2)]
                plot()
                for (i, N) in enumerate(Ns)
                    if N != N_ref
                        scatter!([memory_req(N, Inf)], [rel_error(meas_[(N_ref, "full")], meas_[(N, "full")], norm_p)], label=(i==length(Ns)-2 ? L"P_N" : nothing), markersize=[m_size(N)], text=Plots.text(L"P_{%$N}", 4), color=1, alpha=0.8)
                    end
                    if N != 1
                        scatter!([memory_req(N, ranks_[(N, aug, 0.025)])], [rel_error(meas_[(N_ref, "full")], meas_[(N, aug, 0.025)], norm_p)], label=(i==length(Ns)-2 ? L"\vartheta=0.025" : nothing), markersize=[m_size(N)], text=Plots.text(L"P_{%$N}", 4), color=2, alpha=0.8)
                        scatter!([memory_req(N, ranks_[(N, aug, 0.0125)])], [rel_error(meas_[(N_ref, "full")], meas_[(N, aug, 0.0125)], norm_p)], label=(i==length(Ns)-2 ? L"\vartheta=0.0125" : nothing), markersize=[m_size(N)], text=Plots.text(L"P_{%$N}", 4), color=3, alpha=0.8)
                        scatter!([memory_req(N, ranks_[(N, aug, 0.00625)])], [rel_error(meas_[(N_ref, "full")], meas_[(N, aug, 0.00625)], norm_p)], label=(i==length(Ns)-2 ? L"\vartheta=0.00625" : nothing), markersize=[m_size(N)], text=Plots.text(L"P_{%$N}", 4), color=4, alpha=0.8)
                    end
                end
                xlabel!("DOF", xaxis=:log)
                ylabel!(L"\ell_%$(norm_p_name) \textrm{\, error\, (rel.)} ", yaxis=:log)
                plot!(size=(400, 300), fontfamily="Computer Modern", dpi=1000)
                savefig(joinpath(figpath, "l_$(norm_p)_memory_$(m_name)_$(aug_name).png"))
            end
        end
    end
end

begin
    N_ref = 27

    for (norm_p_name, norm_p) in [("\\infty", Inf), ("1", 1), ("2", 2)]
        for (m_name, (meas_, m_num)) in [("Al", (meas1, 1)), ("Cr", (meas2, 2))]
            for (aug_name, aug) in [("aug", :aug), ("noaug", :noaug), ("aug2", :aug2)]
                plot()
                for (i, N) in enumerate(Ns)
                    if N != N_ref
                        scatter!([timings[(N, m_num, "full")]], [rel_error(meas_[(N_ref, "full")], meas_[(N, "full")], norm_p)], label=(i==length(Ns)-2 ? L"P_N" : nothing), markersize=[m_size(N)], text=Plots.text(L"P_{%$N}", 4), color=1, alpha=0.8)
                        if N ∈ Ns[1:end-3]
                            scatter!([timings[(N, m_num, "full_noschur")]], [rel_error(meas_[(N_ref, "full")], meas_[(N, "full_noschur")], norm_p)], label=(i==length(Ns)-2 ? L"P_N" : nothing), markersize=[m_size(N)], text=Plots.text(L"P_{%$N}", 4), color=1, alpha=0.3)
                        end
                    end
                
                    if N != 1
                        scatter!([timings[(N, m_num, aug, 0.025)]], [rel_error(meas_[(N_ref, "full")], meas_[(N, aug, 0.025)], norm_p)], label=(i==length(Ns)-2 ? L"\vartheta=0.025" : nothing), markersize=[m_size(N)], text=Plots.text(L"P_{%$N}", 4), color=2, alpha=0.8)
                        scatter!([timings[(N, m_num, aug, 0.0125)]], [rel_error(meas_[(N_ref, "full")], meas_[(N, aug, 0.0125)], norm_p)], label=(i==length(Ns)-2 ? L"\vartheta=0.0125" : nothing), markersize=[m_size(N)], text=Plots.text(L"P_{%$N}", 4), color=3, alpha=0.8)
                        scatter!([timings[(N, m_num, aug, 0.00625)]], [rel_error(meas_[(N_ref, "full")], meas_[(N, aug, 0.00625)], norm_p)], label=(i==length(Ns)-2 ? L"\vartheta=0.00625" : nothing), markersize=[m_size(N)], text=Plots.text(L"P_{%$N}", 4), color=4, alpha=0.8)

                    end
                end
                xlabel!("runtime [s]", xaxis=:log)
                ylabel!(L"\ell_%$(norm_p_name) \textrm{\, error\, (rel.)} ", yaxis=:log)
                plot!(size=(400, 300), fontfamily="Computer Modern", dpi=1000)
                savefig(joinpath(figpath, "l_$(norm_p)_runtime_$(m_name)_$(aug_name).png"))
            end
        end
    end
end

begin
    for (m_name, (meas_, ranks_, m_num)) in [("Al", (meas1, ranks1, 1)), ("Cr", (meas2, ranks2, 2))]
        plot()
        for (aug_name, aug) in [("BUG", :noaug), ("mass cons.", :aug), ("mass cons., beam", :aug2)]
            ls = Dict(:noaug => :solid, :aug => :dot, :aug2 => :dash)
            plot!(ranks_[(27, aug, 0.025)].p + ranks_[(27, aug, 0.025)].m, color=2, label=nothing, ls=ls[aug])
            plot!(ranks_[(27, aug, 0.0125)].p + ranks_[(27, aug, 0.0125)].m, color=3, label=nothing, ls=ls[aug])
            plot!(ranks_[(27, aug, 0.00625)].p + ranks_[(27, aug, 0.00625)].m, color=4, label=nothing, ls=ls[aug])
            plot!([], [], ls=ls[aug], label="$(aug_name)", color=:gray)
        end
        plot!([], [], color=2, label=L"\vartheta=0.025")
        plot!([], [], color=3, label=L"\vartheta=0.0125")
        plot!([], [], color=4, label=L"\vartheta=0.00625")
        xlabel!("energy step")
        ylabel!(L"r^+ + r^-")
        plot!(size=(400, 300), fontfamily="Computer Modern", dpi=1000)
        savefig(joinpath(figpath, "ranks_$(m_name).png"))
    end
end
