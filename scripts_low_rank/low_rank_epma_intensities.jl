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

meas1 = Dict()
meas2 = Dict()

ranks1 = Dict()
ranks2 = Dict()
Ns = [1, 3, 5, 7, 9, 11, 13, 15, 21, 27]

get_model(N) = NExt.epma_model(equations, (-2000u"nm", 0.0u"nm", -2000u"nm", 2000u"nm"), (150, 300), N)

for N in Ns
    model = NExt.epma_model(equations, (-2000u"nm", 0.0u"nm", -2000u"nm", 2000u"nm"), (150, 300), N)
    arch = EPMAfem.cuda(Float64)
    problem = EPMAfem.discretize_problem(equations, model, arch, updatable=true)

    excitation = EPMAfem.pn_excitation([(x=NExt.dimless(x_, equations.dim_basis), y=0.0) for x_ in range(-500u"nm", 500u"nm", 100)], [NExt.dimless(15u"keV", equations.dim_basis)], [VectorValue(-1.0, 0.0, 0.0)], beam_position_σ=0.02, beam_energy_σ=0.05, beam_direction_κ=50)
    discrete_rhs = EPMAfem.discretize_rhs(excitation, model, arch)
    discrete_ext = NExt.discretize_detectors(equations, model, arch, absorption=false)
    discrete_ext[1].vector.bϵ .*= 0.01/maximum(discrete_ext[1].vector.bϵ) # (normalize TODO: there should be a general way to normalize coeffs)
    discrete_ext[2].vector.bϵ .*= 0.01/maximum(discrete_ext[2].vector.bϵ)

    # discrete_ext[1].vector.bϵ .= 0.01 # (normalize TODO: there should be a general way to normalize coeffs)
    # discrete_ext[2].vector.bϵ .= 0.01


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
    # heatmap(reshape(ρs[1, :], 150, 300), aspect_ratio=:equal)
    # heatmap(reshape(ρs[2, :], 150, 300), aspect_ratio=:equal)

    EPMAfem.update_problem!(problem, ρs)
    EPMAfem.update_vector!(discrete_ext[1], ρs)
    EPMAfem.update_vector!(discrete_ext[2], ρs)

    system_full = EPMAfem.implicit_midpoint2(problem.problem, A -> PNLazyMatrices.schur_complement(A, Krylov.minres, PNLazyMatrices.cache ∘ LinearAlgebra.inv!));

    # Ωp, Ωm = EPMAfem.SphericalHarmonicsModels.eval_basis(EPMAfem.direction_model(model), one) |> arch
    # nb = EPMAfem.n_basis(model)
    # for (ϵ, ψ) in discrete_ext[1].vector*system_
    #     ψp, ψm = EPMAfem.pmview(ψ, model)

    #     res = ψm*Ωm |> collect
    #     p = heatmap(reshape(res, (40, 80)))
    #     # func = EPMAfem.SpaceModels.interpolable((p=zeros(nb.nx.p), m=ψm*Ωm |> collect), EPMAfem.space_model(model))
    #     # p = heatmap(-1500u"nm":1u"nm":1500u"nm", -1500u"nm":1u"nm":0u"nm", (x, z) -> func(VectorValue(NExt.dimless.((z, x), Ref(equations.dim_basis)))))
    #     display(p)
    #     sleep(0.1)
    # end

    timings = Dict()

    sol = EPMAfem.IterableDiscretePNSolution(adjoint(system_full), discrete_ext[1].vector, step_callback=(ϵ, ψ) -> @show ϵ)
    for _ in Iterators.take(sol, 2) end # warmup
    timings[(N, 1, "full")] = @elapsed meas1[(N, "full")] = ((sol)*discrete_rhs)[:]
    
    sol = EPMAfem.IterableDiscretePNSolution(adjoint(system_full), discrete_ext[2].vector, step_callback=(ϵ, ψ) -> @show ϵ)
    for _ in Iterators.take(sol, 2) end # warmup
    timings[(N, 2, "full")] = @elapsed meas2[(N, "full")] = ((sol)*discrete_rhs)[:]


    if N > 1
        for (i_r, tol_r) in collect(enumerate([0.025, 0.0125, 0.00625]))
            system_lowrank = EPMAfem.implicit_midpoint_dlr5(problem.problem, max_ranks=(p=30, m=30), tolerance=tol_r);
            ranks1[(N, :noaug, tol_r)] = (p=zeros(length(EPMAfem.energy_model(model))), m=zeros(length(EPMAfem.energy_model(model))))
            sol = EPMAfem.IterableDiscretePNSolution(adjoint(system_lowrank), discrete_ext[1].vector; step_callback=(ϵ, ψ) -> begin
                @show ϵ
                ranks1[(N, :noaug, tol_r)].p[EPMAfem.plus½(ϵ)] = ψ.ranks.p[]
                ranks1[(N, :noaug, tol_r)].m[EPMAfem.plus½(ϵ)] = ψ.ranks.m[]
            end)
            for _ in Iterators.take(sol, 2) end# warmup
            timings[(N, 1, :noaug, tol_r)] = @elapsed meas1[(N, :noaug, tol_r)] = (sol*discrete_rhs)[:]

            ranks2[(N, :noaug, tol_r)] = (p=zeros(length(EPMAfem.energy_model(model))), m=zeros(length(EPMAfem.energy_model(model))))
            sol = EPMAfem.IterableDiscretePNSolution(adjoint(system_lowrank), discrete_ext[2].vector; step_callback=(ϵ, ψ) -> begin
                @show ϵ
                ranks2[(N, :noaug, tol_r)].p[EPMAfem.plus½(ϵ)] = ψ.ranks.p[]
                ranks2[(N, :noaug, tol_r)].m[EPMAfem.plus½(ϵ)] = ψ.ranks.m[]
            end)
            for _ in Iterators.take(sol, 2) end # warmup
            timings[(N, 2, :noaug, tol_r)] = @elapsed meas2[(N, :noaug, tol_r)] = (sol*discrete_rhs)[:]

            system_lowrank_aug = EPMAfem.implicit_midpoint_dlr5(problem.problem, max_ranks=(p=30, m=30), tolerance=tol_r, basis_augmentation=:mass);
            ranks1[(N, :aug, tol_r)] = (p=zeros(length(EPMAfem.energy_model(model))), m=zeros(length(EPMAfem.energy_model(model))))
            sol = EPMAfem.IterableDiscretePNSolution(adjoint(system_lowrank_aug), discrete_ext[1].vector; step_callback=(ϵ, ψ) -> begin
                @show ϵ
                ranks1[(N, :aug, tol_r)].p[EPMAfem.plus½(ϵ)] = ψ.ranks.p[]
                ranks1[(N, :aug, tol_r)].m[EPMAfem.plus½(ϵ)] = ψ.ranks.m[]
            end)
            for _ in Iterators.take(sol, 2) end# warmup
            timings[(N, 1, :aug, tol_r)] = @elapsed meas1[(N, :aug, tol_r)] = (sol*discrete_rhs)[:]

            ranks2[(N, :aug, tol_r)] = (p=zeros(length(EPMAfem.energy_model(model))), m=zeros(length(EPMAfem.energy_model(model))))
            sol = EPMAfem.IterableDiscretePNSolution(adjoint(system_lowrank_aug), discrete_ext[2].vector; step_callback=(ϵ, ψ) -> begin
                @show ϵ
                ranks2[(N, :aug, tol_r)].p[EPMAfem.plus½(ϵ)] = ψ.ranks.p[]
                ranks2[(N, :aug, tol_r)].m[EPMAfem.plus½(ϵ)] = ψ.ranks.m[]
            end)
            for _ in Iterators.take(sol, 2) end # warmup
            timings[(N, 2, :aug, tol_r)] = @elapsed meas2[(N, :aug, tol_r)] = (sol*discrete_rhs)[:]
        end
    end
end

using Serialization
serialize(joinpath(figpath, "data.jls"), (meas1, meas2, ranks1, ranks2))

plot()
for N in Ns[1:end-1] plot!(meas2[(N, "full")]) end
plot!()

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

p1 = scatter(Ns[1:end-2], [rel_error(meas1[(21, "full")], meas1[(N, "full")], Inf) for N in Ns[1:end-2]], yaxis=:log, marker=:x)
scatter!(Ns[2:end-2], [rel_error(meas1[(21, "full")], meas1[(N, :noaug, 0.025)], Inf) for N in Ns[2:end-2]], yaxis=:log, marker=:x)
scatter!(Ns[2:end-2], [rel_error(meas1[(21, "full")], meas1[(N, :noaug, 0.0125)], Inf) for N in Ns[2:end-2]], yaxis=:log, marker=:x)
scatter!(Ns[2:end-2], [rel_error(meas1[(21, "full")], meas1[(N, :noaug, 0.00625)], Inf) for N in Ns[2:end-2]], yaxis=:log, marker=:x)

p2 = scatter(Ns[1:end-2], [rel_error(meas2[(21, "full")], meas2[(N, "full")], Inf) for N in Ns[1:end-2]], yaxis=:log, marker=:x)
scatter!(Ns[2:end-2], [rel_error(meas2[(21, "full")], meas2[(N, :noaug, 0.025)], Inf) for N in Ns[2:end-2]], yaxis=:log, marker=:x)
scatter!(Ns[2:end-2], [rel_error(meas2[(21, "full")], meas2[(N, :noaug, 0.0125)], Inf) for N in Ns[2:end-2]], yaxis=:log, marker=:x)
scatter!(Ns[2:end-2], [rel_error(meas2[(21, "full")], meas2[(N, :noaug, 0.00625)], Inf) for N in Ns[2:end-2]], yaxis=:log, marker=:x)


m_size(N) = 5*N^(1/4)

begin
    N_ref = 21

    for norm_p in [Inf, 1, 2]
        for (m_name, (meas_, ranks_)) in [("Al", (meas1, ranks1)), ("Cr", (meas2, ranks2))]
            for (aug_name, aug) in [("aug", :aug), ("noaug", :noaug)]
                plot()
                for (i, N) in enumerate(Ns[1:end-1])
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
                ylabel!(L"\ell_\infty \textrm{\, error\, (rel.)} ", yaxis=:log)
                # xlims!(1e7, 1e9)
                # ylims!(2e-3, 0.5)
                plot!(size=(400, 300), fontfamily="Computer Modern", dpi=1000)
                savefig(joinpath(figpath, "l_$(norm_p)_memory_$(m_name)_$(aug_name).png"))
            end
        end
    end
end


plot()
for (i, tol) in enumerate([0.025, 0.0125, 0.00625])
    plot!(meas1[(21, :noaug, tol)], color=i)
    plot!(meas1[(21, :aug, tol)], ls=:dash, color=i)
end
plot!()

plot()
for (i, tol) in enumerate([0.025, 0.0125, 0.00625])
    plot!(ranks2[(21, :noaug, tol)].p, color=i+1)
    plot!(ranks2[(21, :noaug, tol)].m, color=i+1, alpha=0.5)
    plot!(ranks2[(21, :aug, tol)].p, ls=:dash, color=i+1)
    plot!(ranks2[(21, :aug, tol)].m, ls=:dash, color=i+1, alpha=0.5)
end
plot!()



    
    # scatter!([memory_req(N, ranks1[(N, :noaug, 0.025  )]) for N in Ns[2:end-2]], [rel_error(meas1[(21, "full")], meas1[(N, :noaug, 0.025)], Inf) for N in Ns[2:end-2]], yaxis=:log, marker=:x)
    # scatter!([memory_req(N, ranks1[(N, :noaug, 0.0125 )]) for N in Ns[2:end-2]], [rel_error(meas1[(21, "full")], meas1[(N, :noaug, 0.0125)], Inf) for N in Ns[2:end-2]], yaxis=:log, marker=:x)
    # scatter!([memory_req(N, ranks1[(N, :noaug, 0.00625)]) for N in Ns[2:end-2]], [rel_error(meas1[(21, "full")], meas1[(N, :noaug, 0.00625)], Inf) for N in Ns[2:end-2]], yaxis=:log, marker=:x)
    # ylims!(1e-4, 1)
    # title!("Al")
    # p2 = scatter([memory_req(N, Inf) for N in Ns[1:end-2]], [rel_error(meas2[(21, "full")], meas2[(N, "full")], Inf) for N in Ns[1:end-2]], yaxis=:log, xaxis=:log, marker=:x)
    # scatter!([memory_req(N, ranks2[(N, :noaug, 0.025  )]) for N in Ns[2:end-2]], [rel_error(meas2[(21, "full")], meas2[(N, :noaug, 0.025)], Inf) for N in Ns[2:end-2]], yaxis=:log, marker=:x)
    # scatter!([memory_req(N, ranks2[(N, :noaug, 0.0125 )]) for N in Ns[2:end-2]], [rel_error(meas2[(21, "full")], meas2[(N, :noaug, 0.0125)], Inf) for N in Ns[2:end-2]], yaxis=:log, marker=:x)
    # scatter!([memory_req(N, ranks2[(N, :noaug, 0.00625)]) for N in Ns[2:end-2]], [rel_error(meas2[(21, "full")], meas2[(N, :noaug, 0.00625)], Inf) for N in Ns[2:end-2]], yaxis=:log, marker=:x)
    # ylims!(1e-4, 1)
    # title!("Cr")
    # plot(p1, p2)
    # savefig(joinpath(figpath, "memory_error_meas_noaug.png"))
end

begin
    p1 = scatter([memory_req(N, Inf) for N in Ns[1:end-2]], [rel_error(meas1[(21, "full")], meas1[(N, "full")], Inf) for N in Ns[1:end-2]], yaxis=:log, xaxis=:log, marker=:x)
    scatter!([memory_req(N, ranks1[(N, :aug, 0.025  )]) for N in Ns[2:end-2]], [rel_error(meas1[(21, "full")], meas1[(N, :aug, 0.025)], Inf) for N in Ns[2:end-2]], yaxis=:log, marker=:x)
    scatter!([memory_req(N, ranks1[(N, :aug, 0.0125 )]) for N in Ns[2:end-2]], [rel_error(meas1[(21, "full")], meas1[(N, :aug, 0.0125)], Inf) for N in Ns[2:end-2]], yaxis=:log, marker=:x)
    scatter!([memory_req(N, ranks1[(N, :aug, 0.00625)]) for N in Ns[2:end-2]], [rel_error(meas1[(21, "full")], meas1[(N, :aug, 0.00625)], Inf) for N in Ns[2:end-2]], yaxis=:log, marker=:x)
    ylims!(1e-4, 1)
    title!("Al")

    p2 = scatter([memory_req(N, Inf) for N in Ns[1:end-2]], [rel_error(meas2[(21, "full")], meas2[(N, "full")], Inf) for N in Ns[1:end-2]], yaxis=:log, xaxis=:log, marker=:x)
    scatter!([memory_req(N, ranks2[(N, :aug, 0.025  )]) for N in Ns[2:end-2]], [rel_error(meas2[(21, "full")], meas2[(N, :aug, 0.025)], Inf) for N in Ns[2:end-2]], yaxis=:log, marker=:x)
    scatter!([memory_req(N, ranks2[(N, :aug, 0.0125 )]) for N in Ns[2:end-2]], [rel_error(meas2[(21, "full")], meas2[(N, :aug, 0.0125)], Inf) for N in Ns[2:end-2]], yaxis=:log, marker=:x)
    scatter!([memory_req(N, ranks2[(N, :aug, 0.00625)]) for N in Ns[2:end-2]], [rel_error(meas2[(21, "full")], meas2[(N, :aug, 0.00625)], Inf) for N in Ns[2:end-2]], yaxis=:log, marker=:x)
    ylims!(1e-4, 1)
    title!("Cr")

    plot(p1, p2)
    savefig(joinpath(figpath, "memory_error_meas_aug.png"))
end

plot(ranks1[(N)])


p2 = scatter(Ns[1:end-2], [rel_error(meas2[(21, "full")], meas2[(N, "full")], Inf) for N in Ns[1:end-2]], yaxis=:log, marker=:x)
scatter!(Ns[2:end-2], [rel_error(meas2[(21, "full")], meas2[(N, :noaug, 0.025)], Inf) for N in Ns[2:end-2]], yaxis=:log, marker=:x)
scatter!(Ns[2:end-2], [rel_error(meas2[(21, "full")], meas2[(N, :noaug, 0.0125)], Inf) for N in Ns[2:end-2]], yaxis=:log, marker=:x)
scatter!(Ns[2:end-2], [rel_error(meas2[(21, "full")], meas2[(N, :noaug, 0.00625)], Inf) for N in Ns[2:end-2]], yaxis=:log, marker=:x)


plot(meas1[(1)])


plot(meas1["full"])
plot!(meas1[(:noaug, 0.025)])
plot!(meas1[(:noaug, 0.0125)])
plot!(meas1[(:noaug, 0.00625)])

plot!(meas1[(:aug, 0.025)])
plot!(meas1[(:aug, 0.0125)])
plot!(meas1[(:aug, 0.00625)])

plot(meas2["full"])
plot!(meas2[(:noaug, 0.025)])
plot!(meas2[(:noaug, 0.0125)])
plot!(meas2[(:noaug, 0.00625)])

plot!(meas2[(:aug, 0.025)])
plot!(meas2[(:aug, 0.0125)])
plot!(meas2[(:aug, 0.00625)])

plot(ranks1[(:aug, 0.00625)].p)
plot!(ranks1[(:aug, 0.00625)].m)
plot(ranks2[(:aug, 0.00625)].p)
plot!(ranks2[(:aug, 0.00625)].m)
