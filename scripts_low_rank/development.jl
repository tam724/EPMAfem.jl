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

using Plots

# figpath = mkpath(joinpath(dirname(@__FILE__), "figures/2D_epma_adjoint"))

equations = EPMAfem.PNEquations()

measurements = Dict()
# for N in [11, 41]
    N = 41
    direction_model = EPMAfem.SphericalHarmonicsModels.EOSphericalHarmonicsModel(N, 1)
    space_model = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((-1, 0), 300))
    model = EPMAfem.DiscretePNModel(space_model, 0:0.005:1, direction_model)

    excitation = EPMAfem.pn_excitation([(x=0.0, y=0.0)], [0.8], [VectorValue(-1.0, 0.0, 0.0)], beam_position_σ=0.02, beam_energy_σ=0.05, beam_direction_κ=50)
    extraction = EPMAfem.PNExtraction([0.2, 0.3], equations)

    EPMAfem.mass_concentrations(::EPMAfem.PNEquations, e, x) = if (x[1] > -0.2 && x[1] < -0.1) return e == 1 ? 1.0 : 0.0 else return e == 1 ? 0.0 : 1.0 end

    problem = EPMAfem.discretize_problem(equations, model, EPMAfem.cuda())
    discrete_rhs = EPMAfem.discretize_rhs(excitation, model, EPMAfem.cuda())
    discrete_ext = EPMAfem.discretize_extraction(extraction, model, EPMAfem.cuda())

    plot(EPMAfem.energy_model(model), ϵ -> EPMAfem.beam_energy_distribution(excitation, 1, ϵ))
    plot(EPMAfem.energy_model(model), ϵ -> EPMAfem.extraction_energy_distribution(extraction, 1, ϵ))
    discrete_rhs[1].bΩp |> collect |> plot
    discrete_ext[1].vector.bΩp |> collect |> plot

    plot(EPMAfem.energy_model(model), ϵ -> EPMAfem.absorption_coefficient(equations, 1, ϵ))
    plot(EPMAfem.energy_model(model), ϵ -> EPMAfem.stopping_power(equations, 1, ϵ))
    plot!(EPMAfem.energy_model(model), ϵ -> EPMAfem.stopping_power(equations, 2, ϵ))
    ylims!(0, 10)
    system_full = EPMAfem.implicit_midpoint2(problem, A -> PNLazyMatrices.schur_complement(A, Krylov.minres, PNLazyMatrices.cache ∘ LinearAlgebra.inv!));

    measurements[N] = Dict()
    measurements[N]["full"] = [ ((discrete_ext[1].vector*system_full)*discrete_rhs),
                                ((discrete_ext[2].vector*system_full)*discrete_rhs)]
    nb = EPMAfem.n_basis(model)
    nba = (p=5, m=5)
    basis_augmentation = (np=nba.p, nm=nba.m,
                          p=(U = EPMAfem.allocate_mat(EPMAfem.cuda(), nb.nx.p, nba.p),
                             V = EPMAfem.allocate_mat(EPMAfem.cuda(), nb.nΩ.p, nba.p)),
                          m=(U = EPMAfem.allocate_mat(EPMAfem.cuda(), nb.nx.m, nba.m),
                             V = EPMAfem.allocate_mat(EPMAfem.cuda(), nb.nΩ.m, nba.m)))
    # copy!(basis_augmentation.p.U[:], discrete_rhs[1].bxp |> collect |> EPMAfem.cuda())
    # copy!(basis_augmentation.p.V[:], discrete_rhs[1].bΩp |> collect |> EPMAfem.cuda())
    
    probe = EPMAfem.PNProbe(model, EPMAfem.cuda(), ϵ=ϵ->1.0)
    test = probe(system_full * discrete_rhs[1])
    
    svd_p = svd(test.p)
    svd_m = svd(test.m)

    copy!(basis_augmentation.p.U, @view(svd_p.U[:, 1:nba.p]))
    copy!(basis_augmentation.p.V, collect(svd_p.V)[:, 1:nba.p])
    copy!(basis_augmentation.m.U, @view(svd_m.U[:, 1:nba.m]))
    copy!(basis_augmentation.m.V, collect(svd_m.V)[:, 1:nba.m])

    system_lr = EPMAfem.implicit_midpoint_dlr5(problem, max_ranks=(p=50, m=50), tolerance=0.01);
    system_lr_aug = EPMAfem.implicit_midpoint_dlr5(problem, max_ranks=(p=50, m=50), tolerance=0.01, basis_augmentation=basis_augmentation);

    @gif for ((ϵ, ψ), (ϵ2, ψ2), (ϵ3, ψ3)) in zip(discrete_ext[2].vector * system_full, discrete_ext[2].vector * system_lr, discrete_ext[2].vector * system_lr_aug)
        ψp, ψm = EPMAfem.pmview(ψ, model)
        Ωp, Ωm = EPMAfem.SphericalHarmonicsModels.eval_basis(direction_model, Ω->EPMAfem.beam_direction_distribution(excitation, 1, Ω))
        func = EPMAfem.SpaceModels.interpolable((p=collect(ψp)*Ωp, m=collect(ψp)*Ωm), space_model)
        plot(-1:0.01:0, x -> func(VectorValue(x)))

        # ψ2p, ψ2m = EPMAfem.pmview(ψ2, model)
        # func2 = EPMAfem.SpaceModels.interpolable((p=collect(ψ2p)*Ωp, m=collect(ψ2p)*Ωm), space_model)
        # plot!(-1:0.01:0, x -> func2(VectorValue(x)))

        # ψ3p, ψ3m = EPMAfem.pmview(ψ3, model)
        # func3 = EPMAfem.SpaceModels.interpolable((p=collect(ψ3p)*Ωp, m=collect(ψ3p)*Ωm), space_model)
        # plot!(-1:0.01:0, x -> func3(VectorValue(x)))
    end

    for tol ∈ [0.1, 0.025, 0.00625, 0.0015625]
        system_lr = EPMAfem.implicit_midpoint_dlr5(problem, max_ranks=(p=50, m=50), tolerance=tol)
        measurements[N][("lowrank", tol)] = [ ((discrete_ext[1].vector*system_lr)*discrete_rhs),
                                    ((discrete_ext[2].vector*system_lr)*discrete_rhs)]

        system_lr_aug = EPMAfem.implicit_midpoint_dlr5(problem, max_ranks=(p=50, m=50), tolerance=tol, basis_augmentation=basis_augmentation)
        measurements[N][("lowrank_aug", tol)] = [ ((discrete_ext[1].vector*system_lr_aug)*discrete_rhs),
                                    ((discrete_ext[2].vector*system_lr_aug)*discrete_rhs)]
    end
# end

plot()
for N in [41]
    ts = [0.1, 0.025, 0.00625, 0.0015625]
    plot!(ts, [measurements[N][("lowrank", t)][1][1] for t in ts], color=1, label="lowrank")
    plot!(ts, [measurements[N][("lowrank_aug", t)][1][1] for t in ts], color=1, ls=:dot, label="lowrank_aug")
    hline!([measurements[N]["full"][1][1]], color=1, ls=:dash)

    plot!(ts, [measurements[N][("lowrank", t)][2][1] for t in ts], color=2, label="lowrank")
    plot!(ts, [measurements[N][("lowrank_aug", t)][2][1] for t in ts], color=2, ls=:dot, label="lowrank_aug")
    hline!([measurements[N]["full"][2][1]], color=2, ls=:dash)
end
plot!(xaxis=:log)



probe = EPMAfem.PNProbe(model, EPMAfem.cuda(), ϵ=ϵ->EPMAfem.beam_energy_distribution(excitation, 1, ϵ), Ω=Ω->EPMAfem.beam_direction_distribution(excitation, 1, -Ω))

func1 = EPMAfem.interpolable(probe, discrete_ext[1].vector*system_full)
func1_lr = EPMAfem.interpolable(probe, discrete_ext[1].vector*system_lowrank)

func2 = EPMAfem.interpolable(probe, discrete_ext[2].vector*system_full)
func2_lr = EPMAfem.interpolable(probe, discrete_ext[2].vector*system_lowrank)

for (i_r, tol) in enumerate([0.05, 0.01, 0.005, 0.001])
    system_lowrank = EPMAfem.implicit_midpoint_dlr5(problem.problem, max_ranks=(p=30, m=30), tolerance=tol);

    # meas = ((discrete_ext[1].vector*system_lowrank)*discrete_rhs)[:]
    meas1_lowrank[:, i_r] .= ((discrete_ext[1].vector*system_lowrank)*discrete_rhs)[:]
    # meas2_lowrank[:, i_r] .= ((discrete_ext[2].vector*system_lowrank)*discrete_rhs)[:]
end
# end
p1 = contourf(-500u"nm":1u"nm":500u"nm", -500u"nm":1u"nm":0u"nm",
        (x, z) -> -func1(VectorValue(NExt.dimless(z, equations.dim_basis), NExt.dimless(x, equations.dim_basis))), aspect_ratio=:equal, linewidth=0, clims=(3, 6))
plot!(80u"nm" .+ 80u"nm".*cos.(0:0.01:2π), -200u"nm" .+ 80u"nm".*sin.(0:0.01:2π), color=:lightgray, ls=:dash, label=nothing)
p2 = contourf(-500u"nm":1u"nm":500u"nm", -500u"nm":1u"nm":0u"nm",
        (x, z) -> -func1_lr(VectorValue(NExt.dimless(z, equations.dim_basis), NExt.dimless(x, equations.dim_basis))), aspect_ratio=:equal, linewidth=0, clims=(3, 6))
plot!(80u"nm" .+ 80u"nm".*cos.(0:0.01:2π), -200u"nm" .+ 80u"nm".*sin.(0:0.01:2π), color=:lightgray, ls=:dash, label=nothing)
plot(p1, p2, size=(1000, 500))

p1 = contourf(-500u"nm":1u"nm":500u"nm", -500u"nm":1u"nm":0u"nm",
        (x, z) -> -func2(VectorValue(NExt.dimless(z, equations.dim_basis), NExt.dimless(x, equations.dim_basis))), aspect_ratio=:equal, linewidth=0)
plot!(80u"nm" .+ 80u"nm".*cos.(0:0.01:2π), -200u"nm" .+ 80u"nm".*sin.(0:0.01:2π), color=:lightgray, ls=:dash, label=nothing)
p2 = contourf(-500u"nm":1u"nm":500u"nm", -500u"nm":1u"nm":0u"nm",
        (x, z) -> -func2_lr(VectorValue(NExt.dimless(z, equations.dim_basis), NExt.dimless(x, equations.dim_basis))), aspect_ratio=:equal, linewidth=0)
plot!(80u"nm" .+ 80u"nm".*cos.(0:0.01:2π), -200u"nm" .+ 80u"nm".*sin.(0:0.01:2π), color=:lightgray, ls=:dash, label=nothing)
plot(p1, p2, size=(1000, 500), clims=(-0.01, 0.1))

contourf(-500u"nm":1u"nm":500u"nm", -500u"nm":1u"nm":0u"nm",
        (x, z) -> -func2(VectorValue(NExt.dimless(z, equations.dim_basis), NExt.dimless(x, equations.dim_basis))), aspect_ratio=:equal, linewidth=0)
plot!(80u"nm" .+ 80u"nm".*cos.(0:0.01:2π), -200u"nm" .+ 80u"nm".*sin.(0:0.01:2π), color=:lightgray, ls=:dash, label=nothing)


using EPMAfem.SparseArrays
discrete_rhs[1].bxp |> nonzeros |> collect

A = zeros(length(discrete_rhs[1].bxp |> nonzeros), length(discrete_rhs))
for i in 1:100
    A[:, i] = collect(nonzeros(discrete_rhs[i].bxp))
end

A = zeros(40, 50)
for i in rand(1:40, 10)
    A[i, :] = rand(50)
end

spy(A)
spy(svd(A).Vt, cmap=:bluesreds)

plot()
for i in 1:26
    plot!(svd(A).U[:, i])
end
plot!(svd(A).U[:, 27])
svd(A).S[27]
plot!()
plot!(A[:, 1])

svd(A).S |> plot

[SparseVector()]


let
    p1 = plot(meas1, label=L"P_{11}", color=:black)
    plot!(meas1_lowrank[:, 1], label=L"t=0.05", color=1)
    plot!(meas1_lowrank[:, 2], label=L"t=0.025", color=2)
    plot!(meas1_lowrank[:, 3], label=L"t=0.0125", color=3)
    plot!(meas1_lowrank[:, 3], label=L"t=0.00625", color=4)

    savefig(("figures_low_rank/k_ratios1.png"))

    p2 = plot(meas2, label=L"P_{11}", color=:black)
    plot!(meas2_lowrank[:, 1], label=L"r=3", color=1)
    plot!(meas2_lowrank[:, 2], label=L"r=5", color=2)
    plot!(meas2_lowrank[:, 3], label=L"r=10", color=3)
    plot!(meas2_lowrank[:, 3], label=L"r=20", color=4)
    savefig(("figures_low_rank/k_ratios2.png"))
end
