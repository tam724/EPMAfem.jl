using Revise
using EPMAfem

using EPMAfem.PNLazyMatrices
using EPMAfem.Krylov
using EPMAfem.Gridap
using LinearAlgebra
using Plots
# using LaTeXStrings
# using BenchmarkTools
# include("plot_overloads.jl")
using NeXLCore
using NeXLCore.Unitful
using Serialization
NExt = Base.get_extension(EPMAfem, :NeXLCoreExt)


equations = NExt.epma_equations(
    [n"Al", n"Cr"],
    [NExt.EPMADetector(n"Al K-L2", VectorValue(1.0, 0.0, 0.0)), NExt.EPMADetector(n"Cr K-L2", VectorValue(1.0, 0.0, 0.0))],
    range(50u"eV", 20u"keV", length=100), 27)

meas = Dict()
Ns = [1, 5, 9, 13, 17, 21, 27]

for N in Ns
    model = NExt.epma_model(equations, (-2000u"nm", 0.0u"nm"), (10), N)

    arch = EPMAfem.cpu(Float64)
    problem = EPMAfem.discretize_problem(equations, model, arch, updatable=true)

    excitation = EPMAfem.pn_excitation([(x=0.0, y=0.0)], [NExt.dimless(15u"keV", equations.dim_basis)], [VectorValue(-1.0, 0.0, 0.0)], beam_position_σ=0.02, beam_energy_σ=0.05, beam_direction_κ=50)
    discrete_rhs = EPMAfem.discretize_rhs(excitation, model, arch)
    discrete_ext = NExt.discretize_detectors(equations, model, arch, absorption=false)
    discrete_ext[1].vector.bϵ .*= 0.01/maximum(discrete_ext[1].vector.bϵ) # (normalize TODO: there should be a general way to normalize coeffs)
    discrete_ext[2].vector.bϵ .*= 0.01/maximum(discrete_ext[2].vector.bϵ)

    function mass_concentrations(e, x_)
        # return e == 1 ? NExt.dimless(n"Al".density, equations.dim_basis) : 0.0
        z = NExt.dimful(x_[1], u"nm", equations.dim_basis)

        if (z - (-200u"nm"))^2 < (80u"nm")^2
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

    sol1 = EPMAfem.IterableDiscretePNSolution(adjoint(system_full), discrete_ext[1].vector, step_callback=(ϵ, ψ) -> @show "1", N, ϵ)
    meas[(1, N)] = (sol1 * discrete_rhs)[1]
    sol2 = EPMAfem.IterableDiscretePNSolution(adjoint(system_full), discrete_ext[2].vector, step_callback=(ϵ, ψ) -> @show "1", N, ϵ)
    meas[(2, N)] = (sol2 * discrete_rhs)[1]

    # run a forward simulation to figure out the "important directions"
    forward_solution = EPMAfem.PNProbe(model, arch, ϵ=ϵ->1.0)(EPMAfem.IterableDiscretePNSolution(system_full, discrete_rhs[1], step_callback=(ϵ, ψ) -> @show "1", N, ϵ))
    svd_forw = (p=svd(collect(forward_solution.p)), m=svd(collect(forward_solution.m)))

    if N != 1 && N != 3
        for tol in [0.025, 0.0125, 0.00625, 0.003125]
            system_lr = EPMAfem.implicit_midpoint_dlr5(problem.problem, solver=Krylov.minres, max_ranks=(p=20, m=20), tolerance=tol);
            sol1_lr = EPMAfem.IterableDiscretePNSolution(adjoint(system_lr), discrete_ext[1].vector, step_callback=(ϵ, ψ) -> @show "1", N, ϵ)
            meas[(1, N, :lr, tol)] = (sol1_lr * discrete_rhs)[1]
            sol2_lr = EPMAfem.IterableDiscretePNSolution(adjoint(system_lr), discrete_ext[2].vector, step_callback=(ϵ, ψ) -> @show "1", N, ϵ)
            meas[(2, N, :lr, tol)] = (sol2_lr * discrete_rhs)[1]

            system_lr_aug = EPMAfem.implicit_midpoint_dlr5(problem.problem, solver=Krylov.minres, max_ranks=(p=20, m=20), tolerance=tol, basis_augmentation=:mass);
            sol1_lr_aug = EPMAfem.IterableDiscretePNSolution(adjoint(system_lr_aug), discrete_ext[1].vector, step_callback=(ϵ, ψ) -> @show "1", N, ϵ)
            meas[(1, N, :lr_aug, tol)] = (sol1_lr_aug * discrete_rhs)[1]
            sol2_lr_aug = EPMAfem.IterableDiscretePNSolution(adjoint(system_lr_aug), discrete_ext[2].vector, step_callback=(ϵ, ψ) -> @show "1", N, ϵ)
            meas[(2, N, :lr_aug, tol)] = (sol2_lr_aug * discrete_rhs)[1]

            nb = EPMAfem.n_basis(model)
            basis_augmentation = (  p=(V = EPMAfem.allocate_mat(EPMAfem.architecture(problem.problem), nb.nΩ.p, 1), ),
                                    m=(V = EPMAfem.allocate_mat(EPMAfem.architecture(problem.problem), nb.nΩ.m, 1), ))
            Ωp, Ωm = EPMAfem.SphericalHarmonicsModels.eval_basis(EPMAfem.direction_model(model), one)
            basis_augmentation.m.V[:, 1] .= Ωm |> normalize |> arch
            basis_augmentation.p.V[:, 1] .= collect(discrete_rhs[1].bΩ.p) |> normalize |> arch

            system_lr_aug2 = EPMAfem.implicit_midpoint_dlr5(problem.problem, solver=Krylov.minres, max_ranks=(p=20, m=20), tolerance=tol, basis_augmentation=basis_augmentation);
            sol1_lr_aug2 = EPMAfem.IterableDiscretePNSolution(adjoint(system_lr_aug2), discrete_ext[1].vector, step_callback=(ϵ, ψ) -> @show "1", N, ϵ, ψ.ranks.p[], ψ.ranks.m[])
            meas[(1, N, :lr_aug2, tol)] = (sol1_lr_aug2 * discrete_rhs)[1]
            sol2_lr_aug2 = EPMAfem.IterableDiscretePNSolution(adjoint(system_lr_aug2), discrete_ext[2].vector, step_callback=(ϵ, ψ) -> @show "2", N, ϵ, ψ.ranks.p[], ψ.ranks.m[])
            meas[(2, N, :lr_aug2, tol)] = (sol2_lr_aug2 * discrete_rhs)[1]

            basis_augmentation = (  p=(V = EPMAfem.allocate_mat(EPMAfem.architecture(problem.problem), nb.nΩ.p, 2), ),
                                    m=(V = EPMAfem.allocate_mat(EPMAfem.architecture(problem.problem), nb.nΩ.m, 2), ))
            Ωp, Ωm = EPMAfem.SphericalHarmonicsModels.eval_basis(EPMAfem.direction_model(model), one)
            basis_augmentation.p.V[:, 1] .= discrete_rhs[1].bΩ.p
            basis_augmentation.p.V[:, 2] .= svd_forw.p.Vt[1, :] |> arch
            basis_augmentation.m.V[:, 1] .= Ωm |> arch
            basis_augmentation.m.V[:, 2] .= svd_forw.m.Vt[1, :] |> arch
            basis_augmentation.p.V .= qr(basis_augmentation.p.V).Q |> EPMAfem.mat_type(arch)
            basis_augmentation.m.V .= qr(basis_augmentation.m.V).Q |> EPMAfem.mat_type(arch)

            system_lr_aug3 = EPMAfem.implicit_midpoint_dlr5(problem.problem, solver=Krylov.minres, max_ranks=(p=20, m=20), tolerance=tol, basis_augmentation=basis_augmentation);
            sol1_lr_aug3 = EPMAfem.IterableDiscretePNSolution(adjoint(system_lr_aug3), discrete_ext[1].vector, step_callback=(ϵ, ψ) -> @show "1", N, ϵ, ψ.ranks.p[], ψ.ranks.m[])
            meas[(1, N, :lr_aug3, tol)] = (sol1_lr_aug3 * discrete_rhs)[1]
            sol2_lr_aug3 = EPMAfem.IterableDiscretePNSolution(adjoint(system_lr_aug3), discrete_ext[2].vector, step_callback=(ϵ, ψ) -> @show "2", N, ϵ, ψ.ranks.p[], ψ.ranks.m[])
            meas[(2, N, :lr_aug3, tol)] = (sol2_lr_aug3 * discrete_rhs)[1]
            serialize("meas_data.jls", meas)
        end
    end
end

serialize("meas_data.jls", meas)


begin
    plot(Ns[2:end], [meas[(1, N)] for N in Ns[2:end]], color=1, label="full P_N")
    plot!(Ns[3:end], [meas[(1, N, :lr, 0.025)] for N in Ns[3:end]], color=2, ls=:dot, label=nothing)
    plot!(Ns[3:end], [meas[(1, N, :lr_aug, 0.025)] for N in Ns[3:end]], color=3, ls=:dot, label=nothing)
    plot!(Ns[3:end], [meas[(1, N, :lr_aug2, 0.025)] for N in Ns[3:end]], color=4, ls=:dot, label=nothing)
    plot!(Ns[3:end], [meas[(1, N, :lr_aug3, 0.025)] for N in Ns[3:end]], color=5, ls=:dot, label=nothing)

    plot!(Ns[3:end], [meas[(1, N, :lr, 0.0125)] for N in Ns[3:end]], color=2, ls=:dash, label=nothing)
    plot!(Ns[3:end], [meas[(1, N, :lr_aug, 0.0125)] for N in Ns[3:end]], color=3, ls=:dash, label=nothing)
    plot!(Ns[3:end], [meas[(1, N, :lr_aug2, 0.0125)] for N in Ns[3:end]], color=4, ls=:dash, label=nothing)
    plot!(Ns[3:end], [meas[(1, N, :lr_aug3, 0.0125)] for N in Ns[3:end]], color=5, ls=:dash, label=nothing)

    plot!(Ns[3:end], [meas[(1, N, :lr, 0.00625)] for N in Ns[3:end]], color=2, ls=:dashdot, label=nothing)
    plot!(Ns[3:end], [meas[(1, N, :lr_aug, 0.00625)] for N in Ns[3:end]], color=3, ls=:dashdot, label=nothing)
    plot!(Ns[3:end], [meas[(1, N, :lr_aug2, 0.00625)] for N in Ns[3:end]], color=4, ls=:dashdot, label=nothing)
    plot!(Ns[3:end], [meas[(1, N, :lr_aug3, 0.00625)] for N in Ns[3:end]], color=5, ls=:dashdot, label=nothing)

    plot!(Ns[3:end], [meas[(1, N, :lr, 0.003125)] for N in Ns[3:end]], color=2, ls=:dashdotdot, label=nothing)
    plot!(Ns[3:end], [meas[(1, N, :lr_aug, 0.003125)] for N in Ns[3:end]], color=3, ls=:dashdotdot, label=nothing)
    plot!(Ns[3:end], [meas[(1, N, :lr_aug2, 0.003125)] for N in Ns[3:end]], color=4, ls=:dashdotdot, label=nothing)
    plot!(Ns[3:end], [meas[(1, N, :lr_aug3, 0.003125)] for N in Ns[3:end]], color=5, ls=:dashdotdot, label=nothing)
    plot!([], [], color=2, label="lr default")
    plot!([], [], color=3, label="lr mass cons")
    plot!([], [], color=4, label="lr mass + out cons")
    plot!([], [], color=5, label="lr mass + out + forw cons")
    plot!([], [], color=:gray, ls=:dot, label="tol=0.025")
    plot!([], [], color=:gray, ls=:dash, label="tol=0.0125")
    plot!([], [], color=:gray, ls=:dashdot, label="tol=0.00625")
    plot!([], [], color=:gray, ls=:dashdotdot, label="tol=0.003125")
    plot!(size=(400, 300), dpi=1000, fontfamily="Computer Modern", legend_columns=2)
    savefig("test.png")
end

begin
    plot(Ns[2:end], [meas[(2, N)] for N in Ns[2:end]], color=1, label="full P_N")
    plot!(Ns[3:end], [meas[(2, N, :lr, 0.025)] for N in Ns[3:end]], color=2, ls=:dot, label=nothing)
    plot!(Ns[3:end], [meas[(2, N, :lr_aug, 0.025)] for N in Ns[3:end]], color=3, ls=:dot, label=nothing)
    plot!(Ns[3:end], [meas[(2, N, :lr_aug2, 0.025)] for N in Ns[3:end]], color=4, ls=:dot, label=nothing)
    plot!(Ns[3:end], [meas[(2, N, :lr_aug3, 0.025)] for N in Ns[3:end]], color=5, ls=:dot, label=nothing)

    plot!(Ns[3:end], [meas[(2, N, :lr, 0.0125)] for N in Ns[3:end]], color=2, ls=:dash, label=nothing)
    plot!(Ns[3:end], [meas[(2, N, :lr_aug, 0.0125)] for N in Ns[3:end]], color=3, ls=:dash, label=nothing)
    plot!(Ns[3:end], [meas[(2, N, :lr_aug2, 0.0125)] for N in Ns[3:end]], color=4, ls=:dash, label=nothing)
    plot!(Ns[3:end], [meas[(2, N, :lr_aug3, 0.0125)] for N in Ns[3:end]], color=5, ls=:dash, label=nothing)

    plot!(Ns[3:end], [meas[(2, N, :lr, 0.00625)] for N in Ns[3:end]], color=2, ls=:dashdot, label=nothing)
    plot!(Ns[3:end], [meas[(2, N, :lr_aug, 0.00625)] for N in Ns[3:end]], color=3, ls=:dashdot, label=nothing)
    plot!(Ns[3:end], [meas[(2, N, :lr_aug2, 0.00625)] for N in Ns[3:end]], color=4, ls=:dashdot, label=nothing)
    plot!(Ns[3:end], [meas[(2, N, :lr_aug3, 0.00625)] for N in Ns[3:end]], color=5, ls=:dashdot, label=nothing)

    plot!(Ns[3:end], [meas[(2, N, :lr, 0.003125)] for N in Ns[3:end]], color=2, ls=:dashdotdot, label=nothing)
    plot!(Ns[3:end], [meas[(2, N, :lr_aug, 0.003125)] for N in Ns[3:end]], color=3, ls=:dashdotdot, label=nothing)
    plot!(Ns[3:end], [meas[(2, N, :lr_aug2, 0.003125)] for N in Ns[3:end]], color=4, ls=:dashdotdot, label=nothing)
    plot!(Ns[3:end], [meas[(2, N, :lr_aug3, 0.003125)] for N in Ns[3:end]], color=5, ls=:dashdotdot, label=nothing)
    plot!([], [], color=2, label="lr default")
    plot!([], [], color=3, label="lr mass cons")
    plot!([], [], color=4, label="lr mass + out cons")
    plot!([], [], color=5, label="lr mass + out + forw cons")
    plot!([], [], color=:gray, ls=:dot, label="tol=0.025")
    plot!([], [], color=:gray, ls=:dash, label="tol=0.0125")
    plot!([], [], color=:gray, ls=:dashdot, label="tol=0.00625")
    plot!([], [], color=:gray, ls=:dashdotdot, label="tol=0.003125")
    plot!(size=(400, 300), dpi=1000, fontfamily="Computer Modern", legend_columns=2)
    savefig("test2.png")
end
