using Revise
using EPMAfem

using EPMAfem.PNLazyMatrices
using EPMAfem.Krylov
using EPMAfem.Gridap
using LinearAlgebra
using Plots

using NeXLCore
using NeXLCore.Unitful
using Serialization
NExt = Base.get_extension(EPMAfem, :NeXLCoreExt)

equations = NExt.epma_equations(
    [n"Al", n"Cr"],
    [NExt.EPMADetector(n"Al K-L2", VectorValue(1.0, 0.0, 0.0)), NExt.EPMADetector(n"Cr K-L2", VectorValue(1.0, 0.0, 0.0))],
    range(50u"eV", 20u"keV", length=200), 35)

model = NExt.epma_model(equations, (-2000u"nm", 0.0u"nm"), (200), 35)

arch = EPMAfem.cpu(Float64)
problem = EPMAfem.discretize_problem(equations, model, arch, updatable=true)

excitation = EPMAfem.pn_excitation([(x=0.0, y=0.0)], [NExt.dimless(15u"keV", equations.dim_basis)], [VectorValue(-1.0, 0.0, 0.0)], beam_position_σ=0.02, beam_energy_σ=0.05, beam_direction_κ=50)
discrete_rhs = EPMAfem.discretize_rhs(excitation, model, arch)
discrete_ext = NExt.discretize_detectors(equations, model, arch, absorption=false)
discrete_ext[1].vector.bϵ .*= 0.01/maximum(discrete_ext[1].vector.bϵ) # (normalize TODO: there should be a general way to normalize coeffs)
discrete_ext[2].vector.bϵ .*= 0.01/maximum(discrete_ext[2].vector.bϵ)

function mass_concentrations(e, x_)
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

sol1 = EPMAfem.IterableDiscretePNSolution(adjoint(system_full), discrete_ext[1].vector, step_callback=(ϵ, ψ) -> @show "1", ϵ)
meas_full1 = (sol1 * discrete_rhs)[1]
sol2 = EPMAfem.IterableDiscretePNSolution(adjoint(system_full), discrete_ext[2].vector, step_callback=(ϵ, ψ) -> @show "2", ϵ)
meas_full2 = (sol2 * discrete_rhs)[1]

system_lr = EPMAfem.implicit_midpoint_dlr5(problem.problem, max_ranks=(p=20, m=20))
sol_forward = EPMAfem.IterableDiscretePNSolution(system_lr, discrete_rhs[1])
directional_basis = Dict()
Ωp, Ωm = EPMAfem.SphericalHarmonicsModels.eval_basis(EPMAfem.direction_model(model), one)
@gif for (ϵ, ψ) in sol_forward
    ψp, ψm = EPMAfem.pmview(ψ, model)
    ψm*Ωm
    ff = EPMAfem.SpaceModels.interpolable((p=ψp*zeros(size(ψp, 2)), m=ψm*Ωm), EPMAfem.space_model(model))
    plot(-2000u"nm":1u"nm":0u"nm", x -> ff(VectorValue(NExt.dimless(x, equations.dim_basis))))
    @show ϵ
    ((_, _, Vtp), (_, _, Vtm)) = EPMAfem.USVt(ψ)
    directional_basis[(ϵ, :p)] = copy(Vtp)
    directional_basis[(ϵ, :m)] = copy(Vtm)
end

measurements1 = zeros(4, 10)
measurements2 = zeros(4, 10)
n_r = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for (i_r, n_r) in enumerate(n_r)
    let
        nb = EPMAfem.n_basis(problem.problem)
        basis_augmentation = (p=(V = EPMAfem.allocate_mat(EPMAfem.architecture(problem.problem), nb.nΩ.p, n_r), ),
                            m=(V = EPMAfem.allocate_mat(EPMAfem.architecture(problem.problem), nb.nΩ.m, n_r), ))
        function update_basis_augmentation!(b_aug, ϵ)
            copy!(b_aug.p.V, transpose(directional_basis[(ϵ, :p)][1:n_r, :]))
            copy!(b_aug.m.V, transpose(directional_basis[(ϵ, :m)][1:n_r, :]))
            return
        end
        ϵ0 = EPMAfem.plus½(EPMAfem.first_index(EPMAfem.energy_model(model), true))
        update_basis_augmentation!(basis_augmentation, ϵ0)
        system_lr_aug = EPMAfem.implicit_midpoint_dlr5(problem.problem, max_ranks=(p=2n_r, m=2n_r), basis_augmentation=basis_augmentation);
        sol1 = EPMAfem.IterableDiscretePNSolution(adjoint(system_lr_aug), discrete_ext[1].vector, step_callback=(ϵ, ψ) -> begin @show ϵ; update_basis_augmentation!(basis_augmentation, EPMAfem.plus½(ϵ)) end);
        measurements1[1, i_r] = (sol1 * discrete_rhs)[1]
        sol2 = EPMAfem.IterableDiscretePNSolution(adjoint(system_lr_aug), discrete_ext[2].vector, step_callback=(ϵ, ψ) -> begin @show ϵ; update_basis_augmentation!(basis_augmentation, EPMAfem.plus½(ϵ)) end);
        measurements2[1, i_r] = (sol2 * discrete_rhs)[1]
    end

    let
        system_lr_noaug = EPMAfem.implicit_midpoint_dlr5(problem.problem, max_ranks=(p=2n_r, m=2n_r));
        sol1 = EPMAfem.IterableDiscretePNSolution(adjoint(system_lr_noaug), discrete_ext[1].vector, step_callback=(ϵ, ψ) -> @show ϵ);
        measurements1[2, i_r] = (sol1 * discrete_rhs)[1]
        sol2 = EPMAfem.IterableDiscretePNSolution(adjoint(system_lr_noaug), discrete_ext[2].vector, step_callback=(ϵ, ψ) -> @show ϵ);
        measurements2[2, i_r] = (sol2 * discrete_rhs)[1]
    end

    let
        system_lr_mass = EPMAfem.implicit_midpoint_dlr5(problem.problem, max_ranks=(p=2n_r, m=2n_r), basis_augmentation=:mass);
        sol1 = EPMAfem.IterableDiscretePNSolution(adjoint(system_lr_mass), discrete_ext[1].vector, step_callback=(ϵ, ψ) -> @show ϵ);
        measurements1[3, i_r] = (sol1 * discrete_rhs)[1]
        sol2 = EPMAfem.IterableDiscretePNSolution(adjoint(system_lr_mass), discrete_ext[2].vector, step_callback=(ϵ, ψ) -> @show ϵ);
        measurements2[3, i_r] = (sol2 * discrete_rhs)[1]
    end

    let
        nb = EPMAfem.n_basis(model)
        basis_augmentation = (  p=(V = EPMAfem.allocate_mat(EPMAfem.architecture(problem.problem), nb.nΩ.p, 1), ),
                                m=(V = EPMAfem.allocate_mat(EPMAfem.architecture(problem.problem), nb.nΩ.m, 1), ))
        Ωp, Ωm = EPMAfem.SphericalHarmonicsModels.eval_basis(EPMAfem.direction_model(model), one)
        basis_augmentation.m.V[:, 1] .= Ωm |> normalize |> arch
        basis_augmentation.p.V[:, 1] .= collect(discrete_rhs[1].bΩ.p) |> normalize |> arch
        system_lr_mass_ext = EPMAfem.implicit_midpoint_dlr5(problem.problem, max_ranks=(p=2n_r, m=2n_r), basis_augmentation=basis_augmentation);
        sol1 = EPMAfem.IterableDiscretePNSolution(adjoint(system_lr_mass_ext), discrete_ext[1].vector, step_callback=(ϵ, ψ) -> @show ϵ);
        measurements1[4, i_r] = (sol1 * discrete_rhs)[1]
        sol2 = EPMAfem.IterableDiscretePNSolution(adjoint(system_lr_mass_ext), discrete_ext[2].vector, step_callback=(ϵ, ψ) -> @show ϵ);
        measurements2[4, i_r] = (sol2 * discrete_rhs)[1]
    end
end

figpath = mkpath(joinpath(dirname(@__FILE__), "figures/1D_epma_measurements_error"))

begin
    r_max = 8
    plot(2n_r[1:r_max], abs.(measurements1[2, 1:r_max].- meas_full1) / meas_full1, label="default")
    plot!(2n_r[1:r_max], abs.(measurements1[3, 1:r_max].- meas_full1) / meas_full1, label="mass")
    plot!(2n_r[1:r_max], abs.(measurements1[4, 1:r_max].- meas_full1) / meas_full1, label="mass + bc")
    plot!(2n_r[1:r_max], abs.(measurements1[1, 1:r_max] .- meas_full1) / meas_full1, yaxis=:log, label="adjoint error")
    plot!(yticks=[1e-2, 1e-4, 1e-6, 1e-8], xticks=2n_r)
    ylabel!("error (rel.)")
    xlabel!("ranks")
    plot!(size=(400, 300), dpi=1000, fontfamily="Computer Modern")
    savefig(joinpath(figpath, "error_measurement_Al.png"))

    plot(2n_r[1:r_max], abs.(measurements2[2, 1:r_max].- meas_full2) / meas_full2, label="default")
    plot!(2n_r[1:r_max], abs.(measurements2[3, 1:r_max].- meas_full2) / meas_full2, label="mass")
    plot!(2n_r[1:r_max], abs.(measurements2[4, 1:r_max].- meas_full2) / meas_full2, label="mass + bc")
    plot!(2n_r[1:r_max], abs.(measurements2[1, 1:r_max] .- meas_full2) / meas_full2, yaxis=:log, label="adjoint error")
    plot!(yticks=[1e-2, 1e-4, 1e-6, 1e-8], xticks=2n_r)
    ylabel!("error (rel.)")
    xlabel!("ranks")
    plot!(size=(400, 300), dpi=1000, fontfamily="Computer Modern")
    savefig(joinpath(figpath, "error_measurement_Cr.png"))
end
