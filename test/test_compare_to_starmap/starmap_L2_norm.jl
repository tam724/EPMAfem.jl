
@concrete struct ExpandingGaussianEquations <: EPMAfem.AbstractPNEquations end
# 1: purely absorbing material
# 2: scattering material
EPMAfem.number_of_elements(::ExpandingGaussianEquations) = 1
EPMAfem.number_of_scatterings(::ExpandingGaussianEquations) = 1
EPMAfem.stopping_power(::ExpandingGaussianEquations, e, ϵ) = 1.0
EPMAfem.absorption_coefficient(::ExpandingGaussianEquations, e, ϵ) = 0.0
EPMAfem.scattering_coefficient(::ExpandingGaussianEquations, e, i, ϵ) = 0.0
EPMAfem.scattering_kernel(eq::ExpandingGaussianEquations, e, i) = μ -> 0.0
EPMAfem.mass_concentrations(::ExpandingGaussianEquations, e, (x, y)) = 1.0
initial_space_distribution(::ExpandingGaussianEquations, (x, y), σ=1e-2) = 1/(4π*σ)*exp(-(x*x+y*y)/(4σ))

function discretize_initial(eq::ExpandingGaussianEquations, mdl::EPMAfem.PNGridapModel, arch::EPMAfem.PNArchitecture)
    T = EPMAfem.base_type(arch)
    SM = EPMAfem.SpaceModels

    (nϵ, (nxp, nxm), (nΩp, nΩm)) = EPMAfem.n_basis(mdl)

    space_mdl = EPMAfem.space_model(mdl)

    ψ0xp = SM.assemble_linear(SM.∫R_μv(x -> initial_space_distribution(eq, x)), space_mdl, SM.even(space_mdl)) |> arch
    ψ0Ωp = zeros(nΩp)
    ψ0Ωp[1] = 1.0

    ψ0 = EPMAfem.allocate_vec(arch, nxp*nΩp+nxm*nΩm)

    ψ0p = EPMAfem.pview(ψ0, mdl)
    ψ0m = EPMAfem.mview(ψ0, mdl)

    mul!(ψ0p, reshape(ψ0xp, (nxp, 1)), reshape(ψ0Ωp |> arch, (1, nΩp)))
    EPMAfem.my_rmul!(ψ0m, false)

    return ψ0, EPMAfem.Rank1DiscretePNVector(false, mdl, arch, zeros(T, nϵ), zeros(T, nxp)|> arch, zeros(T, nΩp))
end

function compute(nx)
    eq = ExpandingGaussianEquations()
    space_model = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((-1, 1, -1, 1), (nx, nx)))
    direction_model = EPMAfem.SphericalHarmonicsModels.EEEOSphericalHarmonicsModel(5, 2)

    mdl = EPMAfem.PNGridapModel(space_model, range(0, 0.5, length=25), direction_model)
    discrete_problem = EPMAfem.discretize_problem(eq, mdl, EPMAfem.cpu())
    discrete_system = EPMAfem.schurimplicitmidpointsystem(discrete_problem)

    initial_state, rhs = discretize_initial(eq, mdl, EPMAfem.cpu())

    solution = EPMAfem.IterableDiscretePNSolution(discrete_system, rhs; initial_solution=initial_state)
    # @show solution |> typeof

    integral = zeros(length(EPMAfem.energy_model(mdl)))
    for (ϵidx, ψ) in solution
        @show ϵidx
        ψ0 = reshape(EPMAfem.pview(ψ, mdl)[:, 1] |> collect, (nx+1, nx+1))
        integral[ϵidx.i] = sum(ψ0)
    end
    ψ = solution.current_solution
    return reverse(integral), reshape(EPMAfem.pview(ψ, mdl)[:, 1] |> collect, (nx+1, nx+1))
end

L2_norm50, _ = compute(50)
L2_norm100, _ = compute(100)
L2_norm200, _sol200 = compute(200)

scatter(range(0, 0.5, length=25), L2_norm50, label="50x50")
scatter!(range(0, 0.5, length=25), L2_norm100, label="100x100")
scatter!(range(0, 0.5, length=25), L2_norm200, label="200x200")

heatmap(_sol200, aspect_ratio=:equal, cmap=:roma)
# @gif for (i, ϵ) in solution
#     @show ϵ
#     ψ = EPMAfem.current_solution(discrete_system)
#     ψ0 = reshape(EPMAfem.pview(ψ, model)[:, 1] |> collect, (101, 101))
#     heatmap(ψ0, aspect_ratio=:equal)
# end

# ψ0 = reshape(EPMAfem.pview(EPMAfem.current_solution(discrete_system), model)[:, 1] |> collect, (101, 101))
# heatmap(log.(fix0.(ψ0))', aspect_ratio=:equal, clim=(-7, 0), color=:jet)