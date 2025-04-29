using Revise
using EPMAfem
using Plots
using Gridap
using ConcreteStructs
using Distributions
using LinearAlgebra

@concrete struct PN1DEquations <: EPMAfem.AbstractPNEquations end
# 1: purely absorbing material
# 2: scattering material
EPMAfem.number_of_elements(::PN1DEquations) = 1
EPMAfem.number_of_scatterings(::PN1DEquations) = 1
EPMAfem.stopping_power(::PN1DEquations, e, ϵ) = 1.0
EPMAfem.absorption_coefficient(::PN1DEquations, e, ϵ) = 0.0
EPMAfem.scattering_coefficient(::PN1DEquations, e, i, ϵ) = 0.0
EPMAfem.scattering_kernel(eq::PN1DEquations, e, i) = μ -> 0.0
EPMAfem.mass_concentrations(::PN1DEquations, e, (x, )) = 1.0
initial_space_distribution(::PN1DEquations, (x,)) = tanh(-100*x)*0.5 + 0.5

function discretize_rhs(eq::PN1DEquations, mdl::EPMAfem.PNGridapModel, arch::EPMAfem.PNArchitecture)
    T = EPMAfem.base_type(arch)
    SM = EPMAfem.SpaceModels
    SH = EPMAfem.SphericalHarmonicsModels

    (nϵ, (nxp, nxm), (nΩp, nΩm)) = EPMAfem.n_basis(mdl)

    space_mdl = EPMAfem.space_model(mdl)

    # boundary condition
    direction_mdl = EPMAfem.direction_model(mdl)
    ## assemble excitation 
    # gϵ = Vector{T}([tanh(-100(ϵ-2.6))*0.5+0.5 for ϵ ∈ EPMAfem.energy_model(mdl)])
    # gϵ = Vector{T}([exp(-30*(ϵ-2.5)^2) for ϵ ∈ EPMAfem.energy_model(mdl)])
    gϵ = Vector{T}([1.0 for ϵ ∈ EPMAfem.energy_model(mdl)])
    gxp = SM.assemble_linear(SM.∫∂R_ngv{EPMAfem.Dimensions.Z}(x -> x[1] < 0.0 ? -1.0 : 0.0), space_mdl, SM.even(space_mdl))  |> arch
    nz = EPMAfem.Dimensions.cartesian_unit_vector(EPMAfem.Dimensions.Z(), EPMAfem.dimensionality(mdl))
    nz3D = EPMAfem.Dimensions.extend_3D(nz)
    # gΩp = SH.assemble_linear(SH.∫S²_nΩgv(nz3D, Ω -> 2/sqrt(π)), direction_mdl, SH.even(direction_mdl)) |> arch
    gΩp = SH.assemble_linear(SH.∫S²_nΩgv(nz3D, Ω -> dot(nz3D, Ω) < 0 ? 1.0 : 0.0), direction_mdl, SH.even(direction_mdl), SH.lebedev_quadrature_max()) |> arch

    return EPMAfem.Rank1DiscretePNVector(false, mdl, arch, gϵ, gxp, gΩp)
end

function discretize_initial_state(eq::PN1DEquations, mdl::EPMAfem.PNGridapModel, arch::EPMAfem.PNArchitecture)
    T = EPMAfem.base_type(arch)
    SM = EPMAfem.SpaceModels
    SH = EPMAfem.SphericalHarmonicsModels

    (nϵ, (nxp, nxm), (nΩp, nΩm)) = EPMAfem.n_basis(mdl)

    space_mdl = EPMAfem.space_model(mdl)
        # initial state
    # ψ0xp = SM.assemble_linear(SM.∫R_μv(x -> initial_space_distribution(eq, x)), space_mdl, SM.even(space_mdl)) |> arch
    ψ0xp = SM.projection(x -> initial_space_distribution(eq, x), space_mdl, SM.even(space_mdl))

    ψ0Ωp = zeros(nΩp)
    ψ0Ωp[1] = 1.0

    ψ0 = EPMAfem.allocate_vec(arch, nxp*nΩp+nxm*nΩm)

    ψ0p = EPMAfem.pview(ψ0, mdl)
    ψ0m = EPMAfem.mview(ψ0, mdl)

    mul!(ψ0p, reshape(ψ0xp, (nxp, 1)), reshape(ψ0Ωp |> arch, (1, nΩp)))
    EPMAfem.my_rmul!(ψ0m, false)
    return ψ0
end

function compute(nx)
    eq = PN1DEquations()
    space_model = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((-1, 1), nx))
    direction_model = EPMAfem.SphericalHarmonicsModels.EEEOSphericalHarmonicsModel(29, 1)

    mdl = EPMAfem.PNGridapModel(space_model, range(0, 0.8, length=100), direction_model)
    discrete_problem = EPMAfem.discretize_problem(eq, mdl, EPMAfem.cpu())
    discrete_system = EPMAfem.schurimplicitmidpointsystem(discrete_problem)

    rhs = discretize_initial(eq, mdl, EPMAfem.cpu())

    solution = EPMAfem.iterator(discrete_system, rhs)

    integral = zeros(length(EPMAfem.energy_model(mdl)))
    for (ϵ_i, i) in solution
        @show i
        ψ = EPMAfem.current_solution(discrete_system)
        ψ0 = reshape(EPMAfem.pview(ψ, mdl)[:, 1] |> collect, (nx+1, ))
        integral[i] = sum(ψ0)
    end
    ψ = EPMAfem.current_solution(discrete_system)
    return reverse(integral), reshape(EPMAfem.pview(ψ, mdl)[:, 1] |> collect, (nx+1, ))
end

eq = PN1DEquations()
nx = 500
space_model = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((-1, 1), nx))
direction_model = EPMAfem.SphericalHarmonicsModels.EEEOSphericalHarmonicsModel(3, 1)

mdl = EPMAfem.PNGridapModel(space_model, range(0, 1.0, length=300), direction_model)
discrete_problem = EPMAfem.discretize_problem(eq, mdl, EPMAfem.cpu())

discrete_problem = EPMAfem.discretize_problem(eq, mdl, EPMAfem.cpu())
discrete_system = EPMAfem.schurimplicitmidpointsystem(discrete_problem)

rhs = discretize_rhs(eq, mdl, EPMAfem.cpu())

rhs.bΩp ./= sqrt(π) # /2 * 2 (because the rhs should at some point be multiplied by 2)


initial_state = discretize_initial_state(eq, mdl, EPMAfem.cpu())

#fill!(initial_state, 0.0)

solution = EPMAfem.DiscretePNIterator(discrete_system, rhs, false, initial_state)

integral = zeros(length(EPMAfem.energy_model(mdl)))
@gif for (idx, sol) in solution
    @show idx, size(sol)
    ψ0 = reshape(EPMAfem.pview(sol, mdl)[:, 1] |> collect, (nx+1, ))
    integral[idx.i] = sum(ψ0)
    plot(range(-1, 1, length=nx+1), ψ0)
    vline!(((EPMAfem.ϵ(idx) .- 1.0)).*[0.339981, -0.339981, 0.86113, -0.86113])
    ylims!(-0.1, 1.5)
end every 5

function solve()
    for idx in solution end
end

@profview solve()
@benchmark solve()

A = discrete_problem.ρp[1]
A_sym = Symmetric(A)
AT = Transpose(A)

x = rand(3001, 30)
y = zeros(3001, 30)

xT = transpose(x) |> collect
yT = transpose(y) |> collect

@benchmark mul!($y, $A, $x)
@benchmark mul!($y, $A_sym, $x)
@benchmark mul!($y, $AT, $x)
@benchmark mul!($yT, $xT, $A)

ψ = EPMAfem.current_solution(discrete_system)
return reverse(integral), reshape(EPMAfem.pview(ψ, mdl)[:, 1] |> collect, (nx+1, ))

using BenchmarkTools

@benchmark LinearAlgebra.BLAS.nrm2($18003, $initial_state, $1)
@benchmark LinearAlgebra.norm2($initial_state)


L2_norm50, sol = compute(50)
L2_norm100, _ = compute(100)
L2_norm200, sol200 = compute(200)

scatter(range(0, 0.5, length=25), L2_norm50, label="50x50")
scatter!(range(0, 0.5, length=25), L2_norm100, label="100x100")
scatter!(range(0, 0.5, length=25), L2_norm200, label="200x200")
