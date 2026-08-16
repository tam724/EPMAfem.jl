using Revise
using EPMAfem
using MathLink
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

struct TestExcitations end
EPMAfem.number_of_beam_energies(eq::TestExcitations) = 1
EPMAfem.number_of_beam_positions(eq::TestExcitations) = 1
EPMAfem.number_of_beam_directions(eq::TestExcitations) = 10

function EPMAfem.beam_space_distribution(::TestExcitations, i, (z, x, y)) # y is unused, z is 0
    return isapprox(0.0, z, atol=1e-12)
end

function EPMAfem.beam_direction_distribution(::TestExcitations, i, Ω)
    κs = range(10.0, 100.0, 10)
    # HACK! smuggle the 2 in here
    # HACK! divide by n*Omega to compensate
    return 2.0*pdf(VonMisesFisher([-1.0, 0.0, 0.0], κs[i]), [Ω...]) #/ abs(dot([-1.0, 0.0, 0.0], Ω))
end

function EPMAfem.beam_energy_distribution(::TestExcitations, i, ϵ)
    return pdf(Uniform(0.8, 1.0), ϵ)
end


# forward_plots
equations = TestEquations()

energy_model = range(0, 1, 100)
space_model = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((0.0, 1.0), (100)), plus=(name=lagrangian, order=1, conformity=:H1), minus=(name=lagrangian, order=0, conformity=:L2))
direction_model = EPMAfem.SphericalHarmonicsModels.EOSphericalHarmonicsModel(21, 1, :EO)
model = EPMAfem.DiscretePNModel(space_model, energy_model, direction_model)

problem = EPMAfem.discretize_problem(equations, model, EPMAfem.cpu())
rhs = EPMAfem.discretize_rhs(TestExcitations(), model, EPMAfem.cpu())

m1 = direction_model.moments[1]
m2 = EPMAfem.SphericalHarmonicsModels.odd(direction_model)[1]



[EPMAfem.SphericalHarmonicsModels.get_cached_boundary_coefficient(m1, m2, EPMAfem.Dimensions.Z()) for m1 in direction_model.moments, m2 in direction_model.moments]


probe = EPMAfem.PNProbe(model, EPMAfem.cpu(), ϵ=ϵ->1.0, Ω=Ω->1.0)

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




ML = Base.get_extension(EPMAfem, :MathLinkExt)

temp = [ML.get_boundary_coefficient_symbolic(m1, m2, EPMAfem.Dimensions.X()) |> ML.w_num
    for m1 in moms, m2 in moms]

temp2 = [try SH.get_cached_boundary_coefficient(m1, m2, EPMAfem.Dimensions.X()) catch NaN end for m1 in moms, m2 in moms]
temp2 = [isnothing(t) : Float64(NaN) : Float64(t) for t in temp2]

ML.populate_boundary_dict_spherical_harmonics(3, 61; test=true, dim=EPMAfem.Dimensions._1D())
EPMAfem.SphericalHarmonicsModels.serialize_boundary_dicts()
ML.populate_boundary_dict_spherical_harmonics(3, 71; test=false, dim=EPMAfem.Dimensions._1D())
EPMAfem.SphericalHarmonicsModels.serialize_boundary_dicts()
ML.populate_boundary_dict_spherical_harmonics(3, 81; test=false, dim=EPMAfem.Dimensions._1D())
EPMAfem.SphericalHarmonicsModels.serialize_boundary_dicts()
ML.populate_boundary_dict_spherical_harmonics(3, 91; test=false, dim=EPMAfem.Dimensions._1D())
EPMAfem.SphericalHarmonicsModels.serialize_boundary_dicts()
