using Revise
using NeXLCore
using EPMAfem
using NeXLCore.Unitful
using Plots
using Dimensionless
using EPMAfem.SphericalHarmonicsModels.LegendrePolynomials

NExt = Base.get_extension(EPMAfem, :NeXLCoreExt)

plot(0:0.01:2π, θ -> dimless(NeXLCore.δσδΩ(NeXLCore.ScreenedRutherford, θ, n"Cu", ϵ_range[1]/u"eV" |> upreferred)*u"cm"^2, eq.dim_basis))

f(μ) = dimless(NeXLCore.δσδΩ(NeXLCore.ScreenedRutherford, acos(μ), n"Cu", ϵ_range[end]/u"eV" |> upreferred)*u"cm"^2, eq.dim_basis)
f_lg = EPMAfem.SphericalHarmonicsModels.expand_legendre(f, 35, EPMAfem.SphericalHarmonicsModels.hcubature_quadrature(1e-9, 1e-9))



plot(-1:0.001:1, f)
plot!(-1:0.001:1, μ -> f_lg(μ))

SH = EPMAfem.SphericalHarmonicsModels
direction_mdl = SH.EEEOSphericalHarmonicsModel(12, 2)
A1 = SH.assemble_bilinear(SH.∫S²_kuv(μ->f(μ)), direction_mdl, SH.even(direction_mdl), SH.even(direction_mdl), SH.hcubature_quadrature(1e-9, 1e-9))
A2 = SH.assemble_bilinear(SH.∫S²_kuv(f_lg), direction_mdl, SH.even(direction_mdl), SH.even(direction_mdl), SH.hcubature_quadrature(1e-9, 1e-9))

# trans = NExt.compute_unit_transformation(4.0u"keV", 10u"keV")
# ϵ_trans = NExt.compute_unit_transformation(4.0u"keV", 10u"keV")
# x_trans = NExt.compute_unit_transformation(-500u"nm", 0u"nm")

quant = 10.0u"keV"

# quant_unitless = NExt.without_units(quant, ϵ_trans)
# quant_unitful = NExt.with_units(quant_unitless, ϵ_trans)

# dim_basis = NExt.construct_dim_basis((-100u"nm", 0u"nm"), (4u"keV", 10u"keV"), 9u"g"/u"cm"^3)
# dim_basis.basis_M |> upreferred
# dim_basis.basis_L |> upreferred
# dim_basis.basis_T |> upreferred

quant / dim_basis.basis_E

using Dimensionless

bas = DimBasis(100u"nm", 6u"keV", 9u"g"/u"cm"^3)

val_dimless = dimless(4u"keV", bas)
dimful(val_dimless, u"keV", bas)

13u"eV" / 1u"eV" |> upreferred

mat = [n"Cu", n"Ni"]
ϵ_range = range(5.0u"keV", 15.0u"keV", length=35)
eq = NExt.epma_equations(mat, ϵ_range, 35, 4);

ϵ_range_dimless = dimless(ϵ_range, eq.dim_basis)

plot(ϵ_range_dimless, ϵ -> NExt.stopping_power(eq, 1, ϵ))
plot!(ϵ_range_dimless, ϵ -> NExt.σₜ_recon(eq, 1, ϵ))

NExt.σₜ_recon(eq, 1, dimless(ϵ_range[1], eq.dim_basis))
dimless(NeXLCore.σₜ(NeXLCore.ScreenedRutherford, n"Cu", ϵ_range[1]/u"eV" |> upreferred)*u"cm"^2, eq.dim_basis)


my_dimless(q) = dimless(q, eq.dim_basis)

NeXLCore.σₜ(NeXLCore.ScreenedRutherford, n"Cu", 5000.0)u"cm"^2 / n"Cu".atomic_mass |> my_dimless
# -NeXLCore.dEds(NeXLCore.Bethe, 1000.0, n"Cu", n"Cu".density / (u"g"/u"cm"^3) |> upreferred)u"eV"/u"cm" |> my_dimless
-NeXLCore.dEds(NeXLCore.Bethe, 1000.0, n"Cu", 1.0)u"eV"*u"cm"^2/u"g"  |> my_dimless

NExt.stopping_power(eq, 1, dimless(1u"keV", eq.dim_basis))
NExt.σₜ_recon(eq, 1, dimless(5u"keV", eq.dim_basis))

NExt.electron_scattering_kernel_f(eq, 1, 3)



s(ϵ) = -NeXLCore.dEds(NeXLCore.Bethe, ϵ/u"eV" |> upreferred, n"Cu", 1.0)u"eV"*u"cm"^2/u"g"
(-s(1.0u"keV" + 0.0001u"keV") - (-s(1.0u"keV")))/(0.0001u"keV") |> my_dimless

plot()
for ϵ in ϵ_range[1:1]
    ϵ_dimless = dimless(ϵ, eq.dim_basis)
    plot!(-1:0.01:1, μ -> NExt.δσδΩ_recon(eq, μ, 1, ϵ_dimless))
    plot!(-1:0.01:1, μ -> dimless(NeXLCore.δσδΩ(NeXLCore.ScreenedRutherford, acos(μ), n"Cu", ϵ_range[1]/u"eV" |> upreferred)*u"cm"^2, eq.dim_basis))
end
plot!()

NExt.stopping_power(eq, 1, 1.0)

plot(1u"keV":0.1u"keV":10u"keV", x -> -NExt.stopping_power(eq, 1, dimless(x, eq.dim_basis)))

plot(0:0.01:2π, θ -> NeXLCore.δσδΩ(NeXLCore.ScreenedRutherford, θ, n"Cu", 1000.0))
plot(-1:0.01:1, μ -> NeXLCore.δσδΩ(NeXLCore.ScreenedRutherford, acos(μ), n"Cu", 1000.0))
plot!(0:0.01:2π, θ -> δσδΩ(Val(:NiklasScreenedRutherford), θ, n"Cu", 1000.0))
dimless(NeXLCore.σₜ(NeXLCore.ScreenedRutherford, n"Fe", 1000.0)*u"cm"^2, eq.dim_basis)

using HCubature

A = zeros(length(5000:500:15000), 29)

for (i_n, n) in enumerate(0:28)
    for (i_ϵ, ϵ) in enumerate(5000.0:500.0:15000.0)
        A[i_ϵ, i_n] = dimless(hquadrature(μ -> NeXLCore.δσδΩ(NeXLCore.ScreenedRutherford, acos(μ), n"Fe", ϵ)*Pl.(μ, n), -1, 1)[1]*u"cm"^2, eq.dim_basis)
    end
end
using LinearAlgebra

S, V, D = svd(A)
E = S*Diagonal(V)*Diagonal(D[1, :])
K = inv(Diagonal(D[1, :])) * transpose(D)

E*K .- A

n_vals = 3


A_lr = S[:, 1:n_vals] * Diagonal(V[1:n_vals]) * transpose(D[:, 1:n_vals])
maximum(abs.(A_lr .- A))

plot(A[1, :])
plot!(A_lr[1, :])

plot(A[:, 1])
plot!(A_lr[:, 1])

plot(D)

S, V, D = tsvd(A, 5)
plot(S[:, 1])
plot!(S[:, 2])
plot!(S[:, 3])

dimless(hquadrature(μ -> NeXLCore.δσδΩ(NeXLCore.ScreenedRutherford, acos(μ), n"Fe", 1000.0)*Pl.(μ, 0:21), -1, 1)[1]*u"cm"^2, eq.dim_basis)



using EPMAfem.Gridap

PN_N = 21
mat = [n"Cu", n"Ni"]
ϵ_range = range(5.0u"keV", 15.0u"keV", length=35)
eq = NExt.epma_equations(mat, ϵ_range, PN_N, 6);

space_extensions_dimless = dimless.((-200u"nm", 0u"nm", -1000u"nm", 1000u"nm"), eq.dim_basis)
space_model = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel(space_extensions_dimless, (40, 120)))
energy_model = dimless(ϵ_range, eq.dim_basis)
direction_model = EPMAfem.SphericalHarmonicsModels.EEEOSphericalHarmonicsModel(PN_N, 2)


model = EPMAfem.DiscretePNModel(space_model, energy_model, direction_model)
pnproblem = EPMAfem.discretize_problem(eq, model, EPMAfem.cuda(), updatable=false)

discrete_system = EPMAfem.schurimplicitmidpointsystem(pnproblem)


excitation = EPMAfem.PNExcitation([(x=0.0, y=0.0)], [energy_model[end-10]], [VectorValue(-1.0, 0.0, 0.0)])
discrete_rhs = EPMAfem.discretize_rhs(excitation, model, EPMAfem.cuda())

probe = EPMAfem.PNProbe(model, EPMAfem.cuda(); Ω=Ω->1.0, ϵ=ϵ->1.0)

res = probe(discrete_system * discrete_rhs[1])

func = EPMAfem.SpaceModels.interpolable(res, space_model)

include("plot_overloads.jl")

heatmap(-0.4:0.01:0, -2:0.01:1, func, swapxy=true, aspect_ratio=:equal)

## 1D
include("plot_overloads.jl")

PN_N = 21
mat = [n"Cu"]
ϵ_range = range(5.0u"keV", 20.0u"keV", length=80)
eq = NExt.epma_equations(mat, ϵ_range, PN_N, 5);

space_extensions_dimless = dimless.((-700u"nm", 0u"nm"), eq.dim_basis)
space_model = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel(space_extensions_dimless, (100)))
energy_model = dimless(ϵ_range, eq.dim_basis)
direction_model = EPMAfem.SphericalHarmonicsModels.EEEOSphericalHarmonicsModel(PN_N, 1)

model = EPMAfem.DiscretePNModel(space_model, energy_model, direction_model)
pnproblem = EPMAfem.discretize_problem(eq, model, EPMAfem.cuda(), updatable=false)

discrete_system = EPMAfem.schurimplicitmidpointsystem(pnproblem)

excitation = EPMAfem.PNExcitation([(x=0.0, y=0.0)], [dimless(15.0u"keV", eq.dim_basis)], [VectorValue(-1.0, 0.0, 0.0)])
discrete_rhs = EPMAfem.discretize_rhs(excitation, model, EPMAfem.cuda())

probe = EPMAfem.PNProbe(model, EPMAfem.cuda(); Ω=Ω->1.0, ϵ=ϵ -> dimless(ionizationcrosssection(n"Cu K", dimful(ϵ, u"eV", eq.dim_basis) / u"eV" |> upreferred)u"cm"^2, eq.dim_basis))

@time res = probe(discrete_system * discrete_rhs[1])

func = EPMAfem.SpaceModels.interpolable(res, space_model)
z = range(-700.0u"nm", 0.0u"nm", length=100)
z_dimless = dimless(z, eq.dim_basis)

# plot(8000.0:10.0:15000.0, ϵ -> dimless(ionizationcrosssection(n"Cu K", ϵ)u"cm"^2, eq.dim_basis))

# @gif for i in reverse(1:length(ϵ_range))
#     func = EPMAfem.SpaceModels.interpolable(res.p[:, i], space_model)
#     plot(-1:0.01:0, func)
#     ylims!(-0.1, 0.9)
# end

alg = NeXLMatrixCorrection.XPP(Material("", Dict(n"Cu"=>1.0)), n"Cu K", dimful(excitation.beam_energies[1], u"eV", eq.dim_basis) / u"eV" |> upreferred)
function phi_rho_z(alg, z)
    ρz = n"Cu".density * z * (u"cm"^2 / u"g") |> upreferred
    @show ρz
    ϕ(alg, -ρz)
end

plot(z, func.(Point.(z_dimless))/2.047990649409106e-14)
plot!(z, x -> phi_rho_z(alg, x))
