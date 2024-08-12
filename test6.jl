using Revise

using HCubature
using StaticArrays
using Unitful
using Gridap
using LinearAlgebra
using Plots

include("epmaequations.jl")
include("pnequations.jl")

epma_eq = dummy_epma_equations(
    [0.85u"keV"],                                               # beam energy
    range(-0.5u"nm", 0.5u"nm", length=10),                      # beam position
    [[-0.5, 0.0, -0.5], [0.0, 0.0, -1.0], [0.5, 0.0, -0.5]],    # beam direction. Careful, this is (x, y, z), not (z, x, y)
    300.0,                                                      # beam concentration
    [0.1u"keV", 0.2u"keV"])                                     # extraction energy 
pn_eq = PNEquations(epma_eq)

specific_stopping_power(epma_eq, 1)(1u"keV")
specific_electron_scattering_cross_section(epma_eq, 1, 1)(1u"keV")

stopping_power(epma_eq)(1u"keV", [1u"nm", 0u"nm", 0u"nm"]) |> upreferred

beam_energy_distribution(epma_eq, 1)(0.85u"keV")

beam_spatial_distribution(epma_eq, 1)((0.0u"nm", -0.499u"nm", 0.0u"nm"))
x = range(-1u"nm", 1u"nm", 100)

heatmap(x, x, (x, y) -> beam_spatial_distribution(epma_eq, 1)((0.0u"nm", x, y)))


_s(pn_eq, 1)(1.0)

@time _τ(pn_eq, 1)(1.0)

pn_sys = build_solver(model, 23, 2)

n_e = number_of_basis_functions(pn_sys).Ω.p
n_o = number_of_basis_functions(pn_sys).Ω.m

a_z = assemble_transport_matrix(pn_sys.PN, Val(3), :pm, Val(3))
A_z = Matrix([zeros(n_e, n_e) transpose(a_z)
a_z zeros(n_o, n_o)])


a_x = assemble_transport_matrix(pn_sys.PN, Val(1), :pm, Val(3))
A_x = Matrix([zeros(n_e, n_e) transpose(a_x)
a_x zeros(n_o, n_o)])

a_y = assemble_transport_matrix(pn_sys.PN, Val(2), :pm, Val(3))
A_y = Matrix([zeros(n_e, n_e) transpose(a_y)
a_y zeros(n_o, n_o)])

eigen(A_z).vectors * eigen(A_x).vectors


a_z = assemble_transport_matrix(pn_sys.PN, Val(3), :pm, Val(3))
b_z = assemble_boundary_matrix(pn_sys.PN, Val(3), :pp, Val(3))

b_z = assemble_boundary_matrix(pn_sys.PN, Val(3), :pp, Val(3))
b_z = b_z |> cu
plot(reverse(sort(abs.(b_z.nzval))), yaxis=:log)

length(b_z.nzVal)/prod(size(b_z))

a_z_dense = Matrix(a_z) |> cu
b_z_dense = Matrix(b_z) |> cu

test = rand(100000, 378) |> cu
res = zeros(100000, 378) |> cu

@benchmark mul!(res, test, b_z)
@benchmark mul!(res, test, b_z_dense)