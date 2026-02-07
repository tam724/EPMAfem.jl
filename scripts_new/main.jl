using Revise
using EPMAfem
using EPMAfem.PNLazyMatrices
using EPMAfem.BlockDiagonals
using EPMAfem.CUDA
using EPMAfem.SparseArrays
using LinearAlgebra

using EPMAfem.SphericalHarmonicsModels
SH = EPMAfem.SphericalHarmonicsModels

model = SH.EOSphericalHarmonicsModel(10, 3)

SH.plus(model)

A = SH.assemble_bilinear(SH.∫S²_μuv(Ω -> Ω[1]*Ω[3]), model, SH.minus(model), SH.minus(model), SH.lebedev_quadrature_max())
heatmap(A)
eigen(A).values


using EPMAfem.Gridap

model = CartesianDiscreteModel((0, 1), 10)
V = TestFESpace(model, ReferenceFE(raviart_thomas, Float64, 0), conformity=:Hdiv)

Ω = Triangulation(model)
dx = Measure(Ω, 5)

a(u,v) = ∫(dot(u, v))dx
A = assemble_matrix(a, TrialFESpace(V), V)
