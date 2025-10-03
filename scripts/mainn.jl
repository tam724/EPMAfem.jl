using Revise
using EPMAfem
using EPMAfem.PNLazyMatrices
using EPMAfem.Krylov
using BenchmarkTools
using LinearAlgebra
using EPMAfem
using SparseArrays

SH = EPMAfem.SphericalHarmonicsModels

basis = SH.EOSphericalHarmonicsModel(5, 3)

k = SH.∫∫S²_kuv((Ω₁, Ω₂) -> exp(Ω₁[1] * (Ω₁[2] + Ω₁[3])) + cos(Ω₂[3] + Ω₂[1] * Ω₂[2])) # some weird scattering kernel
A = SH.assemble_bilinear(k, basis, basis.moments, basis.moments)
round.(A; digits=8) |> sparse

k = SH.∫∫S²_kuv((Ω₁, Ω₂) -> exp(dot(Ω₁, Ω₂)))
A = SH.assemble_bilinear(k, basis, basis.moments, basis.moments)
round.(A; digits=8) |> sparse

# test against other impl
k_iso = SH.∫S²_kuv(μ -> exp(μ))
A_iso = SH.assemble_bilinear(k_iso, basis, basis.moments, basis.moments, SH.hcubature_quadrature(1e-5, 1e-5))
round.(A_iso; digits=8) |> sparse

k = SH.∫∫S²_kuv((Ω₁, Ω₂) -> 1.0) 
A = SH.assemble_bilinear(k, basis, basis.moments, basis.moments)
round.(A; digits=8) |> sparse

