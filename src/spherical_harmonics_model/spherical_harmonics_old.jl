# module SphericalHarmonicsModel

# # using SparseArrays
# # using SphericalHarmonics
# # using HCubature
# # using LinearAlgebra
# # using LegendrePolynomials
# # using Serialization
# # using Logging
# # # using MathLink

# function get_moments(N, ::Val{1})
#     # only allocate moments for z
#     moments = [(l, k) for l in 0:N for k in -l:l if (!is_odd_in(l, k, Val{1}()) && !is_odd_in(l, k, Val{2}()))]
#     return moments
# end

# function get_moments(N, ::Val{2})
#     # only allocate moments for x and z
#     moments = [(l, k) for l in 0:N for k in -l:l if !is_odd_in(l, k, Val{2}())]
#     return moments
# end

# function get_moments(N, ::Val{3})
#     # allocate moments for x, y, and z
#     moments = [(l, k) for l in 0:N for k in -l:l]
#     return moments
# end

# function get_even_moments2(N, nd)
#     eee_moments = [m for m in get_moments(N, nd) if is_even_in(m..., Val(1)) && is_even_in(m..., Val(2)) && is_even_in(m..., Val(3))]
#     eoo_moments = [m for m in get_moments(N, nd) if is_even_in(m..., Val(1)) && is_odd_in(m..., Val(2)) && is_odd_in(m..., Val(3))]
#     oeo_moments = [m for m in get_moments(N, nd) if is_odd_in(m..., Val(1)) && is_even_in(m..., Val(2)) && is_odd_in(m..., Val(3))]
#     ooe_moments = [m for m in get_moments(N, nd) if is_odd_in(m..., Val(1)) && is_odd_in(m..., Val(2)) && is_even_in(m..., Val(3))]
#     return reduce(vcat, [eee_moments, eoo_moments, oeo_moments, ooe_moments])
# end

# function get_odd_moments2(N, nd)
#     oee_moments = [m for m in get_moments(N, nd) if is_odd_in(m..., Val(1)) && is_even_in(m..., Val(2)) && is_even_in(m..., Val(3))]
#     eoe_moments = [m for m in get_moments(N, nd) if is_even_in(m..., Val(1)) && is_odd_in(m..., Val(2)) && is_even_in(m..., Val(3))]
#     eeo_moments = [m for m in get_moments(N, nd) if is_even_in(m..., Val(1)) && is_even_in(m..., Val(2)) && is_odd_in(m..., Val(3))]
#     ooo_moments = [m for m in get_moments(N, nd) if is_odd_in(m..., Val(1)) && is_odd_in(m..., Val(2)) && is_odd_in(m..., Val(3))]
#     return reduce(vcat, [oee_moments, eoe_moments, eeo_moments, ooo_moments])
# end

# function get_even_moments(N, nd)
#     even_moments = [m for m in get_moments(N, nd) if is_even(m...)]
#     return even_moments
# end

# function get_odd_moments(N, nd)
#     odd_moments = [m for m in get_moments(N, nd) if !is_even(m...)]
#     return odd_moments
# end

# function get_eo_moments(N, nd)
#     return vcat(get_even_moments(N, nd), get_odd_moments(N, nd))
# end

# function assemble_transport_matrix(N, dim, parity, nd)
#     if parity == :pm
#         eo_moms1 = get_even_moments(N, nd)
#         eo_moms2 = get_odd_moments(N, nd)
#     else
#         eo_moms1 = get_odd_moments(N, nd)
#         eo_moms2 = get_even_moments(N, nd)
#     end
#     # eo_moms = get_eo_moments(N, nd)
#     A = spzeros(length(eo_moms2), length(eo_moms1))
#     for (i, (l, k)) = enumerate(eo_moms1)
#         for (j, (l_, k_)) = enumerate(eo_moms2)
#             c = get_coefficient(l, k, l_, k_, dim)
#             if c != 0
#                 A[j, i] = get_coefficient(l, k, l_, k_, dim)
#             end
#         end
#     end
#     return A
# end

# # const realsphericalharmonics = W`RealSphericalHarmonicY[l_, m_, \[Theta]_, \[Phi]_] := FullSimplify[If[m < 0, 
# #     (-1)^m*I/Sqrt[2]*(SphericalHarmonicY[l, m, \[Theta], \[Phi]] - (-1)^m*SphericalHarmonicY[l, -m, \[Theta], \[Phi]]),
# #     If[m > 0,
# #     (-1)^m*1/Sqrt[2]*(SphericalHarmonicY[l, -m, \[Theta], \[Phi]] + (-1)^m*SphericalHarmonicY[l, m, \[Theta], \[Phi]]),
# #     SphericalHarmonicY[l, m, \[Theta], \[Phi]]]]]`
# # weval(realsphericalharmonics)

# # const xintegral = W`xintegral[l1_, m1_, l2_, m2_] := Integrate[
# #             Abs[Sin[\[Theta]]*Cos[\[Phi]]]*RealSphericalHarmonicY[l1, m1, \[Theta], \[Phi]]*
# #             RealSphericalHarmonicY[l2, m2, \[Theta], \[Phi]]*
# #             Sin[\[Theta]], {\[Theta], 0, \[Pi]}, {\[Phi], 0, 2  \[Pi]}]`

# # const yintegral = W`yintegral[l1_, m1_, l2_, m2_] := Integrate[
# #                 Abs[Sin[\[Theta]]*Sin[\[Phi]]]*RealSphericalHarmonicY[l1, m1, \[Theta], \[Phi]]*
# #                 RealSphericalHarmonicY[l2, m2, \[Theta], \[Phi]]*
# #                 Sin[\[Theta]], {\[Theta], 0, \[Pi]}, {\[Phi], 0, 2  \[Pi]}]`

# # const zintegral = W`zintegral[l1_, m1_, l2_, m2_] := Integrate[
# #                 Abs[Cos[\[Theta]]]*RealSphericalHarmonicY[l1, m1, \[Theta], \[Phi]]*
# #                 RealSphericalHarmonicY[l2, m2, \[Theta], \[Phi]]*
# #                 Sin[\[Theta]], {\[Theta], 0, \[Pi]}, {\[Phi], 0, 2  \[Pi]}]`

# # weval(xintegral)
# # weval(yintegral)
# # weval(zintegral)

# # # const comp_xintegral = W`cxintegral = Compile[{{l1, _Integer}, {m1, _Integer}, {l2, _Integer}, {m2, _Integer}}, N[xintegral[l1, m1, l2, m2]]]`
# # # const comp_yintegral = W`cyintegral = Compile[{{l1, _Integer}, {m1, _Integer}, {l2, _Integer}, {m2, _Integer}}, N[yintegral[l1, m1, l2, m2]]]`
# # # const comp_zintegral = W`czintegral = Compile[{{l1, _Integer}, {m1, _Integer}, {l2, _Integer}, {m2, _Integer}}, N[zintegral[l1, m1, l2, m2]]]`

# # # weval(comp_xintegral)
# # # weval(comp_yintegral)
# # # weval(comp_zintegral)

# # const call_xintegral = W`N[xintegral[l1, m1, l2, m2]]` 
# # const call_yintegral = W`N[yintegral[l1, m1, l2, m2]]`  
# # const call_zintegral = W`N[zintegral[l1, m1, l2, m2]]` 



# # const hist_length = "\$HistoryLength"
# # weval(W`$hist_length = 0`) 

# function compute_boundary_matrix_entry_mathematica(D, m1, m2)
#     boundary_integral = (
#         call_xintegral,
#         call_yintegral,
#         call_zintegral)[D]

#     val = weval(boundary_integral; l1=m1[1], m1=m1[2], l2=m2[1], m2=m2[2])
#     return val
# end 

# function has_same_eee(m1, m2)
#     return ((is_even_in(m1..., Val(1)) && is_even_in(m2..., Val(1))) || (is_odd_in(m1..., Val(1)) && is_odd_in(m2..., Val(1)))) &&
#     ((is_even_in(m1..., Val(2)) && is_even_in(m2..., Val(2))) || (is_odd_in(m1..., Val(2)) && is_odd_in(m2..., Val(2)))) &&
#     ((is_even_in(m1..., Val(3)) && is_even_in(m2..., Val(3))) || (is_odd_in(m1..., Val(3)) && is_odd_in(m2..., Val(3))))
# end

# function compute_boundary_matrix_entry(D, m1, m2, tol)
#     # check if the eee is the same (I dont have a proof for that, feel free to check!)
#     if !has_same_eee(m1, m2) return 0.0, 0.0 end
#     if tol == 0 return compute_boundary_matrix_entry_mathematica(D, m1, m2), 0.0 end
#     function integrand((θ, ϕ))
#         abs_Ω = abs((sin(θ)*cos(ϕ), sin(θ)*sin(ϕ), cos(θ))[D])
#         # Y = computeYlm(θ, ϕ, lmax=max(m1[1], m2[1]), SHType=SphericalHarmonics.RealHarmonics())
#         Y1 = SphericalHarmonics.sphericalharmonic(θ, ϕ, m1[1], m1[2], SphericalHarmonics.RealHarmonics())
#         Y2 = SphericalHarmonics.sphericalharmonic(θ, ϕ, m2[1], m2[2], SphericalHarmonics.RealHarmonics())
#         return abs_Ω * (Y1 * Y2)
#     end
#     return hcubature(x -> integrand(x)*sin(x[1]), (0, 0), (π, 2π), rtol=tol, atol=tol)
# end

# function assemble_boundary_matrix_old(N, ::Val{D}, parity, nd::Val{ND}) where {D, ND}
#     function integrand((θ, ϕ))
#         abs_Ω = abs((sin(θ)*cos(ϕ), sin(θ)*sin(ϕ), cos(θ))[D])
#         Y = computeYlm(θ, ϕ, lmax=N, SHType=SphericalHarmonics.RealHarmonics())
#         if parity == :pp
#             Y_e = [Y[m] for m ∈ get_eo_moments(N, nd) if is_even(m...)]
#         else
#             Y_e = [0]
#         end
#         return abs_Ω * ((Y_e' .* Y_e))
#     end
#     A = hcubature(x -> integrand(x)*sin(x[1]), (0, 0), (π, 2π), rtol=1e-8, atol=1e-8, maxevals=100000)[1]
#     return round.(sparse(A), digits=8)
# end



# # function assemble_direction_source(N, qΩ, nd::Val{ND}) where ND
# #     eo_moms = get_eo_moments(N, nd)
# #     function integrand((θ, ϕ))
# #         Ω = (sin(θ)*cos(ϕ), sin(θ)*sin(ϕ), cos(θ))
# #         Y = computeYlm(θ, ϕ, lmax=N, SHType=SphericalHarmonics.RealHarmonics())
# #         Y_eo = [Y[m] for m ∈ eo_moms]
# #         return qΩ(Ω)*Y_eo
# #     end
# #     b = hcubature(x -> integrand(x)*sin(x[1]), (0, 0), (π, 2π), rtol=1e-8, atol=1e-8, maxevals=100000)[1]
# #     return (p=[b[i] for (i, m) ∈ enumerate(eo_moms) if is_even(m...)], m=[b[i] for (i, m) ∈ enumerate(eo_moms) if is_odd(m...)])
# # end

# function assemble_gram_matrix(N, nd::Val{ND}) where ND
#     function integrand((θ, ϕ))
#         # Ω = (sin(θ)*cos(ϕ), sin(θ)*sin(ϕ), cos(θ))
#         Y = computeYlm(θ, ϕ, lmax=N, SHType=SphericalHarmonics.RealHarmonics())
#         Y_eo = [Y[m] for m ∈ get_eo_moments(N, nd)]
#         return Y_eo*Y_eo'
#     end
#     return hcubature(x -> integrand(x)*sin(x[1]), (0, 0), (π, 2π))[1]
# end

# function cut_pos(x)
#     return x <= 0.0 ? x : 0.0
# end

# function compute_direction_boundary_entry(n, g_Ω, m)
#     function integrand((θ, ϕ))
#         Ω = (sin(θ)*cos(ϕ), sin(θ)*sin(ϕ), cos(θ))
#         Y = SphericalHarmonics.sphericalharmonic(θ, ϕ, m[1], m[2], SphericalHarmonics.RealHarmonics())
#         return cut_pos(dot(Ω, n))*g_Ω(Ω)*Y
#     end
#     return hcubature(x -> integrand(x)*sin(x[1]), (0, 0), (π, 2π), rtol=1e-7, atol=1e-7)
# end

# function assemble_direction_boundary_old(N, g_Ω, n, nd::Val{ND}) where {ND}
#     eo_moms = get_eo_moments(N, nd)
#     function integrand((θ, ϕ))
#         Ω = (sin(θ)*cos(ϕ), sin(θ)*sin(ϕ), cos(θ))
#         Y = computeYlm(θ, ϕ, lmax=N, SHType=SphericalHarmonics.RealHarmonics())
#         Y_eo_even = [is_even(m...) ? Y[m] : 0.0 for m ∈ eo_moms]
#         return cut_pos(dot(Ω, n))*g_Ω(Ω)*Y_eo_even
#     end
#     b = hcubature(x -> integrand(x)*sin(x[1]), (0, 0), (π, 2π), rtol=1e-8, atol=1e-8, maxevals=100000)[1]
#     # return [b[i] for (i, m) ∈ enumerate(eo_moms) if is_even(m...)], [b[i] for (i, m) ∈ enumerate(eo_moms) if is_odd(m...)]
#     return (p=[b[i] for (i, m) ∈ enumerate(eo_moms) if is_even(m...)], m=spzeros(length([m for m in eo_moms if is_odd(m...)])))
# end

# export assemble_transport_matrix, assemble_scattering_matrices
# end

# function assemble_boundary_matrix(N, ::Val{D}, parity, nd::Val{ND}, tol=0.0) where {D, ND}
#     filename = "boundary_matrix_dict.jls"
#     boundary_matrix_dict = isfile(filename) ? Serialization.deserialize(filename) : Dict{Tuple{Int64, Tuple{Int64, Int64}, Tuple{Int64, Int64}}, Tuple{Float64, Float64}}()
#     @assert parity == :pp
#     # even_moments = [m for m ∈ SphericalHarmonicsMatrices.get_eo_moments(N, nd) if SphericalHarmonicsMatrices.is_even(m...)]
#     even_moments = SphericalHarmonicsMatrices.get_even_moments(N, nd)
#     A = zeros(length(even_moments), length(even_moments))
#     for (i, (l, k)) = enumerate(even_moments)
#         for (j, (l_, k_)) = enumerate(even_moments)
#             if haskey(boundary_matrix_dict, (D, (l, k), (l_, k_))) && boundary_matrix_dict[(D, (l, k), (l_, k_))][2] <= tol
#                 A[i, j] = boundary_matrix_dict[(D, (l, k), (l_, k_))][1]
#             elseif haskey(boundary_matrix_dict, (D, (l_, k_), (l, k))) && boundary_matrix_dict[(D, (l_, k_), (l, k))][2] <= tol
#                 A[i, j] = boundary_matrix_dict[(D, (l_, k_), (l, k))][1]
#             else
#                 a, error = SphericalHarmonicsMatrices.compute_boundary_matrix_entry(D, (l, k), (l_, k_), tol)
#                 if haskey(boundary_matrix_dict, (D, (l, k), (l_, k_)))
#                     a_old, tol_old = boundary_matrix_dict[(D, (l, k), (l_, k_))]
#                     @info "replacing the old value ∫_S^2 |Ω_$(D)| Y_$(l)^$k Y_$(l_)^$(k_) dΩ = $(a_old) with tolerance $(tol_old)."
#                     @info "replacing with new vale ∫_S^2 |Ω_$(D)| Y_$(l)^$k Y_$(l_)^$(k_) dΩ = $(a) with tolerance $(tol)."
#                 else
#                     @info "computed and stored ∫_S^2 |Ω_$(D)| Y_$(l)^$k Y_$(l_)^$(k_) dΩ = $(a) with tolerance $(tol) and with error of $(error)."
#                 end
#                 boundary_matrix_dict[(D, (l, k), (l_, k_))] = (a, tol)
#                 Serialization.serialize(filename, boundary_matrix_dict)
#                 A[i, j] = boundary_matrix_dict[(D, (l, k), (l_, k_))][1]
#             end
            
#         end
#     end
#     return A
# end

# function assemble_direction_boundary(N, g_Ω, n, nd::Val{ND}) where {ND}
#     even_moments = [m for m ∈ SphericalHarmonicsMatrices.get_eo_moments(N, nd) if SphericalHarmonicsMatrices.is_even(m...)]
#     odd_moments = [m for m ∈ SphericalHarmonicsMatrices.get_eo_moments(N, nd) if SphericalHarmonicsMatrices.is_odd(m...)]
#     bp = zeros(length(even_moments))
#     for (i, (l, k)) = enumerate(even_moments)
#         a, error = SphericalHarmonicsMatrices.compute_direction_boundary_entry(n, g_Ω, (l, k))
#         bp[i] = a
#     end
#     return (p=bp, m=spzeros(length(odd_moments)))
# end

# function assemble_direction_boundary(N, g_Ω::VMFBeam, n, nd::Val{ND}) where {ND}
#     even_moments = [m for m ∈ SphericalHarmonicsMatrices.get_eo_moments(N, nd) if SphericalHarmonicsMatrices.is_even(m...)]
#     odd_moments = [m for m ∈ SphericalHarmonicsMat    θ, ϕ = unitsphere_cartesian_to_spherical(VectorValue(Ω[1], Ω[2], Ω[3]))
#     rices.get_eo_moments(N, nd) if SphericalHarmonicsMatrices.is_odd(m...)]
#     filename = "direction_boundary_dict.jls"
#     direction_boundary_dict = isfile(filename) ? Serialization.deserialize(filename) : Dict{Tuple{BeamDirection, SVector{3, Float64}, Tuple{Int64, Int64}}, Float64}()
#     bp = zeros(length(even_moments))
#     for (i, (l, k)) = enumerate(even_moments)
#         if !haskey(direction_boundary_dict, (g_Ω, n, (l, k)))
#             a, error = SphericalHarmonicsMatrices.compute_direction_boundary_entry(n, g_Ω, (l, k))
#             @info "computed and stored $(a) with error of $(error)."
#             direction_boundary_dict[(g_Ω, n, (l, k))] = a
#             Serialization.serialize(filename, direction_boundary_dict)
#         end
#         bp[i] = direction_boundary_dict[(g_Ω, n, (l, k))]
#     end
#     return (p=bp, m=spzeros(length(odd_moments)))
# end

# # there is a legacy version of this in SphericalHarmonicsMatrices
# function assemble_direction_source(N, ::IsotropicExtraction, nd::Val{ND}) where ND
#     even_moments = [m for m ∈ SphericalHarmonicsMatrices.get_eo_moments(N, nd) if SphericalHarmonicsMatrices.is_even(m...)]
#     odd_moments = [m for m ∈ SphericalHarmonicsMatrices.get_eo_moments(N, nd) if SphericalHarmonicsMatrices.is_odd(m...)]
#     bp = zeros(length(even_moments))
#     bp[1] = 1.0
#     return (p=bp, m=zeros(length(odd_moments)))
# end


# #### FOR TESTING

# # A = assemble_transport_matrix(3, Val{3}())
# # K = assemble_scattering_matrix(3, scattering)

# # N = 3

# # function integrand((θ, ϕ), ::Val{D}) where D
# #     Y = computeYlm(θ, ϕ, lmax=N, SHType=SphericalHarmonics.RealHarmonics())
# #     Y_eo = [Y[m] for m ∈ get_eo_moments(N)]
# #     Ω = (sin(θ)*cos(ϕ), sin(θ)*sin(ϕ), cos(θ))
# #     return Ω[D]*(Y_eo' .* Y_eo)
# # end

# # function scattering(μ) 
# #     # @show μ
# #     @assert -1.0001 <= μ <= 1.0001
# #     σ = 0.1
# #     return exp(-0.5*μ^2/(σ*σ))/(sqrt(2π)σ)
# # end



# # using Plots
# # plot(-1:0.01:1, scattering)

# # function integrand_2((θ, ϕ, θ_, ϕ_))
# #     Y = computeYlm(θ, ϕ, lmax=N, SHType=SphericalHarmonics.RealHarmonics())
# #     Y_ = computeYlm(θ_, ϕ_, lmax=N, SHType=SphericalHarmonics.RealHarmonics())
# #     Y_eo = [Y[m] for m ∈ get_eo_moments(N)]
# #     Y_eo_ = [Y_[m] for m ∈ get_eo_moments(N)]
# #     Ω = (sin(θ)*cos(ϕ), sin(θ)*sin(ϕ), cos(θ))
# #     Ω_ = (sin(θ_)*cos(ϕ_), sin(θ_)*sin(ϕ_), cos(θ_))
# #     return scattering(dot(Ω, Ω_))*(Y_eo .* Y_eo_)
# # end


# # A_x = hcubature(x -> integrand(x, Val{1}())*sin(x[1]), (0, 0), (π, 2π))[1]
# # A_x2 = assemble_transport_matrix(N, Val{1}())

# # round.(A_x, digits=4)
# # round.(Matrix(A_x2), digits=4)

# # abs.(A_x .- A_x2) |> maximum

# # K = hcubature(x -> integrand_2(x)*sin(x[1])*sin(x[3]), (0, 0, 0, 0), (π, 2π, π, 2π), rtol=1e-4, atol=1e-4, maxevals=1000000)[1]
# # K2 = assemble_scattering_matrix(N, scattering)
