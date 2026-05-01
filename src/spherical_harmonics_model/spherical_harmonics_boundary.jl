# function get_cached_boundary_(D, m1, m2, tol)
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

const boundary_matrix_spherical_harmonics_dict_filename = pkgdir(@__MODULE__, "src/spherical_harmonics_model/boundary_matrix_spherical_harmonics_dict.jls")
const boundary_matrix_spherical_harmonics_dict = isfile(boundary_matrix_spherical_harmonics_dict_filename) ? Serialization.deserialize(boundary_matrix_spherical_harmonics_dict_filename) : Dict{Tuple{Int64, Tuple{Int64, Int64}, Tuple{Int64, Int64}}, Float64}()

const boundary_matrix_circular_harmonics_dict_filename = pkgdir(@__MODULE__, "src/spherical_harmonics_model/boundary_matrix_circular_harmonics_dict.jls")
const boundary_matrix_circular_harmonics_dict = isfile(boundary_matrix_circular_harmonics_dict_filename) ? Serialization.deserialize(boundary_matrix_circular_harmonics_dict_filename) : Dict{Tuple{Int64, Tuple{Int64, Int64}, Tuple{Int64, Int64}}, Float64}()

# note: here we use dim differently, X: 1, Y: 2, Z: 3
dim_to_num(::X) = 1
dim_to_num(::Y) = 2
dim_to_num(::Z) = 3

function get_cached_boundary_coefficient(m1::SphericalHarmonic, m2::SphericalHarmonic, dim)
    if !has_same_eee(m1, m2) return 0.0 end
    l, k = degreeorder(m1)
    l_, k_= degreeorder(m2)

    D = dim_to_num(dim)
    @assert D ∈ (1, 2, 3)

    if haskey(boundary_matrix_spherical_harmonics_dict, (D, (l, k), (l_, k_)))
        return boundary_matrix_spherical_harmonics_dict[(D, (l, k), (l_, k_))]
    elseif haskey(boundary_matrix_spherical_harmonics_dict, (D, (l_, k_), (l, k)))
        # matrix is symmetric
        return boundary_matrix_spherical_harmonics_dict[(D, (l_, k_), (l, k))]
    else
        throw(DomainError("Boundary matrix value $((D, (l, k), (l_, k_))) not precomputed!"))
    end
end

function get_cached_boundary_coefficient(m1::CircularHarmonic, m2::CircularHarmonic, dim)
    l, k = degreeorder(m1)
    l_, k_ = degreeorder(m2)

    D = dim_to_num(dim)
    @assert D == 1 || D == 3
    if haskey(boundary_matrix_circular_harmonics_dict, (D, (l, k), (l_, k_)))
        return boundary_matrix_circular_harmonics_dict[(D, (l, k), (l_, k_))]
    elseif haskey(boundary_matrix_circular_harmonics_dict, (D, (l_, k_), (l, k)))
        # matrix is symmetric
        return boundary_matrix_circular_harmonics_dict[(D, (l_, k_), (l, k))]
    else
        throw(DomainError("Boundary matrix value $((D, (l, k), (l_, k_))) not precomputed!"))
    end
end

function serialize_boundary_dicts()
    Serialization.serialize(boundary_matrix_circular_harmonics_dict_filename, boundary_matrix_circular_harmonics_dict)
    Serialization.serialize(boundary_matrix_spherical_harmonics_dict_filename, boundary_matrix_spherical_harmonics_dict)
end
