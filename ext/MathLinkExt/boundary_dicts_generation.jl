function populate_boundary_dict_circular_harmonics(D, Nmax; test=false)
    @assert D == 1 || D == 3 # D=1 -> X, D=3 -> Z
    b_dict = EPMAfem.SphericalHarmonicsModels.boundary_matrix_circular_harmonics_dict
    moms = circular_harmonics(Nmax)

    dim = D == 1 ? EPMAfem.Dimensions.X() : EPMAfem.Dimensions.Z()

    for (i, m1) in enumerate(moms)
        for m2 in moms[i:end]
            @show m1, m2
            l, k = EPMAfem.SphericalHarmonicsModels.degreeorder(m1)
            l_, k_ = EPMAfem.SphericalHarmonicsModels.degreeorder(m2)
            if haskey(b_dict, (D, (l, k), (l_, k_))) || haskey(b_dict, (D, (l_, k_), (l, k)))
                if test
                    b_val = haskey(b_dict, (D, (l, k), (l_, k_))) ? b_dict[(D, (l, k), (l_, k_))] : b_dict[(D, (l_, k_), (l, k))]
                    @assert b_val ≈ w_num(get_boundary_coefficient_symbolic(m1, m2, dim))::Float64
                end
            else
                b_val = get_boundary_coefficient_symbolic(m1, m2, dim)
                b_dict[(D, (l, k), (l_, k_))] = w_num(b_val)::Float64
            end
        end
    end
end

function populate_boundary_dict_spherical_harmonics(D, Nmax; test=false, dim=EPMAfem.Dimensions._3D())
    @assert D == 1 || D == 2 || D == 3 # D=1 -> X, D=2 -> Y, D=3 -> Z
    b_dict = EPMAfem.SphericalHarmonicsModels.boundary_matrix_spherical_harmonics_dict
    moms = spherical_harmonics(Nmax, dim)

    dim = (EPMAfem.Dimensions.X(), EPMAfem.Dimensions.Y(), EPMAfem.Dimensions.Z())[D]

    for (i, m1) in enumerate(moms)
        for m2 in moms[i:end]
            @show m1, m2
            l, k = EPMAfem.SphericalHarmonicsModels.degreeorder(m1)
            l_, k_ = EPMAfem.SphericalHarmonicsModels.degreeorder(m2)
            if haskey(b_dict, (D, (l, k), (l_, k_))) || haskey(b_dict, (D, (l_, k_), (l, k)))
                if test
                    b_val = haskey(b_dict, (D, (l, k), (l_, k_))) ? b_dict[(D, (l, k), (l_, k_))] : b_dict[(D, (l_, k_), (l, k))]
                    @assert b_val ≈ w_num(get_boundary_coefficient_symbolic(m1, m2, dim))::Float64
                    @info "Test Successful!"
                end
            else
                b_val = w_num(get_boundary_coefficient_symbolic(m1, m2, dim))::Float64
                b_dict[(D, (l, k), (l_, k_))] = w_num(b_val)::Float64
                @info "Stored value $(b_val)!"
            end
        end
    end
end
