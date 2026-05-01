function populate_boundary_dict_circular_harmonics(D, Nmax)
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
                # test value ? 
            else
                b_val = get_boundary_coefficient_symbolic(m1, m2, dim)
                b_dict[(D, (l, k), (l_, k_))] = w_num(b_val)::Float64
            end
        end
    end
end
