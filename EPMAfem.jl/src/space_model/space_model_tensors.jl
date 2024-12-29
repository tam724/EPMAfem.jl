function assemble_trilinear(∫, model, U, V) # W is implicitly given by a 0-th order space
    v = get_fe_basis(V)
    u = get_trial_fe_basis(TrialFESpace(U))
    args = get_args(model)
    domain_contrib = ∫(u, v, args)
    Dc = num_dims(get_background_model(get_triangulation(U)))
    Is = Int64[]
    Js = Int64[]
    Ks = Int64[]
    Vs = Float64[]

    for Ω_temp in Gridap.CellData.get_domains(domain_contrib)
        scell_mat = Gridap.FESpaces.get_contribution(domain_contrib, Ω_temp)
        cell_mats, Ω = Gridap.move_contributions(scell_mat, Ω_temp)
        cell_mats = Gridap.FESpaces.attach_constraints_cols(U,cell_mats,Ω)
        cell_mats = Gridap.FESpaces.attach_constraints_rows(V,cell_mats,Ω)

        # cell_mats = domain_contrib[Ω]
        glue = Gridap.get_glue(Ω, Val(Dc))
        U_ids = Gridap.get_cell_dof_ids(U, Ω)
        V_ids = Gridap.get_cell_dof_ids(V, Ω)

        cell_mat_cache = Gridap.array_cache(cell_mats)
        U_cache = Gridap.array_cache(U_ids)
        V_cache = Gridap.array_cache(V_ids)

        for cell_idx in eachindex(U_ids, V_ids)
            U_id = getindex!(U_cache, U_ids, cell_idx)::Vector
            V_id = getindex!(V_cache, V_ids, cell_idx)::Vector

            cell_mat = getindex!(cell_mat_cache, cell_mats, cell_idx)

            j_p = Int64(glue.tface_to_mface[cell_idx])

            for j in eachindex(U_id)
                U_idj = Int64(U_id[j])

                for i in eachindex(V_id)
                    V_idi = Int64(V_id[i])
                    val = cell_mat[i, j]::Float64

                    push!(Is, V_idi)
                    push!(Js, U_idj)
                    push!(Ks, j_p)
                    push!(Vs, val)
                end
            end
        end
    end
    trian = args[2]
    return Sparse3Tensor.sparse3tensor(Is, Js, Ks, Vs, (num_free_dofs(V), num_free_dofs(U), num_cells(trian)))
end