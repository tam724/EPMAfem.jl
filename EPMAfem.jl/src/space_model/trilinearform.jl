# module GridapBiFormProjectors

# function find_nzval_index2(A, i, j)::Int64
#     for nzval_idx in nzrange(A, j)
#         if A.rowval[nzval_idx] == i
#             return nzval_idx
#         end
#     end
#     @error "index not found."
# end

function find_nzval_index(A, i, j)::Int64
    # idx = findfirst(v -> A.rowval[v] == i, nzrange(A, j))
    # @show idx
    idx = searchsorted(@view(A.rowval[nzrange(A, j)]), i)
    if length(idx) == 1
        return nzrange(A, j)[idx |> first]
    end
    @error "index not found"
end

function compute_projector_sparsity_structure_inner!(Is, Js, U_ids, V_ids)
    U_cache = array_cache(U_ids)
    V_cache = array_cache(V_ids)

    for cell_idx in eachindex(U_ids, V_ids)
        U_id = getindex!(U_cache, U_ids, cell_idx)::Vector
        V_id = getindex!(V_cache, V_ids, cell_idx)::Vector

        # cell_mat = cell_mats[cell_idx]
        for j in eachindex(U_id)
            U_idj = Int64(U_id[j])
            for i in eachindex(V_id)
                V_idi = Int64(V_id[i])
                #if !iszero(cell_mat[i, j]) 
                    push!(Is, V_idi)
                    push!(Js, U_idj)
                #end
            end
        end
    end
end

function compute_projector_sparsity_structure(domain_contrib, U, V)
    # compute sparsity structure of the sparse matrix
    Is = Int64[]
    Js = Int64[]

    for Ω_temp in Gridap.CellData.get_domains(domain_contrib)
        # cell_mats = domain_contrib[Ω]
        scell_mat = Gridap.FESpaces.get_contribution(domain_contrib, Ω_temp)
        _, Ω = Gridap.move_contributions(scell_mat, Ω_temp)
        # cell_mats_c = Gridap.FESpaces.attach_constraints_cols(U,cell_mats,Ω)
        # cell_mats_rc = Gridap.FESpaces.attach_constraints_rows(V,cell_mats_c,Ω)

        U_ids = Gridap.get_cell_dof_ids(U, Ω)
        V_ids = Gridap.get_cell_dof_ids(V, Ω)
        compute_projector_sparsity_structure_inner!(Is, Js, U_ids, V_ids)
    end

    return sparse(Is, Js, zeros(length(Is)))
end

function compute_projector_projector_inner!(I_p, J_p, V_p, U_ids, V_ids, cell_mats, glue, A_skeleton)

    cell_mat_cache = array_cache(cell_mats)
    U_cache = array_cache(U_ids)
    V_cache = array_cache(V_ids)

    for cell_idx in eachindex(U_ids, V_ids)
        U_id = getindex!(U_cache, U_ids, cell_idx)::Vector
        V_id = getindex!(V_cache, V_ids, cell_idx)::Vector

        cell_mat = getindex!(cell_mat_cache, cell_mats, cell_idx)
        j_p = glue.tface_to_mface[cell_idx]

        # @show cell_mat
        for j in eachindex(U_id)
            U_idj = Int64(U_id[j])
            for i in eachindex(V_id)
                V_idi = Int64(V_id[i])
                val = cell_mat[i, j]::Float64
                if !iszero(val)
                    i_p = find_nzval_index(A_skeleton, V_idi, U_idj)
                    # j_p = cell_idx
                    push!(I_p, i_p)
                    push!(J_p, j_p)
                    push!(V_p, val)
                end
            end
        end
    end
end

function compute_projector_projector(domain_contrib, U, V, A_skeleton)
    Dc = num_dims(get_background_model(get_triangulation(U)))

    I_p = Int64[]
    J_p = Int64[]
    V_p = Float64[]
    sizehint!(I_p, length(A_skeleton.nzval))
    sizehint!(J_p, length(A_skeleton.nzval))
    sizehint!(V_p, length(A_skeleton.nzval))
    for Ω_temp in Gridap.CellData.get_domains(domain_contrib)
        scell_mat = Gridap.FESpaces.get_contribution(domain_contrib, Ω_temp)
        cell_mats, Ω = Gridap.move_contributions(scell_mat, Ω_temp)
        cell_mats_c = Gridap.FESpaces.attach_constraints_cols(U,cell_mats,Ω)
        cell_mats_rc = Gridap.FESpaces.attach_constraints_rows(V,cell_mats_c,Ω)

        # cell_mats  = domain_contrib[Ω]
        glue = Gridap.get_glue(Ω, Val(Dc))
        U_ids = Gridap.get_cell_dof_ids(U, Ω)
        V_ids = Gridap.get_cell_dof_ids(V, Ω)
        compute_projector_projector_inner!(I_p, J_p, V_p, U_ids, V_ids, cell_mats_rc, glue, A_skeleton)
    end
    m = length(A_skeleton.nzval)
    n = num_cells(get_background_model(get_triangulation(U)))
    return sparse(I_p, J_p, V_p, m, n)
end


function build_projector(a, model, U, V)
    U_trial = TrialFESpace(U)
    u = get_trial_fe_basis(U_trial)
    v = get_fe_basis(V)

    args = get_args(model)

    domain_contrib = a(u, v, args)

    A_skeleton = compute_projector_sparsity_structure(domain_contrib, U_trial, V)

    projector = compute_projector_projector(domain_contrib, U_trial, V, A_skeleton)

    # detect piecewise constant basis functions
    if length(get_cell_dof_ids(U) |> first) == 1
        A_skeleton = isdiag(A_skeleton) ? Diagonal(Vector(diag(A_skeleton))) : A_skeleton
        projector = isdiag(projector) ? Diagonal(Vector(diag(projector))) : projector
    end
    return A_skeleton, projector
end

SparseArrays.nonzeros(A::Diagonal) = A.diag

function project_matrices(ρs, ρ_projector, vals)
    for (ρi, vi) in zip(ρs, vals)
        mul!(nonzeros(ρi), ρ_projector, vi, 1.0, 0.0)
    end
end

function get_coloring(edgelist)
    if isempty(edgelist)
        colors = fill(1, num_free_dofs(U))
        num_colors = 1
        return colors, num_colors
    end
    g = SimpleGraphFromIterator(edgelist)
    coloring = Graphs.degree_greedy_color(g)
    colors = coloring.colors
    num_colors = coloring.num_colors
    return colors, num_colors    
end

function compute_edgelist_inner!(edgelist, U_ids, V_ids)
    U_cache = Gridap.array_cache(U_ids)
    V_cache = Gridap.array_cache(V_ids)
    for cell_idx in eachindex(U_ids, V_ids)
        U_id = getindex!(U_cache, U_ids, cell_idx)
        V_id = getindex!(V_cache, V_ids, cell_idx)
        for i in eachindex(U_id)
            u_id = Int64(U_id[i])
            for j in eachindex(V_id)
                v_id = Int64(V_id[j])
                if u_id <= v_id continue end
                edge = Graphs.SimpleEdge(u_id, v_id)
                push!(edgelist, edge)
            end
        end
    end
end

function compute_edgelist(domain_contrib, U, V)
    edgelist = Graphs.SimpleEdge{Int64}[]
    for Ω in Gridap.CellData.get_domains(domain_contrib)
        U_ids = Gridap.get_cell_dof_ids(U, Ω)
        V_ids = Gridap.get_cell_dof_ids(V, Ω)
        sizehint!(edgelist, (length(first(U_ids))-1)^2*length(U_ids))
        compute_edgelist_inner!(edgelist, U_ids, V_ids)
    end
    return edgelist
end

function compute_sparsity_structure_inner!(Is, Js, U_ids, V_ids, cell_mats, colors, glue)
    cell_mat_cache = array_cache(cell_mats)
    U_cache = Gridap.array_cache(U_ids)
    V_cache = Gridap.array_cache(V_ids)

    for cell_idx in eachindex(U_ids, V_ids)
        U_id = getindex!(U_cache, U_ids, cell_idx)
        V_id = getindex!(V_cache, V_ids, cell_idx)

        cell_mat = getindex!(cell_mat_cache, cell_mats, cell_idx)

        j_p = Int64(glue.tface_to_mface[cell_idx])

        for (j, U_idj) in enumerate(U_id)
            c = colors[U_idj]
            for (i, V_idi) in enumerate(V_id)
                val = cell_mat[i, j]::Float64
                if !iszero(val)
                    push!(Is[c], j_p)
                    push!(Js[c], V_idi)
                end
            end
        end
    end
end

function compute_sparsity_structure(domain_contrib, U, V, colors, num_colors)
    Is = [Int64[] for _ in 1:num_colors]
    Js = [Int64[] for _ in 1:num_colors]
    Dc = num_dims(get_background_model(get_triangulation(U)))

    for Ω_temp in Gridap.CellData.get_domains(domain_contrib)
        scell_mat = Gridap.FESpaces.get_contribution(domain_contrib, Ω_temp)
        cell_mats, Ω = Gridap.move_contributions(scell_mat, Ω_temp)
        cell_mats_c = Gridap.FESpaces.attach_constraints_cols(U,cell_mats,Ω)
        cell_mats_rc = Gridap.FESpaces.attach_constraints_rows(V,cell_mats_c,Ω)

        # cell_mats = domain_contrib[Ω]
        glue = Gridap.get_glue(Ω, Val(Dc))
        U_ids = Gridap.get_cell_dof_ids(U, Ω)
        V_ids = Gridap.get_cell_dof_ids(V, Ω)
        compute_sparsity_structure_inner!(Is, Js, U_ids, V_ids, cell_mats_rc, colors, glue)
    end

    m = length(get_cell_dof_ids(U))
    n = num_free_dofs(V)
    return [sparse(Is[i], Js[i], ones(length(Is[i])), m, n) for i in eachindex(Is, Js)]
end

function compute_projectors_inner!(I_ps, J_ps, V_ps, U_ids, V_ids, cell_mats, colors, glue, A_skeletons)
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
            c = colors[U_idj]

            for i in eachindex(V_id)
                V_idi = Int64(V_id[i])
                val = cell_mat[i, j]::Float64

                if !iszero(val)
                    i_p = find_nzval_index(A_skeletons[c], j_p, V_idi)
                    # j_p = cell_idx
                    push!(I_ps[c], i_p)
                    push!(J_ps[c], U_idj)
                    push!(V_ps[c], val)
                    # to detect whether A_skeleton is diag. (the value is arbitrary here assuming 1.0)
                    # A_skeletons[c].nzval[i_p] += 1.0
                end
            end
        end
    end
end

function compute_projectors(domain_contrib, U, V, colors, num_colors, A_skeletons)
    Dc = num_dims(get_background_model(get_triangulation(U)))

    # the matrices that map into the sparsity structure of A_skeletons
    I_ps = [Int64[] for _ in 1:num_colors]
    J_ps = [Int64[] for _ in 1:num_colors]
    V_ps = [Float64[] for _ in 1:num_colors]
    for (I_p, J_p, V_p, A_skeleton) in zip(I_ps, J_ps, V_ps, A_skeletons)
        sizehint!(I_p, length(A_skeleton.nzval))
        sizehint!(J_p, length(A_skeleton.nzval))
        sizehint!(V_p, length(A_skeleton.nzval))
    end

    for Ω_temp in Gridap.CellData.get_domains(domain_contrib)
        scell_mat = Gridap.FESpaces.get_contribution(domain_contrib, Ω_temp)
        cell_mats, Ω = Gridap.move_contributions(scell_mat, Ω_temp)
        cell_mats_c = Gridap.FESpaces.attach_constraints_cols(U,cell_mats,Ω)
        cell_mats_rc = Gridap.FESpaces.attach_constraints_rows(V,cell_mats_c,Ω)

        # cell_mats = domain_contrib[Ω]
        glue = Gridap.get_glue(Ω, Val(Dc))
        U_ids = Gridap.get_cell_dof_ids(U, Ω)
        V_ids = Gridap.get_cell_dof_ids(V, Ω)
        compute_projectors_inner!(I_ps, J_ps, V_ps, U_ids, V_ids, cell_mats_rc, colors, glue, A_skeletons)
    end

    ms = [length(A_skeleton.nzval) for A_skeleton in A_skeletons]
    n = num_free_dofs(U)
    return [sparse(I_p, J_p, V_p, m, n) for (I_p, J_p, V_p, m) in zip(I_ps, J_ps, V_ps, ms)]
end
    

function build_backprojector(g, U, V)
    v = get_fe_basis(V)
    u = get_trial_fe_basis(U)

    domain_contrib = g(u, v)

    # compute dof_id colors
    edgelist = compute_edgelist(domain_contrib, U, V)
    colors, num_colors = get_coloring(edgelist)

    # compute sparsity structure of the sparse matrices
    A_skeletons = compute_sparsity_structure(domain_contrib, U, V, colors, num_colors)

    projectors = compute_projectors(domain_contrib, U, V, colors, num_colors, A_skeletons)

    A_skeletons = [num_colors == 1 && isdiag(A_skeleton) ? Diagonal(Vector(diag(A_skeleton))) : A_skeleton for A_skeleton in A_skeletons]
    projectors = [num_colors == 1 && isdiag(projector) ? Diagonal(Vector(diag(projector))) : projector for projector in projectors]
    return A_skeletons, projectors
end

function mul_into_nonzero!(skeleton::SparseMatrixCSC, projector, v)
    mul!(skeleton.nzval, projector, v)
end

function mul_into_nonzero!(skeleton::Diagonal, projector, v)
    mul!(skeleton.diag, projector, v)
end

function backproject!(res, (skeletons, projectors), u, v)
    for (skeleton, projector) in zip(skeletons, projectors)
        mul_into_nonzero!(skeleton, projector, v)
    end
    for skeleton in skeletons
        mul!(res, skeleton, u, 1.0, 1.0)
    end
    return res
end

# end

# function backproject_old!(b, backprojector, u, v)
#     cache, backpr = backprojector
#     for (i, (A, u_id, v_id)) in enumerate(backpr)
#         @inbounds mul!(cache, A, @view(u[u_id]))
#         @inbounds b[i] = dot(@view(v[v_id]), cache)
#     end
# end

# function build_backprojector_old(g, U, V)
#     v = get_fe_basis(V)
#     u = get_trial_fe_basis(U)

#     domain_contrib = g(u, v)
#     Dc = num_dims(get_background_model(get_triangulation(U)))

#     backprojector = Tuple{Matrix{Float64}, Vector{Int32}, Vector{Int32}}[]
#     resize!(backprojector, num_cells(get_background_model(get_triangulation(U))))

#     for Ω in Gridap.CellData.get_domains(domain_contrib)
#         cell_mats = domain_contrib[Ω]

#         glue = Gridap.get_glue(Ω, Val(Dc))

#         U_ids = Gridap.get_cell_dof_ids(U, Ω)
#         V_ids = Gridap.get_cell_dof_ids(V, Ω)
#         for cell_idx in 1:Gridap.num_cells(Ω)
#             U_id = U_ids[cell_idx]
#             V_id = V_ids[cell_idx]

#             cell_mat = cell_mats[cell_idx]
#             idx = glue.tface_to_mface[cell_idx]

#             if isassigned(backprojector, idx)
#                 @assert backprojector[idx][2] == U_id
#                 @assert backprojector[idx][3] == V_id
#                 backprojector[idx][1] .+= cell_mat
#             else
#                 backprojector[idx] = (Matrix(cell_mat), Vector(U_id), Vector(V_id))
#             end 
#         end
#     end

#     for i in eachindex(backprojector)
#         if !isassigned(backprojector, i)
#             backprojector[i] = (zeros(0, 0), zeros(Int64, 0), zeros(Int64, 0))
#         end
#     end
    
#     return (zeros(3), backprojector)
# end

# using Revise

# using Gridap
# using GridapGmsh
# # using SparseArrays
# # using LinearAlgebra
# # using Graphs

# model = CartesianDiscreteModel((0, 1, 0, 1), (300, 100))

# # model = DiscreteModelFromFile("square.msh")

# U = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1); conformity=:H1)
# # W = TestFESpace(model, ReferenceFE(lagrangian, Float64, 0); conformity=:L2)
# V = TrialFESpace(U)

# Ω = Triangulation(model)
# Γ = BoundaryTriangulation(model)
# dx = Measure(Ω, 4)
# dΓ = Measure(Γ, 4)

# n = get_normal_vector(Γ)

# a(u, v) = ∫(u*v)*dx + ∫(u*dot(∇(v), n))*dΓ

# GridapBiFormProjectors.build_projector(a, U, V)
# GridapBiFormProjectors.build_backprojector(a, U, V)

# u_func = FEFunction(U, rand(num_free_dofs(U)))
# v_func = FEFunction(V, rand(num_free_dofs(V)))

# backr_old = build_backprojector_old(a, U, V);
# backr = build_backprojector(a, U, V);

# build_projector(a, U, V)

# using BenchmarkTools
# res = zeros(num_free_dofs(W));
# backproject!(res, backr, v_func.free_values, u_func.free_values)

# res2 = zeros(num_free_dofs(W));
# backproject_old!(res2, backr_old, u_func.free_values, v_func.free_values)
#     # res2
# isapprox.(res .- res2, 0.0, atol=1e-15) |> all