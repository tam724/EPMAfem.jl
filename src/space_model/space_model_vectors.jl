
# ∫R_μv(v, μ, (dims, dx, dΓ, dΓi, n)) = ∫(μ*v)dx
∫R_v(v, (dims, dx, dΓ, dΓi, n)) = ∫((x -> v(; to_args(x)...)))dx

@concrete struct ∫∂R_ngv{D<:Dimensions.SpaceDimension}
    g
end

(int::∫∂R_ngv{Z})(v, (dims, dx, dΓ, dΓi, n)) = ∫(nz(n, dims)*int.g*v)dΓ
(int::∫∂R_ngv{X})(v, (dims, dx, dΓ, dΓi, n)) = ∫(nx(n, dims)*int.g*v)dΓ
(int::∫∂R_ngv{Y})(v, (dims, dx, dΓ, dΓi, n)) = ∫(ny(n, dims)*int.g*v)dΓ

@concrete struct ∫∂Rp_ngv{SD<:Dimensions.SpaceDimension, SB<:Dimensions.SpaceBoundary}
    g
end

(int::∫∂Rp_ngv{Z, SB})(v, (dims, dx, dΓ, dΓi, n)) where {SB} = ∫(abs(nz(n, dims))*(x -> int.g(; Dimensions.omit(x, Z())...)) * v)dΓi[boundary_tag(Z(), SB(), dims)]
(int::∫∂Rp_ngv{X, SB})(v, (dims, dx, dΓ, dΓi, n)) where {SB} = ∫(abs(nx(n, dims))*(x -> int.g(; Dimensions.omit(x, X())...)) * v)dΓi[boundary_tag(X(), SB(), dims)]
(int::∫∂Rp_ngv{Y, SB})(v, (dims, dx, dΓ, dΓi, n)) where {SB} = ∫(abs(ny(n, dims))*(x -> int.g(; Dimensions.omit(x, Y())...)) * v)dΓi[boundary_tag(Y(), SB(), dims)]

@concrete struct ∫R_μv
    μ
end

(int::∫R_μv)(v, (dims, dx, dΓ, dΓi, n)) = ∫(int.μ*v)dx    

function assemble_linear(b, model, V)
    v = get_fe_basis(V)
    args = get_args(model)
    veccontribs = b(v, args)
    data = Gridap.FESpaces.collect_cell_vector(V, veccontribs)
    return assemble_vector(SparseMatrixAssembler(V, V), data)
end

# not faster..

# Integrates v -> ∫((x -> g(x, pᵢ)) * v) dx over Ω
function assemble_vectors(g, p, model, V, dx)
    B = zeros(num_free_dofs(V), length(p))
    Ω = Triangulation(model)
    V_ids = Gridap.get_cell_dof_ids(V, Ω)
    cell_points = Gridap.get_cell_points(dx.quad)
    quad_phys_s = cell_points.cell_phys_point

    V_cache = Gridap.array_cache(V_ids)
    quad_phys_cache = Gridap.array_cache(quad_phys_s)

    v = Gridap.get_fe_basis(V)
    basis_ref_eval = v(cell_points)
    quad_weights = dx.quad.cell_weight[1]

    abs_detJ = get_cell_measure(Ω)

    function assemble_cell_contribs!(
        B, V_ids, quad_phys_s, V_cache, quad_phys_cache,
        basis_ref_eval, quad_weights, abs_detJ, g, p
    )
        cell_local_eval = zeros(size(basis_ref_eval[1], 1), length(p))
        cell_local_quad = zeros(size(basis_ref_eval[1], 2), length(p))
        for cell_idx in eachindex(V_ids)
            V_id = getindex!(V_cache, V_ids, cell_idx)  # Local DOF ids
            quad_phys = getindex!(quad_phys_cache, quad_phys_s, cell_idx)  # Physical points
            φ_vals = basis_ref_eval[cell_idx]  # Shape: (num_quad_points, num_basis_functions)

            cell_local_eval .= g.(quad_phys, reshape(p, (1, :))) .* quad_weights
            mul!(cell_local_quad, φ_vals', cell_local_eval, abs_detJ[cell_idx], false)

            for idx in eachindex(IndexCartesian(), cell_local_quad)
                @inbounds B[V_id[idx[1]], idx[2]] += cell_local_quad[idx]
            end
            # end
        end
    end

    assemble_cell_contribs!(
        B, V_ids, quad_phys_s, V_cache, quad_phys_cache,
        basis_ref_eval, quad_weights, abs_detJ, g, p
    )

    return B
end
