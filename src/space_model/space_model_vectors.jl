
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
