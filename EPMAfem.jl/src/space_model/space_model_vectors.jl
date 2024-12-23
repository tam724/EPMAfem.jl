
# ∫R_μv(v, μ, (dims, R, dx, ∂R, dΓ, n)) = ∫(μ*v)dx
∫R_v(v, (dims, R, dx, ∂R, dΓ, n)) = ∫(v)dx

@concrete struct ∫∂R_ngv{D}
    g
end

(int::∫∂R_ngv{Z})(v, (dims, R, dx, ∂R, dΓ, n)) = ∫(nz(n, dims)*int.g*v)dΓ
(int::∫∂R_ngv{X})(v, (dims, R, dx, ∂R, dΓ, n)) = ∫(nx(n, dims)*int.g*v)dΓ
(int::∫∂R_ngv{Y})(v, (dims, R, dx, ∂R, dΓ, n)) = ∫(ny(n, dims)*int.g*v)dΓ

@concrete struct ∫R_μv
    μ
end

(int::∫R_μv)(v, (dims, R, dx, ∂R, dΓ, n)) = ∫(int.μ*v)dx    

function assemble_linear(b, model, V)
    v = get_fe_basis(V)
    args = get_args(model)
    veccontribs = b(v, args)
    data = Gridap.FESpaces.collect_cell_vector(V, veccontribs)
    return assemble_vector(SparseMatrixAssembler(V, V), data)
end