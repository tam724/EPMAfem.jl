
∫μv(v, μ, (dims, R, dx, ∂R, dΓ, n)) = ∫(μ*v)dx
∫v(v, (dims, R, dx, ∂R, dΓ, n)) = ∫(v)dx
∫nzgv(v, g, (dims, R, dx, ∂R, dΓ, n)) = ∫(nz(n, dims)*g*v)dΓ
∫nxgv(v, g, (dims, R, dx, ∂R, dΓ, n)) = ∫(nx(n, dims)*g*v)dΓ
∫nygv(v, g, (dims, R, dx, ∂R, dΓ, n)) = ∫(ny(n, dims)*g*v)dΓ

function assemble_linear(b, model, V)
    v = get_fe_basis(V)
    args = get_args(model)
    veccontribs = b(v, args)
    data = Gridap.FESpaces.collect_cell_vector(V, veccontribs)
    return assemble_vector(SparseMatrixAssembler(V, V), data)
end