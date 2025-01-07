∂(dim, u, dims) = dot(cartesian_unit_vector(dim, dims), ∇(u))
∂x(u, dims) = ∂(X(), u, dims)
∂y(u, dims) = ∂(Y(), u, dims)
∂z(u, dims) = ∂(Z(), u, dims)

∫R_ρuv(ρ) = (u, v, (dims, R, dx, ∂R, dΓ, n)) -> ∫(ρ*u*v)dx
∫R_uv(u, v, (dims, R, dx, ∂R, dΓ, n)) = ∫(u*v)dx
# ∫R_v(v, (dims, R, dx, ∂R, dΓ, n)) = ∫(v)dx

∫R_∂zu_v(u, v, (dims, R, dx, ∂R, dΓ, n)) = ∫(∂z(u, dims)*v)dx
∫R_∂xu_v(u, v, (dims, R, dx, ∂R, dΓ, n)) = ∫(∂x(u, dims)*v)dx
∫R_∂yu_v(u, v, (dims, R, dx, ∂R, dΓ, n)) = ∫(∂y(u, dims)*v)dx

∫R_u_∂zv(u, v, (dims, R, dx, ∂R, dΓ, n)) = ∫(u*∂z(v, dims))dx
∫R_u_∂xv(u, v, (dims, R, dx, ∂R, dΓ, n)) = ∫(u*∂x(v, dims))dx
∫R_u_∂yv(u, v, (dims, R, dx, ∂R, dΓ, n)) = ∫(u*∂y(v, dims))dx

∫R_∂u_v(::_1D) = (∫R_∂zu_v, )
∫R_∂u_v(::_2D) = (∫R_∂zu_v, ∫R_∂xu_v)
∫R_∂u_v(::_3D) = (∫R_∂zu_v, ∫R_∂xu_v, ∫R_∂yu_v)

∫R_u_∂v(::_1D) = (∫R_u_∂zv, )
∫R_u_∂v(::_2D) = (∫R_u_∂zv, ∫R_u_∂xv)
∫R_u_∂v(::_3D) = (∫R_u_∂zv, ∫R_u_∂xv, ∫R_u_∂yv)

ndot(dim, n, dims) = dot(n, cartesian_unit_vector(dim, dims))
nx(n, dims) = ndot(X(), n, dims)
ny(n, dims) = ndot(Y(), n, dims)
nz(n, dims) = ndot(Z(), n, dims)

∫∂R_absnz_uv(u, v, (dims, R, dx, ∂R, dΓ, n)) = ∫(abs(nz(n, dims))*u*v)dΓ
∫∂R_absnx_uv(u, v, (dims, R, dx, ∂R, dΓ, n)) = ∫(abs(nx(n, dims))*u*v)dΓ
∫∂R_absny_uv(u, v, (dims, R, dx, ∂R, dΓ, n)) = ∫(abs(ny(n, dims))*u*v)dΓ

∫∂R_absn_uv(::_1D) = (∫∂R_absnz_uv, )
∫∂R_absn_uv(::_2D) = (∫∂R_absnz_uv, ∫∂R_absnx_uv)
∫∂R_absn_uv(::_3D) = (∫∂R_absnz_uv, ∫∂R_absnx_uv, ∫∂R_absny_uv)

function assemble_bilinear(a, model, U, V)
    u = get_trial_fe_basis(TrialFESpace(U))
    v = get_fe_basis(V)
    args = get_args(model)
    matcontribs = a(u, v, args)
    data = Gridap.FESpaces.collect_cell_matrix(U, V, matcontribs)
    return assemble_matrix(SparseMatrixAssembler(U, V), data)
end
