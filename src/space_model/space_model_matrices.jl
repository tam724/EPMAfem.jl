∂(dim, u, dims) = dot(cartesian_unit_vector(dim, dims), ∇(u))
∂x(u, dims) = ∂(X(), u, dims)
∂y(u, dims) = ∂(Y(), u, dims)
∂z(u, dims) = ∂(Z(), u, dims)

∫R_ρuv(ρ) = (u, v, (dims, dx, dΓ, dΓi, n)) -> ∫(ρ*u*v)dx
∫R_uv(u, v, (dims, dx, dΓ, dΓi, n)) = ∫(u*v)dx
# ∫R_v(v, (dims, dx, dΓ, dΓi, n)) = ∫(v)dx

∫R_∂zu_v(u, v, (dims, dx, dΓ, dΓi, n)) = ∫(∂z(u, dims)*v)dx
∫R_∂xu_v(u, v, (dims, dx, dΓ, dΓi, n)) = ∫(∂x(u, dims)*v)dx
∫R_∂yu_v(u, v, (dims, dx, dΓ, dΓi, n)) = ∫(∂y(u, dims)*v)dx

∫R_u_∂zv(u, v, (dims, dx, dΓ, dΓi, n)) = ∫(u*∂z(v, dims))dx
∫R_u_∂xv(u, v, (dims, dx, dΓ, dΓi, n)) = ∫(u*∂x(v, dims))dx
∫R_u_∂yv(u, v, (dims, dx, dΓ, dΓi, n)) = ∫(u*∂y(v, dims))dx

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

∫∂R_absnz_uv(u, v, (dims, dx, dΓ, dΓi, n)) = ∫(abs(nz(n, dims))*u*v)dΓ
∫∂R_absnx_uv(u, v, (dims, dx, dΓ, dΓi, n)) = ∫(abs(nx(n, dims))*u*v)dΓ
∫∂R_absny_uv(u, v, (dims, dx, dΓ, dΓi, n)) = ∫(abs(ny(n, dims))*u*v)dΓ

∫∂R_absn_uv(::_1D) = (∫∂R_absnz_uv, )
∫∂R_absn_uv(::_2D) = (∫∂R_absnz_uv, ∫∂R_absnx_uv)
∫∂R_absn_uv(::_3D) = (∫∂R_absnz_uv, ∫∂R_absnx_uv, ∫∂R_absny_uv)

function _assemble_bilinear(a, args, U, V)
    u = get_trial_fe_basis(TrialFESpace(U))
    v = get_fe_basis(V)
    matcontribs = a(u, v, args)
    data = Gridap.FESpaces.collect_cell_matrix(U, V, matcontribs)
    return assemble_matrix(SparseMatrixAssembler(U, V), data)
end

function assemble_bilinear(a, model, U, V)
    return _assemble_bilinear(a, get_args(model), U, V)
end

rank_decomp(::typeof(∫∂R_absnz_uv), ::_1D) = (∫∂R_absnz_uv, )
rank_decomp(::typeof(∫∂R_absnz_uv), ::_2D) = (∫∂R_absnz_uv, ∫R_uv)
rank_decomp(::typeof(∫∂R_absnz_uv), ::_3D) = (∫∂R_absnz_uv, ∫R_uv, ∫R_uv)

rank_decomp(::typeof(∫∂R_absnx_uv), ::_2D) = (∫R_uv, ∫∂R_absnz_uv)
rank_decomp(::typeof(∫∂R_absnx_uv), ::_3D) = (∫R_uv, ∫∂R_absnz_uv, ∫R_uv)

rank_decomp(::typeof(∫∂R_absny_uv), ::_3D) = (∫R_uv, ∫R_uv, ∫∂R_absnz_uv)

rank_decomp(::typeof(∫R_u_∂zv), ::_1D) = (∫R_u_∂zv, )
rank_decomp(::typeof(∫R_u_∂zv), ::_2D) = (∫R_u_∂zv, ∫R_uv)
rank_decomp(::typeof(∫R_u_∂zv), ::_3D) = (∫R_u_∂zv, ∫R_uv, ∫R_uv)

rank_decomp(::typeof(∫R_u_∂xv), ::_2D) = (∫R_uv, ∫R_u_∂zv)
rank_decomp(::typeof(∫R_u_∂xv), ::_3D) = (∫R_uv, ∫R_u_∂zv, ∫R_uv)

rank_decomp(::typeof(∫R_u_∂yv), ::_3D) = (∫R_uv, ∫R_uv, ∫R_u_∂zv)

function assemble_bilinear(a::Union{typeof(∫R_u_∂zv), typeof(∫R_u_∂xv), typeof(∫R_u_∂yv), typeof(∫∂R_absnz_uv), typeof(∫∂R_absnx_uv), typeof(∫∂R_absny_uv)}, model::CartesianSpaceModel{N}, U, V) where N
    if U == plus(model)
        Us = model.plus_fe_spaces
    else
        @assert U == minus(model)
        Us = model.minus_fe_spaces
    end
    if V == plus(model)
        Vs = model.plus_fe_spaces
    else
        @assert V == minus(model)
        Vs = model.minus_fe_spaces
    end
    return kron(reverse(lazy.(Matrix.(_assemble_bilinear.(rank_decomp(a, dimensionality(model)), model._args, Us, Vs))))...)
    # return assemble_bilinear(a, model, U, V)
end
