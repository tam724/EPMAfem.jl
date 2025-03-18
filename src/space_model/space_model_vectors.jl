
# ∫R_μv(v, μ, (dims, R, dx, ∂R, dΓ, n)) = ∫(μ*v)dx
∫R_v(v, (dims, R, dx, ∂R, dΓ, n)) = ∫(v)dx

@concrete struct ∫∂R_ngv{D<:Dimensions.SpaceDimension}
    g
end

(int::∫∂R_ngv{Z})(v, (dims, R, dx, ∂R, dΓ, n)) = ∫(nz(n, dims)*int.g*v)dΓ
(int::∫∂R_ngv{X})(v, (dims, R, dx, ∂R, dΓ, n)) = ∫(nx(n, dims)*int.g*v)dΓ
(int::∫∂R_ngv{Y})(v, (dims, R, dx, ∂R, dΓ, n)) = ∫(ny(n, dims)*int.g*v)dΓ


@concrete struct ∫∂Rp_ngv{SD<:Dimensions.SpaceDimension, SB<:Dimensions.SpaceBoundary}
    g
end

# eval_if_pos_(n, n', x, g) = dot(n, n') > 0 ? dot(n, n')*g(x) : zero(x)

# function test(x, n)
#     @show x, n
# end

function inner_func(int::∫∂Rp_ngv{SD, SB}, x, n) where {SD, SB}
    dot(n, Dimensions.outwards_normal(SD(), SB(), ))
end


# (int::∫∂Rp_ngv{SD, SB})(v, (dims, R, dx, ∂R, dΓ, n)) where {SD<:Dimensions.SpaceDimension, SB<:Dimensions.SpaceBoundary} = ∫(relu(dot(n, Dimensions.outwards_normal(SD(), SB(), dims)))*(x -> int.g(Dimensions.constrain(x, SD())))*v)dΓ
# (int::∫∂Rp_ngv{SD, SB})(v, (dims, R, dx, ∂R, dΓ, n)) where {SD<:Dimensions.SpaceDimension, SB<:Dimensions.SpaceBoundary} = ∫(int.g(Dimensions.omit(x, SD())))dΓ

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
