# direct solver
const InverseMatrix{T} = LazyOpMatrix{T, typeof(Base.inv), <:Tuple{<:AbstractMatrix{T}}, _NO_KWARGS}
A(K::InverseMatrix) = only(K.args)

Base.size(K::InverseMatrix) = size(A(K))
max_size(K::InverseMatrix) = max_size(A(K))
isdiagonal(K::InverseMatrix) = isdiagonal(A(K))

lazy_getindex(::InverseMatrix, ::Int, ::Int) = error("Cannot getindex")
LinearAlgebra.transpose(K::InverseMatrix) = lazy(inv, transpose(A(K)))

function mul_with!(ws::Workspace, Y::AbstractVecOrMat, @nospecialize(K::InverseMatrix), X::AbstractVecOrMat, α::Number, β::Number)
    if isdiagonal(A(K))
        A_, _ = materialize_with(ws, materialize(A(K)))
        ldiv!(Y, A_, X, α, β)
    else
        @assert isone(α)
        @assert iszero(β)
        A_, _ = materialize_with(ws, lazy(decide_materialize_strategy(A(K)), A(K))) # enforce materialization
        ldiv!(Y, lu!(A), X)
    end
end

function mul_with!(ws::Workspace, Y::AbstractVecOrMat, X::AbstractVecOrMat, @nospecialize(K::InverseMatrix), α::Number, β::Number)
    @assert isone(α)
    @assert iszero(β)
    if isdiagonal(A(K))
        A_, _ = materialize_with(ws, materialize(A(K)))
        copyto!(Y, X)
        rdiv!(Y, A_)
    else
        A_, _ = materialize_with(ws, lazy(decide_materialize_strategy(A(K)), A(K))) # enforce materialization
        copyto!(Y, X)
        rdiv!(Y, lu!(A_))
    end
end

function required_workspace(::typeof(mul_with!), K::InverseMatrix, n, cache_notifier)
    if isdiagonal(A(K))
        return required_workspace(materialize_with, materialize(A(K)), cache_notifier)
    else
        return required_workspace(materialize_with, lazy(decide_materialize_strategy(A(K)), A(K)), cache_notifier)
    end
end

function materialize_with(ws::Workspace, K::InverseMatrix, skeleton::AbstractMatrix)
    A_mat, _ = materialize_with(ws, A(K), skeleton)
    LinearAlgebra.inv!(A_mat)
    return A_mat, ws
end

function required_workspace(::typeof(materialize_with), K::InverseMatrix, cache_notifier)
    return required_workspace(materialize_with, A(K), cache_notifier)
end

# krylov_minres
const KrylovMinresMatrix{T} = LazyOpMatrix{T, typeof(Krylov.minres), <:Tuple{<:AbstractMatrix{T}}, <:Any}
A(K::KrylovMinresMatrix) = only(K.args)

Base.size(K::KrylovMinresMatrix) = size(A(K))
max_size(K::KrylovMinresMatrix) = max_size(A(K))
isdiagonal(K::KrylovMinresMatrix) = isdiagonal(A(K))

lazy_getindex(K::KrylovMinresMatrix, i::Int, j::Int) = error("Cannot getindex")

# function krylov_minres_solver_view(temp, x::UnsafeArray)
#     n = size(x, 1)
#     T = eltype(x)
#     skt = solver.lin_solver_skeleton
#     stats = SimpleStats(0, false, false, T[], T[], T[], 0.0, "unknown")
#     return MinresSolver{T, T, typeof(x)}(
#         n, 
#         n,
#         uview(T[], 1:0),
#         x,
#         uview(temp, 0*n+1:1*n), # r1
#         uview(temp, 1*n+1:2*n), # r2
#         uview(temp, 2*n+1:3*n), # w1
#         uview(temp, 3*n+1:4*n), # w2
#         uview(temp, 4*n+1:5*n), # y
#         uview(T[], 1:0), # v
#         skt.err_vec,
#         false,
#         stats)
# end

function LinearAlgebra.transpose(K::KrylovMinresMatrix) # matrix and preconditioner should be symmetric!
    kwargs = K.kwargs
    kwargs = haskey(K.kwargs, :M) ? (; kwargs..., M=transpose(K.kwargs.M)) : kwargs
    return lazy(Krylov.gmres, transpose(A(K)); kwargs...)
end

function mul_with!(ws::Workspace, y::AbstractVector, @nospecialize(K::KrylovMinresMatrix{T}), x::AbstractVector, α::Number, β::Number) where T
    CUDA.NVTX.@range "minres solve" begin
        A_ = NotSoLazy{T}(A(K), ws)

        kwargs = K.kwargs
        kwargs = haskey(kwargs, :M) ? Base.setindex(kwargs, NotSoLazy{T}(kwargs.M, ws), :M) : kwargs
        kwargs = haskey(kwargs, :rtol) ? Base.setindex(kwargs, T(kwargs.rtol), :rtol) : kwargs
        kwargs = haskey(kwargs, :atol) ? Base.setindex(kwargs, T(kwargs.atol), :atol) : kwargs

        solver = Krylov.MinresWorkspace(A_, x) # this allocates!
        Krylov.minres!(solver, A_, x; kwargs...)
        @show solver.stats
        y .= α .* solver.x .+ β .* y
    end
end

function required_workspace(::typeof(mul_with!), K::KrylovMinresMatrix, n, cache_notifier)
    @assert n == 1
    req_ws = required_workspace(mul_with!, A(K), n, cache_notifier)
    req_ws = haskey(K.kwargs, :M) ? max(req_ws, required_workspace(mul_with!, K.kwargs.M, n, cache_notifier)) : req_ws
    return req_ws
end

# krylov_gmres
const KrylovGmresMatrix{T} = LazyOpMatrix{T, typeof(Krylov.gmres), <:Tuple{<:AbstractMatrix{T}}, <:Any}
A(K::KrylovGmresMatrix) = K.args[1]

Base.size(K::KrylovGmresMatrix) = size(A(K))
max_size(K::KrylovGmresMatrix) = max_size(A(K))
isdiagonal(K::KrylovGmresMatrix) = isdiagonal(A(K))

lazy_getindex(K::KrylovGmresMatrix, i::Int, j::Int) = error("Cannot getindex")

function LinearAlgebra.transpose(K::KrylovGmresMatrix)
    kwargs = K.kwargs
    kwargs = haskey(K.kwargs, :M) ? (; kwargs..., N=transpose(K.kwargs.M)) : (; kwargs..., N=I)
    kwargs = haskey(K.kwargs, :N) ? (; kwargs..., M=transpose(K.kwargs.N)) : (; kwargs..., M=I)
    return lazy(Krylov.gmres, transpose(A(K)); kwargs...)
end

function mul_with!(ws::Workspace, y::AbstractVector, @nospecialize(K::KrylovGmresMatrix{T}), x::AbstractVector, α::Number, β::Number) where T
    A_ = NotSoLazy{T}(A(K), ws)
    
    kwargs = K.kwargs
    kwargs = haskey(kwargs, :M) && typeof(kwargs.M) <: AbstractLazyMatrix ? (; kwargs..., M=NotSoLazy{T}(kwargs.M, ws)) : kwargs
    kwargs = haskey(kwargs, :N) && typeof(kwargs.N) <: AbstractLazyMatrix ? (; kwargs..., N=NotSoLazy{T}(kwargs.N, ws)) : kwargs
    kwargs = haskey(kwargs, :rtol) ? (; kwargs..., rtol=T(kwargs.rtol)) : kwargs
    kwargs = haskey(kwargs, :atol) ? (; kwargs..., atol=T(kwargs.atol)) : kwargs

    solver = Krylov.GmresWorkspace(A_, x) # this allocates!
    Krylov.gmres!(solver, A_, x; kwargs...)
    @show solver.stats
    y .= α .* solver.x .+ β .* y
end

function required_workspace(::typeof(mul_with!), K::KrylovGmresMatrix, n, cache_notifier)
    @assert n == 1
    req_ws = required_workspace(mul_with!, A(K), n, cache_notifier)
    req_ws = haskey(K.kwargs, :M) && typeof(K.kwargs.M) <: AbstractLazyMatrix ? max(req_ws, required_workspace(mul_with!, K.kwargs.M, n, cache_notifier)) : req_ws
    req_ws = haskey(K.kwargs, :N) && typeof(K.kwargs.N) <: AbstractLazyMatrix ? max(req_ws, required_workspace(mul_with!, K.kwargs.N, n, cache_notifier)) : req_ws
    return req_ws
end

# krylov cg
const KrylovCGMatrix{T} = LazyOpMatrix{T, typeof(Krylov.cg), <:Tuple{<:AbstractMatrix{T}}, <:Any}
A(K::KrylovCGMatrix) = K.args[1]

Base.size(K::KrylovCGMatrix) = size(A(K))
max_size(K::KrylovCGMatrix) = max_size(A(K))
isdiagonal(K::KrylovCGMatrix) = isdiagonal(A(K))

lazy_getindex(::KrylovCGMatrix, ::Int, ::Int) = error("Cannot getindex")

function LinearAlgebra.transpose(K::KrylovCGMatrix) # matrix and preconditioner should be symmetric!
    kwargs = K.kwargs
    kwargs = haskey(K.kwargs, :M) ? (; kwargs..., M=transpose(K.kwargs.M)) : kwargs
    return lazy(Krylov.gmres, transpose(A(K)); kwargs...)
end

function mul_with!(ws::Workspace, y::AbstractVector, @nospecialize(K::KrylovCGMatrix{T}), x::AbstractVector, α::Number, β::Number) where T
    A_ = NotSoLazy{T}(A(K), ws)
    
    kwargs = K.kwargs
    kwargs = haskey(kwargs, :M) && typeof(kwargs.M) <: AbstractLazyMatrixOrTranspose ? (; kwargs..., M=NotSoLazy{T}(kwargs.M, ws)) : kwargs
    kwargs = haskey(kwargs, :rtol) ? (; kwargs..., rtol=T(kwargs.rtol)) : kwargs
    kwargs = haskey(kwargs, :atol) ? (; kwargs..., atol=T(kwargs.atol)) : kwargs

    solver = Krylov.CgWorkspace(A_, x) # this allocates!
    Krylov.cg!(solver, A_, x; kwargs...)
    @show solver.stats
    y .= α .* solver.x .+ β .* y
end

function required_workspace(::typeof(mul_with!), K::KrylovCGMatrix, n, cache_notifier)
    @assert n == 1
    req_ws = required_workspace(mul_with!, A(K), n, cache_notifier)
    req_ws = haskey(K.kwargs, :M) ? max(req_ws, required_workspace(mul_with!, K.kwargs.M, n, cache_notifier)) : req_ws
    return req_ws
end

# # krylov tricg
# const KrylovTriCGMatrix{T} = LazyOpMatrix{T, typeof(Krylov.tricg), <:NTuple{3, AbstractMatrix{T}}, _NO_KWARGS}
# A⁻¹(K::KrylovTriCGMatrix) = K.args[1]
# B(K::KrylovTriCGMatrix) = K.args[2]
# C⁻¹(K::KrylovTriCGMatrix) = K.args[3]

# function Base.size(K::KrylovTriCGMatrix)
#     n1 = only_unique(size(A⁻¹(K)))
#     n2 = only_unique(size(C⁻¹(K)))
#     size(B(K)) == (n1, n2) || error("size mismatch")
#     return duplicate(n1 + n2)
# end

# function max_size(K::KrylovTriCGMatrix)
#     n1 = only_unique(max_size(A⁻¹(K)))
#     n2 = only_unique(max_size(C⁻¹(K)))
#     max_size(B) == (n1, n2) || error("size mismatch")
#     return duplicate(n1 + n2)
# end

# isdiagonal(K::KrylovTriCGMatrix) = false # only if B(K) == 0
# lazy_getindex(::KrylovTriCGMatrix, i::Int, j::Int) = error("Cannot getindex")

# function mul_with!(ws::Workspace, y::AbstractVector, @nospecialize(K::KrylovTriCGMatrix{T}), x::AbstractVector, α::Number, β::Number) where T
#     n1, n2 = only_unique(size(A⁻¹(K))), only_unique(size(C⁻¹(K)))
#     _A⁻¹ = NotSoLazy{T}(A⁻¹(K), ws)
#     _B = NotSoLazy{T}(B(K), ws)
#     _C⁻¹ = NotSoLazy{T}(C⁻¹(K), ws)
#     solver = Krylov.TricgSolver(_B, x) # this allocates!
#     a_, b_ = @view(x[1:n1]), @view(x[n1+1:n1+n2])
#     Krylov.solve!(solver, _B, a_, b_; M=_A⁻¹, N=_C⁻¹, rtol=T(sqrt(eps(Float64))), atol=zero(T))
#     @show "TRICG", solver.stats
#     u_, v_ = @view(y[1:n1]), @view(y[n1+1:n1+n2])
#     u_ .= α .* solver.x .+ β .* u_
#     v_ .= α .* solver.y .+ β .* v_
# end

# function required_workspace(::typeof(mul_with!), K::KrylovTriCGMatrix, n, cache_notifier)
#     @assert n == 1
#     return max(required_workspace(mul_with!, A⁻¹(K), n, cache_notifier),
#         required_workspace(mul_with!, B(K), n, cache_notifier),
#         required_workspace(mul_with!, C⁻¹(K), n, cache_notifier))
# end

function schur_complement() end

const SchurMatrix{T} = LazyOpMatrix{T, typeof(schur_complement), <:NTuple{4, AbstractMatrix{T}}, _NO_KWARGS}
inv_AmBD⁻¹C(S::SchurMatrix) = S.args[1]
B(S::SchurMatrix) = S.args[2]
C(S::SchurMatrix) = S.args[3]
D⁻¹(S::SchurMatrix) = S.args[4]

LinearAlgebra.transpose(S::SchurMatrix) = lazy(schur_complement, transpose(inv_AmBD⁻¹C(S)), transpose(C(S)), transpose(B(S)), transpose(D⁻¹(S)))

function block_size(S::SchurMatrix)
    n1 = only_unique(size(inv_AmBD⁻¹C(S)))
    n2 = only_unique(size(D⁻¹(S)))
    return (n1, n2)
end
block_size(St::Transpose{T, <:SchurMatrix{T}}) where T = block_size(parent(St))

function Base.size(S::SchurMatrix)
    n1, n2 = block_size(S)
    @assert size(B(S)) == (n1, n2)
    @assert size(C(S)) == (n2, n1)
    return duplicate(n1 + n2)
end
function max_size(S::SchurMatrix)
    n1 = only_unique(max_size(inv_AmBD⁻¹C(S)))
    n2 = only_unique(max_size(D⁻¹(S)))
    @assert max_size(D(S)) == (n1, n2)
    @assert max_size(C(S)) == (n2, n1)
    return duplicate(n1 + n2)
end
isdiagonal(S::SchurMatrix) = false # should not happen..

lazy_getindex(S::SchurMatrix, i::Int, j::Int) = error("Cannot getindex")

function mul_with!(ws::Workspace, y::AbstractVector, @nospecialize(S::SchurMatrix{T}), x::AbstractVector, α::Number, β::Number) where T
    @assert iszero(β)
    n1, n2 = block_size(S)

    tmp, rem = take_ws(ws, max(n1, n2))
    b_x, b_y = @view(tmp[1:n1]), @view(tmp[1:n2]) # aliased!
    u, v = @view(x[1:n1]), @view(x[n1+1:n1+n2])
    x_, y_ = @view(y[1:n1]), @view(y[n1+1:n1+n2])

    copyto!(b_x, u)
    mul_with!(rem, b_x, B(S)*D⁻¹(S), v, -α, α)
    mul_with!(rem, x_, inv_AmBD⁻¹C(S), b_x, true, false)
    copyto!(b_y, v)
    mul_with!(rem, b_y, C(S), x_, -one(T), α)
    mul_with!(rem, y_, D⁻¹(S), b_y, true, false)
end

function mul_with!(ws::Workspace, y::AbstractVector, @nospecialize(St::Transpose{T, <:SchurMatrix{T}}), x::AbstractVector, α::Number, β::Number) where T
    S = parent(St)
    @assert iszero(β)
    n1, n2 = block_size(St)
    
    tmp, rem = take_ws(ws, max(n1, n2))
    b_x, b_y = @view(tmp[1:n1]), @view(tmp[1:n2]) # aliased!
    u, v = @view(x[1:n1]), @view(x[n1+1:n1+n2])
    x_, y_ = @view(y[1:n1]), @view(y[n1+1:n1+n2])

    copyto!(b_x, u)
    mul_with!(rem, b_x, transpose(C(S)) * transpose(D⁻¹(S)), v, -α, α)
    mul_with!(rem, x_, transpose(inv_AmBD⁻¹C(S)), b_x, true, false)
    copyto!(b_y, v)
    mul_with!(rem, b_y, transpose(B(S)), x_, -one(T), α)
    mul_with!(rem, y_, transpose(D⁻¹(S)), b_y, true, false)
end

function required_workspace(::typeof(mul_with!), S::SchurMatrix, n, cache_notifier)
    @assert n == 1
    n1 = only_unique(max_size(inv_AmBD⁻¹C(S)))
    n2 = only_unique(max_size(D⁻¹(S)))
    max(n1, n2) + maximum(A -> required_workspace(mul_with!, A, n, cache_notifier),
        (inv_AmBD⁻¹C(S), B(S)*D⁻¹(S), C(S), D⁻¹(S), transpose(C(S))*transpose(D⁻¹(S)), transpose(B(S))))
end
