

function allocate_minres_krylov_buf(VT, m, n; window::Int=5)
    T = eltype(VT)
    Δx = VT(undef, 0)
    # x  = S(undef, n) # stateful
    r1 = VT(undef, n)
    r2 = VT(undef, n)
    w1 = VT(undef, n)
    w2 = VT(undef, n)
    y  = VT(undef, n)
    v  = VT(undef, 0)
    err_vec = zeros(T, window)
    return (m, n, VT, Δx, r1, r2, w1, w2, y, v, err_vec)
end

cuview(A::Array, slice) = uview(A, slice)
cuview(A::SubArray, slice) = uview(A, slice)
cuview(A::CuArray, slice) = view(A, slice)

function solver_from_buf(x, buf)
    m, n, VT, Δx, r1, r2, w1, w2, y, v, err_vec = buf
    @assert x isa VT
    T = eltype(VT)
    @assert all(sz -> sz == n, (length(x), length(r1), length(r2), length(w1), length(w2), length(y))) 
    stats = SimpleStats(0, false, false, T[], T[], T[], 0.0, "unknown")
    return MinresSolver{T,T,VT}(m, n, Δx, x, r1, r2, w1, w2, y, v, err_vec, false, stats)
end

function solver_from_buf(x::UnsafeArray, buf)
    m, n, VT, Δx, r1, r2, w1, w2, y, v, err_vec = buf
    @assert eltype(x) == eltype(VT)
    T = eltype(VT)
    @assert all(sz -> sz == n, (length(x), length(r1), length(r2), length(w1), length(w2), length(y))) 
    stats = SimpleStats(0, false, false, T[], T[], T[], 0.0, "unknown")
    return MinresSolver{T,T,UnsafeArray{T, 1}}(m, n, uview(Δx, 1:0), x, uview(r1, 1:n), uview(r2, 1:n), uview(w1, 1:n), uview(w2, 1:n), uview(y, 1:n), uview(v, 1:0), err_vec, false, stats)
end


# reduce the dimensions of the solver.
# function get_lin_solver(bs::MinresSolver{T, FC, S}, m, n) where {T, FC, S<:CuArray{T, 1}}
#     ## pull the solver internal caches from the "big solver", that is stored in the type
#     fill!(bs.err_vec, zero(T))
#     stats = bs.stats
#     stats.niter, stats.solved, stats.inconsistent, stats.timer, stats.status = 0, false, false, 0.0, "unknown"
#     return Krylov.MinresSolver{T, FC, S}(m, n, @view(bs.Δx[1:0]), @view(bs.x[1:n]), @view(bs.r1[1:n]), @view(bs.r2[1:n]), @view(bs.w1[1:n]), @view(bs.w2[1:n]), @view(bs.y[1:n]), @view(bs.v[1:0]), bs.err_vec, false, stats)
# end

# function get_lin_solver(bs::MinresSolver{T, FC, S}, m, n) where {T, FC, S<:Vector{T}}
#     ## pull the solver internal caches from the "big solver", that is stored in the type
#     fill!(bs.err_vec, zero(T))
#     stats = bs.stats
#     stats.niter, stats.solved, stats.inconsistent, stats.timer, stats.status = 0, false, false, 0.0, "unknown"
#     return Krylov.MinresSolver{T, FC, UnsafeArray{T, 1}}(m, n, uview(bs.Δx,1:0), uview(bs.x, 1:n), uview(bs.r1, 1:n), uview(bs.r2, 1:n), uview(bs.w1, 1:n), uview(bs.w2, 1:n), uview(bs.y, 1:n), uview(bs.v, 1:0), bs.err_vec, false, stats)
# end

