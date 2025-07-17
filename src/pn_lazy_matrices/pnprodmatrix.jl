const ScaleMatrixL{T, M <: AbstractMatrix{T}} = LazyOpMatrix{T, typeof(*), <:Tuple{Base.RefValue{T}, M}}
const ScaleMatrixR{T, M <: AbstractMatrix{T}} = LazyOpMatrix{T, typeof(*), <:Tuple{M, Base.RefValue{T}}}
const ScaleMatrix{T, M <: AbstractMatrix{T}} = Union{ScaleMatrixL{T, M}, ScaleMatrixR{T, M}}
# ScaleMatricex where M is a "normal" Matrix type should support the 5-arg mul! -> when mul_with!(::ScaleMatrix) we simply fuse the coefficient into a mul!( with α = a(S)*α_)
const NotFusedScaleMatrix{T} = ScaleMatrix{T, <:AbstractLazyMatrixOrTranspose{T}}
# we cannot do this for ScaleMatrices where the M is a LazyMatrix -> NonFusedScaleMatrix
# somewhat that copies what we do with ShouldBroadcastMaterialize, maybe there is duplicate information here (TODO: figure this out..)

@inline a(S::ScaleMatrixL) = S.args[1][]
@inline A(S::ScaleMatrixL) = S.args[2]
@inline a(S::ScaleMatrixR) = S.args[2][]
@inline A(S::ScaleMatrixR) = S.args[1]
Base.size(S::ScaleMatrix) = size(A(S))
max_size(S::ScaleMatrix) = max_size(A(S))
lazy_getindex(S::ScaleMatrix, idx::Vararg{<:Integer}) = *(a(S), getindex(A(S), idx...))
@inline isdiagonal(S::ScaleMatrix) = isdiagonal(A(S))

mul_with!(ws::Workspace, Y::AbstractVecOrMat, S::NotFusedScaleMatrix, X::AbstractVecOrMat, α::Number, β::Number) = mul_with!(ws, Y, A(S), X, a(S)*α, β)
mul_with!(ws::Workspace, Y::AbstractMatrix, X::AbstractMatrix, S::NotFusedScaleMatrix, α::Number, β::Number) = mul_with!(ws, Y, X, A(S), a(S)*α, β)
mul_with!(ws::Workspace, Y::AbstractVecOrMat, St::Transpose{T, <:NotFusedScaleMatrix{T}}, X::AbstractVecOrMat, α::Number, β::Number) where T = mul_with!(ws, Y, transpose(A(parent(St))), X, a(parent(St))*α, β)
mul_with!(ws::Workspace, Y::AbstractMatrix, X::AbstractMatrix, St::Transpose{T, <:NotFusedScaleMatrix{T}}, α::Number, β::Number) where T = mul_with!(ws, Y, X, transpose(A(parent(St))), a(parent(St))*α, β)
required_workspace(::typeof(mul_with!), S::NotFusedScaleMatrix) = required_workspace(mul_with!, A(S))

# for ScaleMatrix (fused) the mul_with! simply calls back into mul!
mul_with!(::Workspace, Y::AbstractVecOrMat, S::ScaleMatrix, X::AbstractVecOrMat, α::Number, β::Number) = mul!(Y, A(S), X, a(S)*α, β)
mul_with!(::Workspace, Y::AbstractMatrix, X::AbstractMatrix, S::ScaleMatrix, α::Number, β::Number) = mul!(Y, X, A(S), a(S)*α, β)
mul_with!(::Workspace, Y::AbstractVecOrMat, St::Transpose{T, <:ScaleMatrix{T}}, X::AbstractVecOrMat, α::Number, β::Number) where T= mul!(Y, transpose(A(parent(St))), X, a(parent(St))*α, β)
mul_with!(::Workspace, Y::AbstractMatrix, X::AbstractMatrix, St::Transpose{T, <:ScaleMatrix{T}}, α::Number, β::Number) where T = mul!(Y, X, transpose(A(parent(St))), a(parent(St))*α, β)
required_workspace(::typeof(mul_with!), S::ScaleMatrix) = 0

# and we can additionally overload the LinearAlgebra.mul! calls by simply fusing in the a(S)
LinearAlgebra.mul!(Y::AbstractVector, S::ScaleMatrix, X::AbstractVector, α::Number, β::Number) = mul!(Y, A(S), X, a(S)*α, β)
LinearAlgebra.mul!(Y::AbstractMatrix, S::ScaleMatrix, X::AbstractMatrix, α::Number, β::Number) = mul!(Y, A(S), X, a(S)*α, β)
LinearAlgebra.mul!(Y::AbstractMatrix, X::AbstractMatrix, S::ScaleMatrix, α::Number, β::Number) = mul!(Y, X, A(S), a(S)*α, β)
LinearAlgebra.mul!(Y::AbstractVector, St::Transpose{T, <:ScaleMatrix{T}}, X::AbstractVector, α::Number, β::Number) where T = mul!(Y, transpose(A(parent(St))), X, a(parent(St))*α, β)
LinearAlgebra.mul!(Y::AbstractMatrix, St::Transpose{T, <:ScaleMatrix{T}}, X::AbstractMatrix, α::Number, β::Number) where T = mul!(Y, transpose(A(parent(St))), X, a(parent(St))*α, β)
LinearAlgebra.mul!(Y::AbstractMatrix, X::AbstractMatrix, St::Transpose{T, <:ScaleMatrix{T}}, α::Number, β::Number) where T = mul!(Y, X, transpose(A(parent(St))), a(parent(St))*α, β)

_rmul!(A::AbstractArray, α::Number) = rmul!(A, α)
_rmul!(A::Diagonal, α::Number) = rmul!(A.diag, α)

function materialize_with(ws::Workspace, S::ScaleMatrix, skeleton::AbstractMatrix)
    A_mat, _ = materialize_with(ws, A(S), skeleton)
    _rmul!(A_mat, a(S))
    # A_mat .= a(S) .* A_mat
    return A_mat, ws
end

broadcast_materialize(S::ScaleMatrix) = broadcast_materialize(A(S))
materialize_broadcasted(ws::Workspace, S::ScaleMatrix) = Base.Broadcast.broadcasted(*, a(S), materialize_broadcasted(ws, A(S)))
required_workspace(::typeof(materialize_with), S::ScaleMatrix) = required_workspace(materialize_with, A(S))

# it seems as if now the fun starts :D this can be heavily optimized (matrix product chain, etc..) well only go for some simple heuristics here
# let's start implementing this with only A*B (the general case follows later..)
const TwoProdMatrix{T} = LazyOpMatrix{T, typeof(*), <:Tuple{<:AbstractMatrix{T}, <:AbstractMatrix{T}}}
@inline A(M::TwoProdMatrix) = M.args[1]
@inline B(M::TwoProdMatrix) = M.args[2]
function Base.size(M::TwoProdMatrix)
    if (size(A(M), 2) != size(B(M), 1))
        error("size mismatch")
    end
    return (size(A(M), 1), size(B(M), 2))
end
function max_size(M::TwoProdMatrix)
    if max_size(A(M), 2) != max_size(B(M), 1)
        error("size mismatch")
    end
    return (max_size(A(M), 1), max_size(B(M), 2))
end

function lazy_getindex(M::TwoProdMatrix, i::Int, j::Int)
    return sum(getindex(A(M), i, k)*getindex(B(M), k, j) for k in 1:only_unique((size(A(M), 2), size(B(M), 1))))
end
@inline isdiagonal(M::TwoProdMatrix) = isdiagonal(A(M)) && isdiagonal(B(M))

mul_with!(ws::Workspace, Y::AbstractMatrix, X::AbstractMatrix, M::TwoProdMatrix, α::Number, β::Number) = mul_with!(ws, transpose(Y), transpose(M), transpose(X), α, β)
mul_with!(ws::Workspace, Y::AbstractMatrix, X::AbstractMatrix, Mt::Transpose{T, <:TwoProdMatrix{T}}, α::Number, β::Number) where T = mul_with!(ws, transpose(Y), parent(Mt), transpose(X), α, β)

function mul_with!(ws::Workspace, y::AbstractVector, M::TwoProdMatrix, x::AbstractVector, α::Number, β::Number)
    WS, rem = take_ws(ws, size(B(M), 1))
    mul_with!(rem, WS, B(M), x, true, false)
    mul_with!(rem, y, A(M), WS, α, β)
end

function mul_with!(ws::Workspace, Y::AbstractMatrix, M::TwoProdMatrix, X::AbstractMatrix, α::Number, β::Number)
    WS, rem = take_ws(ws, size(B(M), 1))
    # if !Base.iscontiguous(@view(X[:, 1])) @warn "@view(X[:, j]) is not contiguous!" end
    # if !Base.iscontiguous(@view(Y[:, 1])) @warn "@view(Y[:, j]) is not contiguous!" end

    # temp, rem = take_ws(rem, max(size(X, 1), size(Y, 1)))
    for j in 1:size(X, 2)
        # if Base.iscontiguous(@view(X[:, j]))
            mul_with!(rem, WS, B(M), @view(X[:, j]), true, false)
        # else
        #     @view(temp[1:size(X, 1)]) .= @view(X[:, j])
        #     mul_with!(rem, WS, B(M), @view(temp[1:size(X, 1)]), true, false)
        # end
        # if Base.iscontiguous(@view(Y[:, j]))
            mul_with!(rem, @view(Y[:, j]), A(M), WS, α, β)
        # else
        #     mul_with!(rem, @view(temp[1:size(Y, 1)]), A(M), WS, α, β)
        #     @view(Y[:, j]) .= @view(temp[1:size(Y, 1)])
        # end
    end
end

function mul_with!(ws::Workspace, y::AbstractVector, Mt::Transpose{T, <:TwoProdMatrix{T}}, x::AbstractVector, α::Number, β::Number) where T
    M = parent(Mt)
    WS, rem = take_ws(ws, size(B(M), 1)) # == size(A(M), 2)
    mul_with!(rem, WS, transpose(A(M)), x, true, false)
    mul_with!(rem, y, transpose(B(M)), WS, α, β)
end

function mul_with!(ws::Workspace, Y::AbstractMatrix, Mt::Transpose{T, <:TwoProdMatrix{T}}, X::AbstractMatrix, α::Number, β::Number) where T
    M = parent(Mt)
    WS, rem = take_ws(ws, size(B(M), 1)) # == size(A(M), 2)
    # if !Base.iscontiguous(@view(X[:, 1])) @warn "@view(X[:, j]) is not contiguous!" end
    # if !Base.iscontiguous(@view(Y[:, 1])) @warn "@view(Y[:, j]) is not contiguous!" end
    # temp, rem = take_ws(rem, max(size(X, 1), size(Y, 1)))
    for j in 1:size(X, 2)
        # if Base.iscontiguous(@view(X[:, j]))
            mul_with!(rem, WS, transpose(A(M)), @view(X[:, j]), true, false)
        # else
        #     copyto!(@view(temp[1:size(X, 1)]), @view(X[:, j]))
        #     mul_with!(rem, WS, transpose(A(M)), @view(temp[1:size(X, 1)]), true, false)
        # end
        # if Base.iscontiguous(@view(Y[:, j]))
            mul_with!(rem, @view(Y[:, j]), transpose(B(M)), WS, α, β)
        # else
        #     mul_with!(rem, @view(temp[1:size(Y, 1)]), transpose(B(M)), WS, α, β)
        #     copyto!(@view(Y[:, j]), @view(temp[1:size(Y, 1)]))
        # end
    end
end

# assuming a vector input (matrix input is simply looped, TODO: maybe introduce threaded ? )
function required_workspace(::typeof(mul_with!), M::TwoProdMatrix)
    return size(B(M), 1) +  max(required_workspace(mul_with!, B(M)), required_workspace(mul_with!, A(M)))

    # this feels unneccesary, we allocate more space to potentially copy the array inputs, to have contiguous memory in the loops
    # return size(B(M), 1) + max(size(A(M), 1), size(B(M), 2)) + max(required_workspace(mul_with!, B(M)), required_workspace(mul_with!, A(M)))
end

function materialize_with(ws::Workspace, M::TwoProdMatrix, skeleton::AbstractMatrix)
    A_mat, rem = materialize_with(ws, materialize(A(M)), nothing)
    B_mat, _ = materialize_with(rem, materialize(B(M)), nothing)
    mul!(skeleton, A_mat, B_mat)
    return skeleton, ws
end

function required_workspace(::typeof(materialize_with), M::TwoProdMatrix)
    return required_workspace(materialize_with, materialize(A(M))) + required_workspace(materialize_with, materialize(B(M)))
end

# ProdMatrix
const ProdMatrix{T} = LazyOpMatrix{T, typeof(*), <:Tuple{Vararg{<:AbstractMatrix{T}}}}
@inline As(M::ProdMatrix) = M.args
function Base.size(M::ProdMatrix)
    size(first(As(M)), 1), _product_matrix_size(size, As(M)...)
end
function max_size(M::ProdMatrix)
    max_size(first(As(M)), 1), _product_matrix_size(max_size, As(M)...)
end

function _product_matrix_size(size_op::Function, (a, b, rest...)::Vararg{<:AbstractMatrix})
    if size_op(a, 2) != size_op(b, 1) error("size mismatch") end
    return _product_matrix_size(size_op, b, rest...)
end

function _product_matrix_size(size_op::Function, (a, b)::Vararg{<:AbstractMatrix, 2})
    if size_op(a, 2) != size_op(b, 1) error("size mismatch") end
    return size_op(b, 2)
end

function lazy_getindex(M::ProdMatrix, i::Int, j::Int)
    # naive implementation
    @warn "getindex of ProdMatrix is probably very slow!" maxlog=5
    x = zeros(maximum(A -> size(A, 2), As(M)))
    y = zeros(maximum(A -> size(A, 2), As(M)))
    A = As(M)[end]
    copyto!(@view(x[1:size(A, 1)]), @view(A[:, j]))
    for i in lastindex(As(M)) - 1 : -1 : firstindex(As(M))
        A = As(M)[i]
        # this should somehow avoid to call mul!...
        mul!(@view(y[1:size(A, 1)]), A, @view(x[1:size(A, 2)]))
        x, y = y, x
    end
    return x[i]
end
isdiagonal(M::ProdMatrix) = all(isdiagonal, As(M))

mul_with!(ws::Workspace, Y::AbstractMatrix, X::AbstractMatrix, M::ProdMatrix, α::Number, β::Number) = mul_with!(ws, transpose(Y), transpose(M), transpose(X), α, β)
mul_with!(ws::Workspace, Y::AbstractMatrix, X::AbstractMatrix, Mt::Transpose{T, <:ProdMatrix{T}}, α::Number, β::Number) where T = mul_with!(ws, transpose(Y), parent(Mt), transpose(X), α, β)

# no strategy here, simply multiply right to left for now.. (its vector anyways)
function mul_with!(ws::Workspace, y::AbstractVector, M::ProdMatrix, x::AbstractVector, α::Number, β::Number)
    my_ws = maximum(A -> size(A, 2), As(M))
    r_, rem = take_ws(ws, my_ws)
    l_, rem = take_ws(rem, my_ws)
    mul_with!(rem, @view(r_[1:size(last(As(M)), 1)]), last(As(M)), x, true, false)
    for (i, A) in enumerate(reverse(As(M)))
        if i == 1 || i == length(As(M)) continue end # skip the first and last iteration
        mul_with!(rem, @view(l_[1:size(A, 1)]), A, @view(r_[1:size(A, 2)]), true, false)
        r_, l_ = l_, r_
    end
    mul_with!(rem, y, first(As(M)), @view(r_[1:size(first(As(M)), 2)]), α, β)
end

# simply loop over the multiple X's (consider materializing first...)
function mul_with!(ws::Workspace, Y::AbstractMatrix, M::ProdMatrix, X::AbstractMatrix, α::Number, β::Number)
    my_ws_size = maximum(A -> size(A, 2), As(M))
    r_, rem = take_ws(ws, my_ws_size)
    l_, rem = take_ws(rem, my_ws_size)
    # if !Base.iscontiguous(@view(X[:, 1])) @warn "@view(X[:, j]) is not contiguous!" end
    # if !Base.iscontiguous(@view(Y[:, 1])) @warn "@view(Y[:, j]) is not contiguous!" end
    for j in 1:size(X, 2)
        mul_with!(rem, @view(r_[1:size(last(As(M)), 1)]), last(As(M)), @view(X[:, j]), true, false)
        for (i, A) in enumerate(reverse(As(M)))
            if i == 1 || i == length(As(M)) continue end # skip the first and last iteration
            mul_with!(rem, @view(l_[1:size(A, 1)]), A, @view(r_[1:size(A, 2)]), true, false)
            r_, l_ = l_, r_
        end
        mul_with!(rem, @view(Y[:, j]), first(As(M)), @view(r_[1:size(first(As(M)), 2)]), α, β)
    end
end

function mul_with!(ws::Workspace, y::AbstractVector, Mt::Transpose{T, <:ProdMatrix{T}}, x::AbstractVector, α::Number, β::Number) where T
    M = parent(Mt)
    n = length(As(M))
    my_ws = maximum(A -> size(A, 1), As(M))
    r_, rem = take_ws(ws, my_ws)
    l_, rem = take_ws(rem, my_ws)
    # Start with the first (transposed) matrix (which is last in the original product)
    mul_with!(rem, @view(r_[1:size(first(As(M)), 2)]), transpose(first(As(M))), x, true, false)
    for (i, A) in enumerate(As(M))
        if i == 1 || i == n continue end # skip the first and last iteration
        mul_with!(rem, @view(l_[1:size(A, 2)]), transpose(A), @view(r_[1:size(A, 1)]), true, false)
        r_, l_ = l_, r_
    end
    mul_with!(rem, y, transpose(last(As(M))), @view(r_[1:size(last(As(M)), 1)]), α, β)
end

# simply loop over the multiple X's (consider materializing first...)
function mul_with!(ws::Workspace, Y::AbstractMatrix, Mt::Transpose{T, <:ProdMatrix{T}}, X::AbstractMatrix, α::Number, β::Number) where T
    M = parent(Mt)
    n = length(As(M))
    my_ws = maximum(A -> size(A, 1), As(M))
    r_, rem = take_ws(ws, my_ws)
    l_, rem = take_ws(rem, my_ws)
    # if !Base.iscontiguous(@view(X[:, 1])) @warn "@view(X[:, j]) is not contiguous!" end
    # if !Base.iscontiguous(@view(Y[:, 1])) @warn "@view(Y[:, j]) is not contiguous!" end
    for j in 1:size(X, 2)
        # Start with the first (transposed) matrix (which is last in the original product)
        mul_with!(rem, @view(r_[1:size(first(As(M)), 2)]), transpose(first(As(M))), @view(X[:, j]), true, false)
        for (i, A) in enumerate(As(M))
            if i == 1 || i == n continue end # skip the first and last iteration
            mul_with!(rem, @view(l_[1:size(A, 2)]), transpose(A), @view(r_[1:size(A, 1)]), true, false)
            r_, l_ = l_, r_
        end
        mul_with!(rem, @view(Y[:, j]), transpose(last(As(M))), @view(r_[1:size(last(As(M)), 1)]), α, β)
    end
end

function required_workspace(::typeof(mul_with!), M::ProdMatrix)
    my_ws = maximum(A -> size(A, 2), As(M)) # we could skip the last here
    int_ws = maximum(A -> required_workspace(mul_with!, A), As(M))
    return 2*my_ws + int_ws
end

function materialize_with(ws::Workspace, M::ProdMatrix, skeleton::AbstractMatrix) # TODO: (GJCBP)
    max_m = maximum(A -> size(A, 1), As(M))
    max_n = maximum(A -> size(A, 2), As(M))
    max_intermediate = max_m*max_n
    T1, rem = take_ws(ws, max_intermediate)

    Aₙ, rem_ = materialize_with(rem, materialize(last(As(M))), nothing)
    Aₙ₋₁, _ = materialize_with(rem_, materialize(As(M)[end-1]), nothing)
    mul!(mat_view(T1, size(Aₙ₋₁, 1), size(Aₙ, 2)), Aₙ₋₁, Aₙ)
    T2, rem_ = take_ws(rem, max_intermediate)
    for i in length(As(M))-2:-1:2
        Aᵢ, _ = materialize_with(rem_, materialize(As(M)[i]), nothing)
        mul!(mat_view(T2, size(Aᵢ, 1), size(Aₙ, 2)), Aᵢ, mat_view(T1, size(Aᵢ, 2), size(Aₙ, 2)))
        T1, T2 = T2, T1
    end
    A₁, _ = materialize_with(rem_, materialize(As(M)[1]), nothing)

    # the final result is always T1
    mul!(skeleton, A₁, mat_view(T1, size(A₁, 2), size(Aₙ, 2)))
    return skeleton, ws
end

function required_workspace(::typeof(materialize_with), M::ProdMatrix)
    # simply exaggerate here.. # TODO bring the workspace size down !
    max_m = maximum(A -> size(A, 1), As(M))
    max_n = maximum(A -> size(A, 2), As(M))
    max_internals = maximum(A -> required_workspace(materialize_with, materialize(A)), As(M))
    return 2*max_m*max_n + max_internals
end
