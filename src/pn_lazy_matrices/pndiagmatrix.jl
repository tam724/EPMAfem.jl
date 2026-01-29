
diagonal(A::AbstractArray) = Diagonal(diagview(A))

const DiagMatrix{T} = LazyOpMatrix{T, typeof(diagonal), <:Tuple{AbstractMatrix{T}}, _NO_KWARGS}
A(D::DiagMatrix) = only(D.args)

Base.size(D::DiagMatrix) = size(A(D))
max_size(D::DiagMatrix) = max_size(A(D))

function lazy_getindex(D::DiagMatrix{T}, i::Int, j::Int) where {T}
    if j != i
        return zero(T)
    else
        return lazy_getindex(A(D), i, j)
    end
end

isdiagonal(D::DiagMatrix) = true
LinearAlgebra.transpose(D::DiagMatrix) = diagonal(transpose(A(D)))

materialize_with(ws::Workspace, D::DiagMatrix, skeleton::Diagonal) = materialize_with(ws, D, skeleton, true, false)
function materialize_with(ws::Workspace, D::DiagMatrix, skeleton::Diagonal, α::Number, β::Number)
    _A, _ = materialize_with(ws, materialize(A(D)))
    skeleton.diag .= α .* diagview(_A) .+ β .* skeleton.diag
    return skeleton, ws
end

function required_workspace(::typeof(materialize_with), D::DiagMatrix, cache_notifier)
    return required_workspace(materialize_with, materialize(A(D)), cache_notifier)
end

PNLazyMatrices.lazy(::typeof(diagonal), S::SumMatrix) = lazy(+, diagonal.(As(S))...)
PNLazyMatrices.lazy(::typeof(diagonal), S::ScaleMatrix) = lazy(*, _a(S), diagonal(A(S)))
PNLazyMatrices.lazy(::typeof(diagonal), K::KronMatrix) = lazy(kron, diagonal.(As(K))...)
