
diagonal(A::AbstractArray) = Diagonal(diagview(A))

const DiagMatrix{T} = LazyOpMatrix{T, typeof(diagonal), <:Tuple{AbstractMatrix{T}}, <:Any}
A(D::DiagMatrix) = only(D.args)

Base.size(D::DiagMatrix) = size(A(D))
max_size(D::DiagMatrix) = max_size(A(D))

function lazy_getindex(D::DiagMatrix{T}, i::Int, j::Int) where {T}
    if j != i
        return zero(T)
    elseif D.kwargs.diagview == diagview || D.kwargs.diagview == diag
        return lazy_getindex(A(D), i, j)
    else
        error("Cannot getindex!")
    end
end

isdiagonal(D::DiagMatrix) = true
LinearAlgebra.transpose(D::DiagMatrix) = diagonal(transpose(A(D)))

materialize_with(ws::Workspace, D::DiagMatrix, skeleton::Diagonal) = materialize_with(ws, D, skeleton, true, false)
function materialize_with(ws::Workspace, D::DiagMatrix, skeleton::Diagonal, α::Number, β::Number)
    _A, _ = materialize_with(ws, materialize(A(D)))
    skeleton.diag .= α .* D.kwargs.diagview(_A) .+ β .* skeleton.diag
    return skeleton, ws
end

function required_workspace(::typeof(materialize_with), D::DiagMatrix, cache_notifier)
    return required_workspace(materialize_with, materialize(A(D)), cache_notifier)
end

lazy(::typeof(diagonal), S::SumMatrix; kwargs...) = lazy(+, lazy.(diagonal, As(S); kwargs...)...)
lazy(::typeof(diagonal), S::ScaleMatrix; kwargs...) = lazy(*, _a(S), lazy(diagonal, A(S); kwargs...))
lazy(::typeof(diagonal), K::KronMatrix; kwargs...) = lazy(kron, lazy.(diagonal, As(K); kwargs...)...)
