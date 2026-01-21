const SumMatrix{T} = LazyOpMatrix{T, typeof(+), <:Tuple{Vararg{AbstractMatrix{T}}}}
@inline As(S::SumMatrix) = S.args
Base.size(S::SumMatrix) = only_unique(size(A) for A in As(S))
max_size(S::SumMatrix) = only_unique(max_size(A) for A in As(S))
lazy_getindex(S::SumMatrix, idx::Vararg{Integer}) = +(getindex.(As(S), idx...)...)
isdiagonal(S::SumMatrix) = all(isdiagonal, As(S))
LinearAlgebra.transpose(S::SumMatrix) = lazy(+, transpose.(As(S))...)


## mul_with
function mul_with!(ws::Workspace, Y::AbstractVecOrMat, S::SumMatrix, X::AbstractVecOrMat, α::Number, β::Number)
    CUDA.NVTX.@range "mul_with! SumMatrix" begin
        for A in As(S)
            mul_with!(ws, Y, A, X, α, β)
            β = true
        end
    end
end
function mul_with!(ws::Workspace, Y::AbstractMatrix, X::AbstractMatrix, S::SumMatrix, α::Number, β::Number)
    CUDA.NVTX.@range "mul_with! SumMatrix" begin
        for A in As(S)
            mul_with!(ws, Y, X, A, α, β)
            β = true
        end
    end
end
required_workspace(::typeof(mul_with!), S::SumMatrix, n, cache_notifier) = maximum(required_workspace(mul_with!, A, n, cache_notifier) for A in As(S))


materialize_with(ws::Workspace, S::SumMatrix, skeleton::AbstractMatrix) = materialize_with(ws, S, skeleton, true, false)
function materialize_with(ws::Workspace, S::SumMatrix, skeleton::AbstractMatrix, α::Number, β::Number)
    S_mat, _ = materialize_with(ws, first(As(S)), skeleton, α, β)
    for A in As(S)[2:end]
        S_mat, _ = materialize_with(ws, A, skeleton, α, true)
    end
    return S_mat, ws
end
required_workspace(::typeof(materialize_with), S::SumMatrix, cache_notifier) = maximum(required_workspace(materialize_with, A, cache_notifier) for A in As(S))


function materialize_diag_with(ws::Workspace, S::SumMatrix, skeleton::Diagonal, α::Number, β::Number)
    S_diag, _ = materialize_diag_with(ws, first(As(S)), skeleton, α, β)
    for A in As(S)[2:end]
        S_diag, _ = materialize_diag_with(ws, A, skeleton, α, true)
    end
    return S_diag, ws
end
