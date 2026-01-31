# now this is special code for EPMA
function blockmatrix(A, B, C, D)
    # weird hack: # TODO!
    if any(A -> A isa CUDA.CUSPARSE.CuSparseMatrixCSC, (A, B, C, D, transpose(A), transpose(B), transpose(C), transpose(D)))
        A_, B_, C_, D_ = collect.((A, B, C, D))
        return sparse([ A_ B_
                        C_ D_]) |> cu
    end
    return [A B
            C D]
end

const BlockMatrix{T} = LazyOpMatrix{T, typeof(blockmatrix), <:Tuple{<:AbstractMatrix{T}, <:AbstractMatrix{T}, <:AbstractMatrix{T}, <:AbstractMatrix{T}}, _NO_KWARGS}
A(BM::BlockMatrix) = BM.args[1]
B(BM::BlockMatrix) = BM.args[2]
C(BM::BlockMatrix) = BM.args[3]
D(BM::BlockMatrix) = BM.args[4]

function LinearAlgebra.transpose(BM::BlockMatrix)
    At, Bt, Ct, Dt = transpose.(blocks(BM))
    return lazy(blockmatrix, At, Ct, Bt, Dt)
end

block_size(BM::BlockMatrix) = (
    only_unique((size(A(BM), 1), size(A(BM), 2), size(B(BM), 1), size(C(BM), 2))), 
    only_unique((size(D(BM), 1), size(D(BM), 2), size(B(BM), 2), size(C(BM), 1)))
)

# may be weaker
max_block_size(BM::BlockMatrix) = (
    only_unique((max_size(A(BM), 1), max_size(A(BM), 2), max_size(B(BM), 1), max_size(C(BM), 2))), 
    only_unique((max_size(D(BM), 1), max_size(D(BM), 2), max_size(B(BM), 2), max_size(C(BM), 1)))
)

blocks(BM::BlockMatrix) = A(BM), B(BM), C(BM), D(BM)
diag_blocks(BM::BlockMatrix) = A(BM), B(BM), C(BM), D(BM)

Base.size(BM::BlockMatrix) = duplicate(sum(block_size(BM)))
max_size(BM::BlockMatrix) = duplicate(sum(max_block_size(BM)))
isdiagonal(BM::BlockMatrix) = false # would need B === 0
structure(::BlockMatrix) = DenseStructure()

function lazy_getindex(BM::BlockMatrix, i::Int, j::Int)
    mA, nA = size(A(BM))

    if i <= mA && j <= nA
        return A(BM)[i, j]
    elseif i <= mA && j > nA
        return B(BM)[i, j - nA]
    elseif i > mA && j <= nA
        return C(BM)[i - mA, j]
    else
        return D(BM)[i - mA, j - nA]
    end
end

function mul_with!(ws::Workspace, y::AbstractVector, @nospecialize(BM::BlockMatrix), x::AbstractVector, α::Number, β::Number)
    n1, n2 = block_size(BM)

    x1 = @view(x[1:n1])
    x2 = @view(x[n1+1:n1+n2])

    y1 = @view(y[1:n1])
    y2 = @view(y[n1+1:n1+n2])

    CUDA.NVTX.@range "blockmatrix mul!" begin
        CUDA.NVTX.@range "A-block" begin
            mul_with!(ws, y1, A(BM), x1, α, β)        
        end
        CUDA.NVTX.@range "B-block" begin
            mul_with!(ws, y1, B(BM), x2, α, true)
        end
        CUDA.NVTX.@range "Bt-block" begin
            mul_with!(ws, y2, C(BM), x1, α, β)
        end
        CUDA.NVTX.@range "D-block" begin
            mul_with!(ws, y2, D(BM), x2, α, true)
        end
    end
end

function required_workspace(::typeof(mul_with!), BM::BlockMatrix, n, cache_notifier)
    @assert n == 1
    return maximum(required_workspace(mul_with!, A_, n, cache_notifier) for A_ in (A(BM), B(BM), C(BM), D(BM)))
end

materialize_with(ws::Workspace, BM::BlockMatrix, skeleton::AbstractMatrix) = materialize_with(ws, BM, skeleton, true, false)
function materialize_with(ws::Workspace, BM::BlockMatrix, skeleton::AbstractMatrix, α::Number, β::Number)
    n1, n2 = block_size(BM)
    materialize_with(ws, A(BM), @view(skeleton[1:n1, 1:n1]), α, β)
    materialize_with(ws, B(BM), @view(skeleton[1:n1, n1+1:n1+n2]), α, β)
    materialize_with(ws, C(BM), @view(skeleton[n1+1:n1+n2, 1:n1]), α, β)
    materialize_with(ws, D(BM), @view(skeleton[n1+1:n1+n2, n1+1:n1+n2]), α, β)
    return skeleton, ws
end
required_workspace(::typeof(materialize_with), BM::BlockMatrix, cache_notifier) = maximum(A -> required_workspace(materialize_with, A, cache_notifier), blocks(BM))

## blockdiag
function blockdiagmatrix(A, B)
    mA, nA = size(A)
    mB, nB = size(B)
    @assert mA == nA
    @assert mB == nB
    # weird hack: # TODO!
    if any(A -> A isa CUDA.CUSPARSE.CuSparseMatrixCSC, (A, B, transpose(A), transpose(B)))
        A_, B_ = collect.((A, B))
        return sparse([ A_ spzeros(mA, nB)
                        spzeros(mB, nA) B_]) |> cu
    end
    return [A spzeros(mA, nB)
            spzeros(mB, nA) D]
end

const BlockDiagMatrix{T} = LazyOpMatrix{T, typeof(blockdiagmatrix), <:Tuple{<:AbstractMatrix{T}, <:AbstractMatrix{T}}, _NO_KWARGS}
A(BM::BlockDiagMatrix) = BM.args[1]
B(BM::BlockDiagMatrix) = BM.args[2]

function LinearAlgebra.transpose(BM::BlockDiagMatrix)
    At, Bt = transpose.(blocks(BM))
    return lazy(blockmatrix, At, Ct, Bt, Dt)
end

block_size(BM::BlockDiagMatrix) = (
    only_unique((size(A(BM), 1), size(A(BM), 2))), 
    only_unique((size(B(BM), 1), size(B(BM), 2)))
)

# may be weaker
max_block_size(BM::BlockDiagMatrix) = (
    only_unique((max_size(A(BM), 1), max_size(A(BM), 2))), 
    only_unique((max_size(B(BM), 1), max_size(B(BM), 2)))
)

blocks(BM::BlockDiagMatrix) = A(BM), B(BM)

duplicate(x) = (x, x)
Base.size(BM::BlockDiagMatrix) = duplicate(sum(block_size(BM)))
max_size(BM::BlockDiagMatrix) = duplicate(sum(max_block_size(BM)))
isdiagonal(BM::BlockDiagMatrix) = isdiagonal(A(BM)) && isdiagonal(B(BM))
structure(BM::BlockDiagMatrix) = isdiagonal(A(BM)) && isdiagonal(B(BM)) ? DiagonalStructure() : DenseStructure()

function lazy_getindex(BM::BlockDiagMatrix{T}, i::Int, j::Int) where T
    mA, nA = size(A(BM))

    if i <= mA && j <= nA
        return A(BM)[i, j]
    elseif i <= mA && j > nA
        return zero(T)
    elseif i > mA && j <= nA
        return zero(T)
    else
        return B(BM)[i - mA, j - nA]
    end
end

function mul_with!(ws::Workspace, y::AbstractVector, @nospecialize(BM::BlockDiagMatrix), x::AbstractVector, α::Number, β::Number)
    n1, n2 = block_size(BM)

    x1 = @view(x[1:n1])
    x2 = @view(x[n1+1:n1+n2])

    y1 = @view(y[1:n1])
    y2 = @view(y[n1+1:n1+n2])

    CUDA.NVTX.@range "blockdiagmatrix mul!" begin
        CUDA.NVTX.@range "A-block" begin
            mul_with!(ws, y1, A(BM), x1, α, β)        
        end
        CUDA.NVTX.@range "D-block" begin
            mul_with!(ws, y2, B(BM), x2, α, true)
        end
    end
end

function required_workspace(::typeof(mul_with!), BM::BlockDiagMatrix, n, cache_notifier)
    @assert n == 1
    return maximum(required_workspace(mul_with!, A_, n, cache_notifier) for A_ in blocks(BM))
end

materialize_with(ws::Workspace, BM::BlockDiagMatrix, skeleton::AbstractMatrix) = materialize_with(ws, BM, skeleton, true, false)
function materialize_with(ws::Workspace, BM::BlockDiagMatrix, skeleton::AbstractMatrix, α::Number, β::Number)
    n1, n2 = block_size(BM)
    materialize_with(ws, A(BM), @view(skeleton[1:n1, 1:n1]), α, β)
    materialize_with(ws, B(BM), @view(skeleton[n1+1:n1+n2, n1+1:n1+n2]), α, β)
    return skeleton, ws
end
required_workspace(::typeof(materialize_with), BM::BlockDiagMatrix, cache_notifier) = maximum(A -> required_workspace(materialize_with, A, cache_notifier), blocks(BM))
