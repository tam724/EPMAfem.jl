abstract type AbstractMatrixStructure end

struct DenseStructure <: AbstractMatrixStructure end
struct DiagonalStructure <: AbstractMatrixStructure end
struct BlockDiagonalStructure{N} <: AbstractMatrixStructure end

(::DenseStructure)(vals::AbstractMatrix) = vals
(::DiagonalStructure)(diag::AbstractVector) = Diagonal(diag)
(::BlockDiagonalStructure{N})(diag::AbstractVector) where N = BlockDiagonal{N}(diag)

required_workspace(::DenseStructure, (m, n)) = (m, n)
function required_workspace(::DiagonalStructure, (m, n))
    @assert m == n
    return (m, )
end

function required_workspace(::BlockDiagonalStructure{N}, (m, n)) where N
    @assert m == n
    return (m*N, )
end
