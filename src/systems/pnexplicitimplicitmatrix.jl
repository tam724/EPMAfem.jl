## Naming Convention: lowercase: vector values, uppercase: matrix valued
##

function assemble_from_op(A_op)
    y = zeros(size(A_op)[2])
    e_i = zeros(size(A_op)[1])
    Is = Int64[]
    Js = Int64[]
    Vs = Float64[]
    for i in 1:size(A_op)[1]
        e_i[i] = 1.0
        mul!(y, A_op, e_i, true, false)
        e_i[i] = 0.0
        y_sparse = sparse(y)
        for j in nzrange(y_sparse, 1)
            push!(Is, y_sparse.nzind[j])
            push!(Js, i)
            push!(Vs, y_sparse.nzval[j])
        end
    end
    return sparse(Is, Js, Vs)
end

""" Constructs a view into tmp with matrix size (nL, nR)"""
function mat_view(tmp, nL, nR)
    return reshape(@view(tmp[1:nL*nR]), (nL, nR))
end

"""
computes
    Y = α*(∑I A_i * X + (γ_i B + ∑J δ_ij C_ij)) + β*Y
"""
@concrete struct ZMatrix2{T}
    A # AbstractVector (size I) of matrices of size nA1 x nA2
    B # Matrix of size nB1 x nB2
    C # AbstractVector of Vector (size I of J) of matrices of size nB1 x nB2

    γ # AbstractVector (size I) of scalars
    δ # AbstractVector of Vector (size I of J) of scalars

    # matrix sizes
    nA1
    nA2
    nB1
    nB2

    # iteration sizes
    I
    J

    W1 # CACHE: matrix of size nA1 x nB1
    W2 # CACHE: matrix of size nB1 x nB2
end

# constructor that figues out the matrix sizes:
function ZMatrix2(A, B, C, γ, δ, W1, W2)
    nA1, nA2 = size(first(A))
    nB1, nB2 = size(B)
    T = eltype(first(A))

    I = length(A)
    J = length(first(C))

    mat = ZMatrix2{T}(A, B, C, γ, δ, nA1, nA2, nB1, nB2, I, J, W1, W2)
    size_check(mat)
    return mat
end

function Base.size(M::ZMatrix2)
    return (M.nA1*M.nB2, M.nA2*M.nB1)
end

function LinearAlgebra.isdiag(M::ZMatrix2)
    return all((isdiag(Ai) for Ai in M.A)) && isdiag(M.B) && all((isdiag(Cij) for Ci in M.C for Cij in Ci))
end

function assemble_diag(M_ass::Diagonal, M::ZMatrix2, α::Number)
    @assert isdiag(M)
    # and square
    @assert M.nA1 == M.nA2
    @assert M.nB1 == M.nB2

    for i in 1:M.I
        cache_W2!(M.W2, M, i)
        if i == 1
            mul!(reshape(M_ass.diag, (M.nA1, M.nB1)), reshape(M.A[i].diag, (M.nA1, 1)), reshape(M.W2.diag, (1, M.nB1)), α, false)
        else
            mul!(reshape(M_ass.diag, (M.nA1, M.nB1)), reshape(M.A[i].diag, (M.nA1, 1)), reshape(M.W2.diag, (1, M.nB1)), α, true)
        end
    end
    return nothing
end

function assemble_diag(M_ass::Diagonal, M::Diagonal, α::Number)
    M_ass.diag .= α .* M.diag
    return nothing
end

function size_check(M::ZMatrix2)
    @assert length(M.A) == M.I
    @assert length(M.C) == M.I
    @assert all((length(Ci) == M.J) for Ci in M.C)
    @assert length(M.γ) == M.I
    @assert length(M.δ) == M.I
    @assert all((length(δi) == M.J for δi in M.δ))

    @assert all((size(Ai) == (M.nA1, M.nA2) for Ai in M.A))
    @assert size(M.B) == (M.nB1, M.nB2)
    @assert all((size(Cij) == (M.nB1, M.nB2) for Ci in M.C for Cij in Ci))

    @assert size(M.W1) == (M.nA1, M.nB1)
    @assert size(M.W2) == (M.nB1, M.nB2)
end

function cache_W2!(W2::AbstractArray, M::ZMatrix2, i)
    axpby!(M.γ[i], M.B, false, W2)
    for j in 1:M.J
        axpy!(M.δ[i][j], M.C[i][j], W2)
    end
end

function cache_W2!(W2::Diagonal, M::ZMatrix2, i)
    axpby!(M.γ[i], M.B.diag, false, W2.diag)
    for j in 1:M.J
        axpy!(M.δ[i][j], M.C[i][j].diag, W2.diag)
    end
end

function LinearAlgebra.mul!(y::AbstractVector, M::ZMatrix2, x::AbstractVector, α::Number, β::Number)
    X = reshape(x, M.nA2, M.nB1)
    Y = reshape(y, M.nA1, M.nB2)

    for i in 1:M.I
        mul!(M.W1, M.A[i], X)

        cache_W2!(M.W2, M, i)

        if i == 1
            mul!(Y, M.W1, M.W2, α, β)
        else
            mul!(Y, M.W1, M.W2, α, true)
        end
    end
    return nothing
end

function LinearAlgebra.transpose(M::ZMatrix2{T}) where T
    return Transpose{T, typeof(M)}(M)
end

function LinearAlgebra.mul!(y::AbstractVector, MT::Transpose{<:Any, <:ZMatrix2}, x::AbstractVector, α::Number, β::Number)
    M = MT.parent
    X = reshape(x, M.nA1, M.nB2)
    Y = reshape(y, M.nA2, M.nB1)

    for i in 1:M.I

        cache_W2!(M.W2, M, i)

        mul!(M.W1, X, transpose(M.W2))

        if i == 1
            mul!(Y, transpose(M.A[i]), M.W1, α, β)
        else
            mul!(Y, transpose(M.A[i]), M.W1, α, true)
        end
    end
    return nothing
end


"""
computes
    Y = α*(∑I A_i * X * B_i) + β*Y
"""
@concrete struct DMatrix2{T}
    A # AbstractVector (size I) of matrices of size nA1 x nA2
    B # AbstractVector (size I) of matrices of size nB1 x nB2

    # matrix sizes
    nA1
    nA2
    nB1
    nB2

    # iteration sizes
    I

    W1 # CACHE: matrix of size nA1 x nB1
end

function Base.size(M::DMatrix2)
    return (M.nA1*M.nB2, M.nA2*M.nB1)
end

function LinearAlgebra.isdiag(M::DMatrix2)
    return all((isdiag(Ai) for Ai in M.A)) && all((isdiag(Bi) for Bi in M.B))
end

function assemble_diag(M_ass::Diagonal, M::DMatrix2, α::Number)
    @assert isdiag(M)
    # and square
    @assert M.nA1 == M.nA2
    @assert M.nB1 == M.nB2

    for i in 1:M.I
        if i == 1
            mul!(reshape(M_ass.diag, (M.nA1, M.nB1)), reshape(M.A[i].diag, (M.nA1, 1)), reshape(M.B[i].diag, (1, M.nB1)), α, false)
        else
            mul!(reshape(M_ass.diag, (M.nA1, M.nB1)), reshape(M.A[i].diag, (M.nA1, 1)), reshape(M.B[i].diag, (1, M.nB1)), α, true)
        end
    end
    return nothing
end

function LinearAlgebra.mul!(y::AbstractVector, M::DMatrix2, x::AbstractVector, α::Number, β::Number)
    X = reshape(x, M.nA2, M.nB1)
    Y = reshape(y, M.nA1, M.nB2)

    for i in 1:M.I
        mul!(M.W1, M.A[i], X)
        if i == 1
            mul!(Y, M.W1, M.B[i], α, β)
        else
            mul!(Y, M.W1, M.B[i], α, true)
        end
    end
    return nothing
end

function LinearAlgebra.transpose(M::DMatrix2{T}) where T
    return Transpose{T, typeof(M)}(M)
end

function LinearAlgebra.mul!(y::AbstractVector, MT::Transpose{<:Any, <:DMatrix2}, x::AbstractVector, α::Number, β::Number)
    M = MT.parent
    X = reshape(x, M.nA1, M.nB2)
    Y = reshape(y, M.nA2, M.nB1)

    for i in 1:M.I
        mul!(M.W1, X, transpose(M.B[i]))
        if i == 1
            mul!(Y, transpose(M.A[i]), M.W1, α, β)
        else
            mul!(Y, transpose(M.A[i]), M.W1, α, true)
        end
    end
    return nothing
end

"""
computes
    Y1 = α*Δ( [A + γ*D]*X1        + [δ*B]*X2 ) + β*Y1
    Y2 = α*Δ( [δT * transp(B)]*X1 + [C]*X2   ) + β*Y2

if `sym`
    Y2 =-α*Δ( [δT * transp(B)]*X1 + [C]*X2   ) + β*Y2
"""
@concrete struct BlockMat2{T}
    A # matrix of size n1 x n1
    B # matrix of size n1 x n2
    C # matrix of size n2 x n2
    D # matrix of size n1 x n1

    # matrix sizes
    n1
    n2

    # coefficients (Ref values!)
    γ
    δ
    δT

    # scales the whole matrix (Ref value!)
    Δ

    # symmetrizes the lower part of the matrix (Ref value!)
    sym
end

block_size(M::BlockMat2) = (M.n1, M.n2)
block_size(MT::Transpose{<:Any, <:BlockMat2}) = (MT.parent.n1, MT.parent.n2)

Base.size(M::BlockMat2) = (M.n1+M.n2, M.n1+M.n2)
Base.size(M::BlockMat2, i) = size(M)[i]
Base.eltype(::BlockMat2{T}) where T = T

# multiplication routines for the individual blocks
function mul_ul!(y1::AbstractVector, M::BlockMat2, x1::AbstractVector, α::Number, β::Number)
    mul!(y1, M.A, x1, M.Δ[]*α, β)
    mul!(y1, M.D, x1, M.Δ[]*M.γ[]*α, true)
end

function mul_ur!(y1::AbstractVector, M::BlockMat2, x2::AbstractVector, α::Number, β::Number)
    mul!(y1, M.B, x2, M.Δ[]*M.δ[]*α, β)
end

function mul_ll!(y2::AbstractVector, M::BlockMat2, x1::AbstractVector, α::Number, β::Number)
    s = M.sym[] ? -1 : 1
    mul!(y2, transpose(M.B), x1, s*M.Δ[]*M.δT[]*α, β)
end

function mul_lr!(y2::AbstractVector, M::BlockMat2, x2::AbstractVector, α::Number, β::Number)
    s = M.sym[] ? -1 : 1
    mul!(y2, M.C, x2, s*M.Δ[]*α, β)
end

# multiplication routines for the respective transposes
function mul_ul!(y1::AbstractVector, MT::Transpose{<:Any, <:BlockMat2}, x1::AbstractVector, α::Number, β::Number)
    M = MT.parent
    mul!(y1, transpose(M.A), x1, M.Δ[]*α, β)
    mul!(y1, transpose(M.D), x1, M.Δ[]*M.γ[]*α, true)
end

function mul_ur!(y1::AbstractVector, MT::Transpose{<:Any, <:BlockMat2}, x2::AbstractVector, α::Number, β::Number)
    M = MT.parent

    s = M.sym[] ? -1 : 1
    mul!(y1, M.B, x2, s*M.Δ[]*M.δT[]*α, β)
end

function mul_ll!(y2::AbstractVector, MT::Transpose{<:Any, <:BlockMat2}, x1::AbstractVector, α::Number, β::Number)
    M = MT.parent

    mul!(y2, transpose(M.B), x1, M.Δ[]*M.δ[]*α, β)
end

function mul_lr!(y2::AbstractVector, MT::Transpose{<:Any, <:BlockMat2}, x2::AbstractVector, α::Number, β::Number)
    M = MT.parent

    s = M.sym[] ? -1 : 1
    mul!(y2, transpose(M.C), x2, s*M.Δ[]*α, β)
end

function LinearAlgebra.mul!(y::AbstractVector, M::BlockMat2, x::AbstractVector, α::Number, β::Number)
    n1, n2 = block_size(M)

    x1 = @view(x[1:n1])
    x2 = @view(x[n1+1:n1+n2])

    y1 = @view(y[1:n1])
    y2 = @view(y[n1+1:n1+n2])

    mul_ul!(y1, M, x1, α, β)
    mul_ur!(y1, M, x2, α, true)

    mul_lr!(y2, M, x2, α, β)
    mul_ll!(y2, M, x1, α, true)
    return nothing
end

function LinearAlgebra.transpose(M::BlockMat2{T}) where T
    return Transpose{T, typeof(M)}(M)
end

function LinearAlgebra.mul!(y::AbstractVector, MT::Transpose{<:Any, <:BlockMat2}, x::AbstractVector, α::Number, β::Number)
    n1, n2 = block_size(MT)

    x1 = @view(x[1:n1])
    x2 = @view(x[n1+1:n1+n2])

    y1 = @view(y[1:n1])
    y2 = @view(y[n1+1:n1+n2])

    mul_ul!(y1, MT, x1, α, β)
    mul_ur!(y1, MT, x2, α, true)

    mul_lr!(y2, MT, x2, α, β)
    mul_ll!(y2, MT, x1, α, true)
    return nothing
end

@concrete struct SchurBlockMat2{T}
    M # the full matrix this schur matrix is derived from: BlockMat2
    C_ass # the assembled cache of the lower right block (Diagonal n1 x n1)

    # cache 
    w1 # vector of size n1
end

Base.size(N::SchurBlockMat2) = (block_size(N.M)[1], block_size(N.M)[1])
Base.size(N::SchurBlockMat2, i) = size(N)[i]
Base.eltype(::SchurBlockMat2{T}) where T = T

function LinearAlgebra.transpose(N::SchurBlockMat2{T}) where T
    return Transpose{T, typeof(N)}(N)
end

function update_cache!(N::SchurBlockMat2)
    # C is diagonal, hence
    M = N.M isa Transpose ? N.M.parent : N.M
    s = M.sym[] ? -1 : 1
    assemble_diag(N.C_ass, M.C, s*M.Δ[])
end

function LinearAlgebra.mul!(y::AbstractVector, N::SchurBlockMat2, x::AbstractVector, α::Number, β::Number)
    M = N.M

    mul_ll!(N.w1, M, x, true, false)
    ldiv!(N.C_ass, N.w1)
    mul_ur!(y, M, N.w1, -α, β)

    mul_ul!(y, M, x, α, true)
end

function schur_rhs!(b_schur, b, N::SchurBlockMat2)
    M = N.M
    n1, n2 = block_size(M)

    b1 = @view(b[1:n1])
    b2 = @view(b[n1+1:n1+n2])
    b_schur .= b1

    ldiv!(N.w1, N.C_ass, b2)
    mul_ur!(b_schur, M, N.w1, -1, true)
    return nothing
end

function schur_sol!(x, b, N::SchurBlockMat2)
    M = N.M
    n1, n2 = block_size(M)

    b2 = @view(b[n1+1:n1+n2])

    x1 = @view(x[1:n1])
    x2 = @view(x[n1+1:n1+n2])

    x2 .= b2
    mul_ll!(x2, M, x1, -1, true)
    ldiv!(N.C_ass, x2)
    return nothing
end
