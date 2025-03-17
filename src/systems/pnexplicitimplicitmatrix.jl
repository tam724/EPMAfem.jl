# import LinearAlgebra: mul!

# somehow try to unify the use of rmul (thats how I mainly use it here)
include("../redefine_rmul.jl")

## Naming Convention: lowercase: vector values, uppercase: matrix valued
##

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

function Base.size(M::ZMatrix2)
    return (M.nA1*M.nB2, M.nA2*M.nB1)
end

function LinearAlgebra.isdiag(M::ZMatrix2)
    return all((isdiag(Ai) for Ai in M.A)) && isdiag(M.B) && all((isdiag(Cij) for Ci in M.C for Cij in Ci))
end

function assemble_diag(M_ass::Diagonal, M::ZMatrix2)
    @assert isdiag(M)
    # and square
    @assert M.nA1 == M.nA2
    @assert M.nB1 == M.nB2

    for i in 1:M.I
        cache_W2!(M.W2, M, i)
        if i == 1
            mul!(reshape(M_ass.diag, (M.nA1, M.nB1)), reshape(M.A[i].diag, (M.nA1, 1)), reshape(M.W2.diag, (1, M.nB1)), true, false)
        else
            mul!(reshape(M_ass.diag, (M.nA1, M.nB1)), reshape(M.A[i].diag, (M.nA1, 1)), reshape(M.W2.diag, (1, M.nB1)), true, true)
        end
    end
    return nothing
end

function assemble_diag(M_ass::Diagonal, M::Diagonal)
    M_ass.diag .= M.diag
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

    # W2 .= M.γ[i] .* M.B
    # for j in 1:M.J
    #     W2 .+= M.δ[i][j] .* M.C[i][j]
    # end
end

function cache_W2!(W2::Diagonal, M::ZMatrix2, i)
    axpby!(M.γ[i], M.B.diag, false, W2.diag)
    for j in 1:M.J
        axpy!(M.δ[i][j], M.C[i][j].diag, W2.diag)
    end
    
    # W2.diag .= M.γ[i] .* M.B.diag
    # for j in 1:M.J
    #     W2.diag .+= M.δ[i][j] .* M.C[i][j].diag
    # end
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

Base.size(M::BlockMat2) = (M.n1+M.n2, M.n1+M.n2)
Base.size(M::BlockMat2, i) = size(M)[i]
Base.eltype(::BlockMat2{T}) where T = T

function LinearAlgebra.mul!(y::AbstractVector, M::BlockMat2, x::AbstractVector, α::Number, β::Number)
    x1 = @view(x[1:M.n1])
    x2 = @view(x[M.n1+1:M.n1+M.n2])

    y1 = @view(y[1:M.n1])
    y2 = @view(y[M.n1+1:M.n1+M.n2])

    # pp
    mul!(y1, M.A, x1, M.Δ[]*α, β)
    mul!(y1, M.D, x1, M.Δ[]*M.γ[]*α, true)

    # # mp
    mul!(y1, M.B, x2, M.Δ[]*M.δ[]*α, true)

    # *(-1) for the lower part of the matrix if symmetrized
    s = M.sym[] ? -1 : 1

    # mm
    mul!(y2, M.C, x2, s*M.Δ[]*α, β)

    # pm
    mul!(y2, transpose(M.B), x1, s*M.Δ[]*M.δT[]*α, true)
    return nothing
end

function LinearAlgebra.transpose(M::BlockMat2{T}) where T
    return Transpose{T, typeof(M)}(M)
end

function LinearAlgebra.mul!(y::AbstractVector, MT::Transpose{<:Any, <:BlockMat2}, x::AbstractVector, α::Number, β::Number)
    M = MT.parent

    x1 = @view(x[1:M.n1])
    x2 = @view(x[M.n1+1:M.n1+M.n2])

    y1 = @view(y[1:M.n1])
    y2 = @view(y[M.n1+1:M.n1+M.n2])

    # pp
    mul!(y1, transpose(M.A), x1, M.Δ[]*α, β)
    mul!(y1, transpose(M.D), x1, M.Δ[]*M.γ[]*α, true)

    # # mp
    mul!(y1, M.B, x2, M.Δ[]*M.δT[]*α, true)

    # *(-1) for the lower part of the matrix if symmetrized
    s = M.sym[] ? -1 : 1

    # mm
    mul!(y2, transpose(M.C), x2, s*M.Δ[]*α, β)

    # pm
    mul!(y2, transpose(M.B), x1, s*M.Δ[]*M.δ[]*α, true)
    return nothing
end

@concrete struct SchurBlockMat2{T}
    M # the full matrix this schur matrix is derived from: BlockMat2
    C_ass # the assembled cache of the lower right block (Diagonal n1 x n1)

    # cache 
    w1 # vector of size n1
end

Base.size(N::SchurBlockMat2) = (N.M.n1, N.M.n1)
Base.size(N::SchurBlockMat2, i) = size(N)[i]
Base.eltype(::SchurBlockMat2{T}) where T = T

function update_cache!(N::SchurBlockMat2)
    assemble_diag(N.C_ass, N.M.C)
end

function LinearAlgebra.mul!(y::AbstractVector, N::SchurBlockMat2, x::AbstractVector, α::Number, β::Number)
    M = N.M

    mul!(N.w1, transpose(M.B), x, M.δT[], false)
    ldiv!(N.C_ass, N.w1)
    mul!(y, M.B, N.w1, -M.Δ[]*M.δ[]*α, β)

    mul!(y, M.A, x, M.Δ[]*α, true)
    mul!(y, M.D, x, M.Δ[]*M.γ[]*α, true)
end

function schur_rhs!(b_schur, b, N::SchurBlockMat2)
    M = N.M

    b1 = @view(b[1:M.n1])
    b2 = @view(b[M.n1+1:M.n1+M.n2])
    b_schur .= b1
    ldiv!(N.w1, N.C_ass, b2)
    s = M.sym[] ? -1 : 1
    mul!(b_schur, M.B, N.w1, -s*M.δ[], true)
    return nothing
end

function schur_sol!(x, b, N::SchurBlockMat2)
    M = N.M

    b1 = @view(b[1:M.n1])
    b2 = @view(b[M.n1+1:M.n1+M.n2])

    x1 = @view(x[1:M.n1])
    x2 = @view(x[M.n1+1:M.n1+M.n2])

    s = M.sym[] ? -1 : 1
    x2 .= (s/M.Δ[]) .* b2
    mul!(x2, transpose(M.B), x1, -M.δT[], true)
    ldiv!(N.C_ass, x2)
    return nothing
end

########### LEGACY

@concrete struct ZMatrix
    ρ
    i
    k
    a
    c
    Tmp1
    Tmp2
end


"""
    computes Y = sum_z ρ[z] * X * (a[z] * i + sum_i c[z][i] k[z][i]) * α + β * Y
"""
function LinearAlgebra.mul!(y::AbstractVector, (; ρ, i, k, a, c, Tmp1, Tmp2)::ZMatrix{R, I, K, A, C, T1, T2}, x::AbstractVector, α::Number, β::Number) where {R, I<:Diagonal, K, A, C, T1, T2<:Diagonal}
    nL1, nL2 = size(first(ρ))
    nR1, nR2 = size(first(first(k)))
    my_rmul!(y, β)

    for (ρz, kz, az, cz) in zip(ρ, k, a, c)
        mul!(Tmp1, ρz, reshape(x, (nL2, nR1)), true, false)
        # first compute the sum of the kps and I (should be Diagonal)
        Tmp2.diag .= az .* i.diag
        for (kzi, czi) in zip(kz, cz)
            @assert kzi isa Diagonal
            axpy!(czi, kzi.diag, Tmp2.diag)
        end
        mul!(reshape(y, (nL1, nR2)), Tmp1, Tmp2, α, true)
    end
end

"""
    computes Y = sum_z ρ[z] * X * (a[z] * i + sum_i c[z][i] k[z][i]) * α + β * Y
"""
function LinearAlgebra.mul!(y::AbstractVector, (; ρ, i, k, a, c, Tmp1, Tmp2)::ZMatrix{R, I, K, A, C, T1, T2}, x::AbstractVector, α::Number, β::Number) where {R, I, K, A, C, T1, T2}
    nL1, nL2 = size(first(ρ))
    nR1, nR2 = size(first(first(k)))
    my_rmul!(y, β)

    for (ρz, kz, az, cz) in zip(ρ, k, a, c)
        mul!(Tmp1, ρz, reshape(x, (nL2, nR1)), true, false)
        # first compute the sum of the kps and I
        Tmp2 .= az .* i
        for (kzi, czi) in zip(kz, cz)
            axpy!(czi, kzi, Tmp2)
        end
        mul!(reshape(y, (nL1, nR2)), Tmp1, Tmp2, α, true)
    end
end

"""
    computes Y = sum_d l[d] * X  * r[d] * b * α + β * Y
"""
@concrete struct DMatrix
    l
    r
    b
    Tmp1
end

"""
    computes Y = sum_d l[d] * X  * r[d] * b * α + β * Y
"""
function LinearAlgebra.mul!(y::AbstractVector, (; l, r, b, Tmp1)::DMatrix{L, R, T1}, x::AbstractVector, α::Number, β::Number) where {L, R, T1}
    nL1, nL2 = size(first(l))
    nR1, nR2 = size(first(r))

    my_rmul!(y, β)
    for (ld, rd) in zip(l, r)
        mul!(Tmp1, ld, reshape(x, (nL2, nR1)), true, false)
        mul!(reshape(y, (nL1, nR2)), Tmp1, rd, b*α, true)
    end
end

# function mul1!(y::AbstractVector, (; l, r, b, Tmp1)::DMatrix{L, R, T1}, x::AbstractVector, α::Number, β::Number) where {L, R, T1}
#     nL1, nL2 = size(first(l))
#     nR1, nR2 = size(first(r))

#     rmul!(y, β)
#     for (ld, rd) in zip(l, r)
#         mul!(Tmp1, ld, reshape(x, (nL2, nR1)), true, false)
#         mul!(reshape(y, (nL1, nR2)), Tmp1, rd, b*α, true)
#     end
# end

# function mul2!(y::AbstractVector, (; l, r, b, Tmp1)::DMatrix{L, R, T1}, x::AbstractVector, α::Number, β::Number) where {L, R, T1}
#     nL1, nL2 = size(first(l))
#     nR1, nR2 = size(first(r))

#     rmul!(y, β)
#     for (ld, rd) in zip(l, r)
#         mul!(Tmp1, ld, reshape(x, (nL2, nR1)), true, false)
#         mul!(reshape(y, (nL1, nR2)), Tmp1, rd, b*α, true)
#     end
# end

""" Constructs a view into tmp with matrix size (nL, nR)"""
function mat_view(tmp, nL, nR)
    return reshape(@view(tmp[1:nL*nR]), (nL, nR))
end

@concrete struct FullBlockMat
    ρp
    ρm
    ∇pm
    ∂p
    Ip
    Im
    kp
    km
    Ωpm 
    absΩp
    a
    b
    c
    tmp
    tmp2
    sym
end

Base.eltype(M::FullBlockMat) = Base.eltype(first(M.ρp)) # maybe we should check more here or in the constructor of FullBlockMat

function block_size(FBM::FullBlockMat)
    nLp = size(first(FBM.ρp), 1)
    nLm = size(first(FBM.ρm), 1)
    #nLm, nLp = size(first(FBM.∇pm))
    nRp = size(FBM.Ip, 1)
    nRm = size(FBM.Im, 1)
    # nRm, nRp = size(first(FBM.Ωpm))
    return ((nLp, nLm), (nRp, nRm))
end

function Base.size(FBM::FullBlockMat)
    ((nLp, nLm), (nRp, nRm)) = block_size(FBM)
    n = nLp*nRp + nLm*nRm
    return (n, n)
end

function LinearAlgebra.mul!(y::AbstractVector, (;ρp, ρm, ∇pm, ∂p, Ip, Im, kp, km, Ωpm, absΩp, a, b, c, tmp, tmp2, sym)::FullBlockMat{Tρp, Tρm, T∇pm, T∂p, TIp, TIm}, x::AbstractVector, α::Number, β::Number) where {Tρp, Tρm, T∇pm, T∂p, TIp<:Diagonal, TIm<:Diagonal}
    nLp = size(first(ρp), 1)
    nLm = size(first(ρm), 1)
    #nLm, nLp = size(first(FBM.∇pm))
    nRp = size(Ip, 1)
    nRm = size(Im, 1)

    np = nLp*nRp
    nm = nLm*nRm

    # @show nLp, nLm, nRp, nRm

    (b1, b2, d) = b

    # the first writes into Yp and Ym have to use β, the others add to the previous
    # pp
    mul!(@view(y[1:np]), ZMatrix(ρp, Ip, kp, a, c, mat_view(tmp, nLp, nRp), Diagonal(@view(tmp2[1:nRp]))), @view(x[1:np]), α, β)
    mul!(@view(y[1:np]), DMatrix(∂p, absΩp, d, mat_view(tmp, nLp, nRp)), @view(x[1:np]), α, true)
    # mp
    mul!(@view(y[1:np]), DMatrix(∇pm, Ωpm, b1, mat_view(tmp, nLp, nRm)), @view(x[np+1:np+nm]), α, true)

    # *(-1) for the lower part of the matrix if symmetrized
    γ = sym ? -1 : 1

    # mm
    mul!(@view(y[np+1:np+nm]), ZMatrix(ρm, Im, km, a, c, mat_view(tmp, nLm, nRm), Diagonal(@view(tmp2[1:nRm]))), @view(x[np+1:np+nm]), γ*α, β)
    # pm
    mul!(@view(y[np+1:np+nm]), DMatrix((transpose(∇pmd) for ∇pmd in ∇pm), (transpose(Ωpmd) for Ωpmd in Ωpm), b2, mat_view(tmp, nLm, nRp)), @view(x[1:np]), γ*α, true)
end

function LinearAlgebra.mul!(y::AbstractVector, (;ρp, ρm, ∇pm, ∂p, Ip, Im, kp, km, Ωpm, absΩp, a, b, c, tmp, tmp2, sym)::FullBlockMat, x::AbstractVector, α::Number, β::Number)
    nLp = size(first(ρp), 1)
    nLm = size(first(ρm), 1)
    #nLm, nLp = size(first(FBM.∇pm))
    nRp = size(Ip, 1)
    nRm = size(Im, 1)

    np = nLp*nRp
    nm = nLm*nRm

    (b1, b2, d) = b

    # the first writes into Yp and Ym have to use β, the others add to the previous
    # pp
    mul!(@view(y[1:np]), ZMatrix(ρp, Ip, kp, a, c, mat_view(tmp, nLp, nRp), reshape(@view(tmp2[1:nRp*nRp]), (nRp, nRp))), @view(x[1:np]), α, β)
    mul!(@view(y[1:np]), DMatrix(∂p, absΩp, d, mat_view(tmp, nLp, nRp)), @view(x[1:np]), α, true)
    # mp
    mul!(@view(y[1:np]), DMatrix(∇pm, Ωpm, b1, mat_view(tmp, nLp, nRm)), @view(x[np+1:np+nm]), α, true)

    # *(-1) for the lower part of the matrix if symmetrized
    γ = sym ? -1 : 1
    # mm
    mul!(@view(y[np+1:np+nm]), ZMatrix(ρm, Im, km, a, c, mat_view(tmp, nLm, nRm), reshape(@view(tmp2[1:nRm*nRm]), (nRm, nRm))), @view(x[np+1:np+nm]), γ*α, β)
    # pm
    mul!(@view(y[np+1:np+nm]), DMatrix((transpose(∇pmd) for ∇pmd in ∇pm), (transpose(Ωpmd) for Ωpmd in Ωpm), b2, mat_view(tmp, nLm, nRp)), @view(x[1:np]), γ*α, true)
end

## SCHUR STUFF
@concrete struct SchurBlockMat
    ρp
    ∇pm
    ∂p
    Ip
    kp
    Ωpm
    absΩp
    D
    a
    b
    c
    tmp
    tmp2
    tmp3
end

Base.eltype(M::SchurBlockMat) = Base.eltype(first(M.ρp))

function LinearAlgebra.mul!(y::AbstractVector, (;ρp, ∇pm, ∂p, Ip, kp, Ωpm, absΩp, D, a, b, c, tmp, tmp2, tmp3)::SchurBlockMat, x::AbstractVector, α::Number, β::Number)
    nLp, nLm = size(first(∇pm))
    nRm, nRp = size(first(Ωpm))

    # Xp = reshape(@view(x[:]), (nLp, nRp))
    # Yp = reshape(@view(y[:]), (nLp, nRp))

    (b1, b2, d) = b

    my_rmul!(y, β)

    # this operator Cp = mp ∘ D-1 ∘ pm(Bp) is still quite sparse. maybe it is reasonable to construct it once before the krylov loop
    fill!(mat_view(tmp3, nLm, nRm), zero(eltype(tmp3)))
    # pm 
    mul!(tmp3, DMatrix((transpose(∇pmd) for ∇pmd in ∇pm), (transpose(Ωpmd) for Ωpmd in Ωpm), b2, mat_view(tmp, nLm, nRp)), x, true, false)
    # D^-1
    ldiv!(D, @view(tmp3[1:nLm*nRm]))
    # mp
    mul!(y, DMatrix(∇pm, Ωpm, b1, mat_view(tmp, nLp, nRm)), tmp3, -α, β)

    #pp
    mul!(y, ZMatrix(ρp, Ip, kp, a, c, mat_view(tmp, nLp, nRp), Diagonal(@view(tmp2[1:nRp]))), x, α, true)
    mul!(y, DMatrix(∂p, absΩp, d, mat_view(tmp, nLp, nRp)), x, α, true)
end

function block_size(SBM::SchurBlockMat)
    nLp, nLm = size(first(SBM.∇pm))
    nRm, nRp = size(first(SBM.Ωpm))
    return ((nLp, nLm), (nRp, nRm))
end

function Base.size(SBM::SchurBlockMat)
    ((nLp, _), (nRp, _)) = block_size(SBM)
    n = nLp*nRp
    return (n, n)
end

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


# ## EXPLICIT STUFF (this will be phased out.., legacy for the explicit solver)
# struct PNExplicitImplicitMatrix{T, V<:AbstractVector{T}, Tpnsemi<:PNSemidiscretization{T, V}} <: AbstractMatrix{T}
#     pn_semi::Tpnsemi
#     α::Vector{T}
#     γ::Vector{Vector{T}}
#     β::Vector{T}

#     tmp::V
#     tmp2::V
# end

# function pn_explicitimplicitmatrix(pn_semi::PNSemidiscretization{T, V}) where {T, V<:AbstractVector{T}}
#     ((nLp, nLm), (nRp, nRm)) = pn_semi.size

#     return PNExplicitImplicitMatrix(
#         pn_semi,

#         ones(T, number_of_elements(equations(pn_semi))),
#         [ones(T, number_of_scatterings(equations(pn_semi))) for _ in 1:number_of_elements(equations(pn_semi))], 
#         ones(T, 1),

#         V(undef, max(nLp, nLm)*max(nRp, nRm)),
#         V(undef, max(nRp, nRm))
#     )
# end

# function Base.size(A::PNExplicitImplicitMatrix, i)
#     i < 1 && error("dimension index out of range")
#     if i == 1 || i == 2
#         ((nLp, nLm), (nRp, nRm)) = A.pn_semi.size
#         return nLp*nRp + nLm*nRm
#     else
#         return 1
#     end
# end
# Base.size(A::PNExplicitImplicitMatrix) = (size(A, 1), size(A, 2))
# Base.eltype(::PNExplicitImplicitMatrix{T}) where T = T
# # Base.getindex(A::PNExplicitImplicitMatrix{T}, i::Int) where T = T(1.0)
# function Base.getindex(A::PNExplicitImplicitMatrix{T}, I::Vararg{Int, 2}) where T
#     return NaN
# end

# # linker oberer block
# function _mul_pp!(Cp, A::PNExplicitImplicitMatrix, Bp, α)
#     # create views into the temporary variables
#     ((nLp, nLm), (nRp, nRm)) = A.pn_semi.size
#     mul!(Cp, ZMatrix(A.pn_semi.ρp, A.pn_semi.kp, A.α, A.γ, mat_view(A.tmp, nLp, nRp), Diagonal(@view(A.tmp2[1:nRp]))), Bp, α, true)
#     mul!(Cp, DMatrix(A.pn_semi.∂p, A.pn_semi.absΩp, mat_view(A.tmp, nLp, nRp)), Bp, A.β[1]*α, true)
# end

# # rechter unterer block
# function _mul_mm!(Cm, A::PNExplicitImplicitMatrix, Bm, α)
#     ((nLp, nLm), (nRp, nRm)) = A.pn_semi.size
#     mul!(Cm, ZMatrix(A.pn_semi.ρm, A.pn_semi.km, (-α for α in A.α), ((-γ for γ in γz) for γz in A.γ), mat_view(A.tmp, nLm, nRm), Diagonal(@view(A.tmp2[1:nRm]))), Bm, α, true)
# end

# #rechter oberer block
# function _mul_mp!(Cp, A::PNExplicitImplicitMatrix, Bm, α)
#     ((nLp, nLm), (nRp, nRm)) = A.pn_semi.size

#     mul!(Cp, DMatrix((transpose(∇pmd) for ∇pmd in A.pn_semi.∇pm), A.pn_semi.Ωpm, mat_view(A.tmp, nLp, nRm)), Bm, A.β[1]*α, true)
# end

# #linker unterer block
# function _mul_pm!(Cm, A::PNExplicitImplicitMatrix, Bp, α)
#     ((nLp, nLm), (nRp, nRm)) = A.pn_semi.size

#     mul!(Cm, DMatrix(A.pn_semi.∇pm, (transpose(Ωpmd) for Ωpmd in A.pn_semi.Ωpm), mat_view(A.tmp, nLm, nRp)), Bp, A.β[1]*α, true)
# end

# function LinearAlgebra.mul!(C::AbstractVector, A::PNExplicitImplicitMatrix, B::AbstractVector, α::Number, β::Number)
#     ((nLp, nLm), (nRp, nRm)) = A.pn_semi.size
#     np = nLp*nRp
#     nm = nLm*nRm

#     rmul!(C, β)

#     Bp = reshape(@view(B[1:np]), (nLp, nRp))
#     Bm = reshape(@view(B[np+1:np+nm]), (nLm, nRm))

#     Cp = reshape(@view(C[1:np]), (nLp, nRp))
#     Cm = reshape(@view(C[np+1:np+nm]), (nLm, nRm))

#     # pp
#     mul!(Cp, ZMatrix(A.pn_semi.ρp, A.pn_semi.kp, A.α, A.γ, mat_view(A.tmp, nLp, nRp), Diagonal(@view(A.tmp2[1:nRp]))), Bp, α, true)
#     mul!(Cp, DMatrix(A.pn_semi.∂p, A.pn_semi.absΩp, mat_view(A.tmp, nLp, nRp)), Bp, A.β[1]*α, true)
#     # mp
#     mul!(Cp, DMatrix((transpose(∇pmd) for ∇pmd in A.pn_semi.∇pm), A.pn_semi.Ωpm, mat_view(A.tmp, nLp, nRm)), Bm, A.β[1]*α, true)

#     # mm (minus because we use negative basis functions for the direction, this makes the resulting matrices symmetric)
#     mul!(Cm, ZMatrix(A.pn_semi.ρm, A.pn_semi.km, (-α for α in A.α), ((-γ for γ in γz) for γz in A.γ), mat_view(A.tmp, nLm, nRm), Diagonal(@view(A.tmp2[1:nRm]))), Bm, α, true)
#     # pm
#     mul!(Cm, DMatrix(A.pn_semi.∇pm, (transpose(Ωpmd) for Ωpmd in A.pn_semi.Ωpm), mat_view(A.tmp, nLm, nRp)), Bp, A.β[1]*α, true)

# end

# function mul_only!(C::AbstractVector, A::PNExplicitImplicitMatrix, B::AbstractVector, α::Number, β::Number, onlypm)
#     ((nLp, nLm), (nRp, nRm)) = A.pn_semi.size
#     np = nLp*nRp
#     nm = nLm*nRm

#     rmul!(C, β)

#     Bp = reshape(@view(B[1:np]), (nLp, nRp))
#     Bm = reshape(@view(B[np+1:np+nm]), (nLm, nRm))

#     Cp = reshape(@view(C[1:np]), (nLp, nRp))
#     Cm = reshape(@view(C[np+1:np+nm]), (nLm, nRm))

#     if onlypm == :p
#         _mul_pp!(Cp, A, Bp, α)
#         _mul_mp!(Cp, A, Bm, α)
#     end
#     if onlypm == :m
#         _mul_mm!(Cm, A, Bm, α)
#         _mul_pm!(Cm, A, Bp, α)
#     end
# end

# struct PNExplicitMatrixP{T, V<:AbstractVector{T}, TA<:PNExplicitImplicitMatrix{T, V}} <: AbstractMatrix{T}
#     A::TA
#     b::V
# end

# function Base.getindex(A_schur::PNExplicitMatrixP{T}, I::Vararg{Int, 2}) where T
#     return NaN
# end

# function pn_pnexplicitmatrixp(A::PNExplicitImplicitMatrix{T, V}) where {T, V}
#     ((nLp, nLm), (nRp, nRm)) = A.pn_semi.size

#     return PNExplicitMatrixP(
#         A,
#         V(undef, nLp)
#     )
# end

# function Base.size(A::PNExplicitMatrixP, i)
#     i < 1 && error("dimension index out of range")
#     if i == 1 || i == 2
#         ((nLp, _), (_, _)) = A.A.pn_semi.size
#         return nLp
#     else
#         return 1
#     end
# end
# Base.size(A::PNExplicitMatrixP) = (size(A, 1), size(A, 2))

# function mul!(cp::AbstractVector, A::PNExplicitMatrixP, bp::AbstractVector, α::Number, β::Number)
#     rmul!(cp, β)

#     for (ρpz, αz) in zip(A.A.pn_semi.ρp, A.A.α)
#         mul!(cp, ρpz, bp, α*αz, true)
#     end
# end

# # struct PNExplicitMatrixM{T, V<:AbstractVector{T}, TA<:PNExplicitImplicitMatrix{T, V}} <: AbstractMatrix{T}
# #     A::TA
# # end

# # function Base.getindex(A_schur::PNExplicitMatrixM{T}, I::Vararg{Int, 2}) where T
# #     return NaN
# # end

# # function pn_pnexplicitmatrixm(A::PNExplicitImplicitMatrix{T, V}) where {T, V}
# #     return PNExplicitMatrixM(
# #         A
# #     )
# # end

# # function Base.size(A::PNExplicitMatrixM, i)
# #     i < 1 && error("dimension index out of range")
# #     if i == 1 || i == 2
# #         ((_, nLm), (_, nRm)) = A.A.pn_semi.size
# #         return nLm*nRm
# #     else
# #         return 1
# #     end
# # end
# # Base.size(A::PNExplicitMatrixM) = (size(A, 1), size(A, 2))

# # function mul!(C::AbstractVector, A::PNExplicitMatrixM, B::AbstractVector, α::Number, β::Number)
# #     ((nLp, nLm), (nRp, nRm)) = A.A.pn_semi.size

# #     Bm = reshape(@view(B[:]), (nLm, nRm))
# #     Cm = reshape(@view(C[:]), (nLm, nRm))

# #     rmul!(Cm, β)
    
# #     _mul_mm!(Cm, A.A, Bm, α)
# # end