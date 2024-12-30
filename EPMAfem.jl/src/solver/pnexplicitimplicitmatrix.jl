# import LinearAlgebra: mul!


## Naming Convention: lowercase: vector values, uppercase: matrix valued
##

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
    rmul!(y, β)

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
    rmul!(y, β)

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

    rmul!(y, β)
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

    rmul!(y, β)

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