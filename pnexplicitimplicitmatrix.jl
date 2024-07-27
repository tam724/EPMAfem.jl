

struct PNExplicitImplicitMatrix{T, V<:AbstractVector{T}, Tpnsemi<:PNSemidiscretization{T, V}} <: AbstractMatrix{T}
    pn_semi::Tpnsemi
    α::Vector{T}
    γ::Vector{Vector{T}}
    β::Vector{T}

    tmp::V
    tmp2::V
end

function pn_explicitimplicitmatrix(pn_semi::PNSemidiscretization{T, V}) where {T, V<:AbstractVector{T}}
    ((nLp, nLm), (nRp, nRm)) = pn_semi.size

    return PNExplicitImplicitMatrix(
        pn_semi,

        ones(T, number_of_elements(equations(pn_semi))),
        [ones(T, number_of_scatterings(equations(pn_semi))) for _ in 1:number_of_elements(equations(pn_semi))], 
        ones(T, 1),

        V(undef, max(nLp, nLm)*max(nRp, nRm)),
        V(undef, max(nRp, nRm))
    )
end

function Base.size(A::PNExplicitImplicitMatrix, i)
    i < 1 && error("dimension index out of range")
    if i == 1 || i == 2
        ((nLp, nLm), (nRp, nRm)) = A.pn_semi.size
        return nLp*nRp + nLm*nRm
    else
        return 1
    end
end
Base.size(A::PNExplicitImplicitMatrix) = (size(A, 1), size(A, 2))
Base.eltype(::PNExplicitImplicitMatrix{T}) where T = T
# Base.getindex(A::PNExplicitImplicitMatrix{T}, i::Int) where T = T(1.0)
function Base.getindex(A::PNExplicitImplicitMatrix{T}, I::Vararg{Int, 2}) where T
    return NaN
end

## Block _mul routines. All routines implement += on the return (Cp or Cm) scaled by alpha
## e.g. running _mul_pp! results in Cp = α * App * Bp + Cp
## All routines override the temporary variables (the tmps are only valid during the method run. Forget about them afterwards!)

# linker oberer block
function _mul_pp!(Cp, A::PNExplicitImplicitMatrix, Bp, α)
    # create views into the temporary variables
    ((nLp, nLm), (nRp, nRm)) = A.pn_semi.size
    tmp_pp = reshape(@view(A.tmp[1:nLp*nRp]), (nLp, nRp))
    tmp2_p = @view(A.tmp2[1:nRp])

    for (ρpz, kpz, αz, γz) in zip(A.pn_semi.ρp, A.pn_semi.kp, A.α, A.γ)
        bα = αz*α != 0
        bγ = any(i -> i*α != 0, γz)
        if bα || bγ
            mul!(tmp_pp, ρpz, Bp)
            # first compute the sum of the kps and I (into tmp2_p)
            fill!(tmp2_p, αz)
            if bγ
                # here we already use that kp is diagonal
                for (kpzi, γzi) in zip(kpz, γz)
                    axpy!(γzi, kpzi.diag, tmp2_p)
                end
            end
            mul!(Cp, tmp_pp, Diagonal(tmp2_p), α, true)
        end
    end

    for (∂pd, absΩpd) in zip(A.pn_semi.∂p, A.pn_semi.absΩp)
        bβ = A.β[1]*α != 0
        if bβ
            mul!(tmp_pp, ∂pd, Bp)
            # use βp
            mul!(Cp, tmp_pp, absΩpd, A.β[1]*α, true)
        end
    end
end

# rechter unterer block
function _mul_mm!(Cm, A::PNExplicitImplicitMatrix, Bm, α)
    ((nLp, nLm), (nRp, nRm)) = A.pn_semi.size
    tmp_mm = reshape(@view(A.tmp[1:nLm*nRm]), (nLm, nRm))
    tmp2_m = @view(A.tmp2[1:nRm])

    for (ρmz, kmz, αz, γz) in zip(A.pn_semi.ρm, A.pn_semi.km, A.α, A.γ)
        bα = αz*α != 0
        bγ = any(i -> i*α != 0, γz)
        if bα || bγ
            mul!(tmp_mm, ρmz, Bm)
            fill!(tmp2_m, -αz)
            if bγ
                # here we already use that km is diagonal
                for (kmzi, γzi) in zip(kmz, γz)
                    axpy!(-γzi, kmzi.diag, tmp2_m)
                end
            end
            mul!(Cm, tmp_mm, Diagonal(tmp2_m), α, true)
        end
    end
end

#rechter oberer block
function _mul_mp!(Cp, A::PNExplicitImplicitMatrix, Bm, α)
    ((nLp, nLm), (nRp, nRm)) = A.pn_semi.size
    tmp_pm = reshape(@view(A.tmp[1:nLp*nRm]), (nLp, nRm))

    for (∇pmd, Ωpmd) in zip(A.pn_semi.∇pm, A.pn_semi.Ωpm)
        bβ = A.β[1]*α != 0
        if bβ
            mul!(tmp_pm, transpose(∇pmd), Bm)
            # use βpm
            mul!(Cp, tmp_pm, Ωpmd, A.β[1]*α, true)
        end
    end
end

#linker unterer block
function _mul_pm!(Cm, A::PNExplicitImplicitMatrix, Bp, α)
    ((nLp, nLm), (nRp, nRm)) = A.pn_semi.size
    tmp_mp = reshape(@view(A.tmp[1:nLm*nRp]), (nLm, nRp))

    for (∇pmd, Ωpmd) in zip(A.pn_semi.∇pm, A.pn_semi.Ωpm)
        bβ = A.β[1]*α != 0
        if bβ
            mul!(tmp_mp, ∇pmd, Bp)
            # use βmp
            mul!(Cm, tmp_mp, transpose(Ωpmd), A.β[1]*α, true)
        end
    end
end

import LinearAlgebra: mul!
function mul!(C::AbstractVector, A::PNExplicitImplicitMatrix, B::AbstractVector, α::Number, β::Number)
    ((nLp, nLm), (nRp, nRm)) = A.pn_semi.size
    np = nLp*nRp
    nm = nLm*nRm

    rmul!(C, β)

    Bp = reshape(@view(B[1:np]), (nLp, nRp))
    Bm = reshape(@view(B[np+1:np+nm]), (nLm, nRm))

    Cp = reshape(@view(C[1:np]), (nLp, nRp))
    Cm = reshape(@view(C[np+1:np+nm]), (nLm, nRm))

    _mul_pp!(Cp, A, Bp, α)
    _mul_mm!(Cm, A, Bm, α)
    _mul_mp!(Cp, A, Bm, α)
    _mul_pm!(Cm, A, Bp, α)
end

function mul_only!(C::AbstractVector, A::PNExplicitImplicitMatrix, B::AbstractVector, α::Number, β::Number, onlypm)
    ((nLp, nLm), (nRp, nRm)) = A.pn_semi.size
    np = nLp*nRp
    nm = nLm*nRm

    rmul!(C, β)

    Bp = reshape(@view(B[1:np]), (nLp, nRp))
    Bm = reshape(@view(B[np+1:np+nm]), (nLm, nRm))

    Cp = reshape(@view(C[1:np]), (nLp, nRp))
    Cm = reshape(@view(C[np+1:np+nm]), (nLm, nRm))

    if onlypm == :p
        _mul_pp!(Cp, A, Bp, α)
        _mul_mp!(Cp, A, Bm, α)
    end
    if onlypm == :m
        _mul_mm!(Cm, A, Bm, α)
        _mul_pm!(Cm, A, Bp, α)
    end
end

## SCHUR STUFF
struct PNSchurImplicitMatrix{T, V<:AbstractVector{T}, TA<:PNExplicitImplicitMatrix{T, V}} <: AbstractMatrix{T}
    A::TA
    D::V
    tmp::V
end

function Base.getindex(A_schur::PNSchurImplicitMatrix{T}, I::Vararg{Int, 2}) where T
    return NaN
end

function pn_schurimplicitmatrix(A::PNExplicitImplicitMatrix{T, V}) where {T, V}
    ((nLp, nLm), (nRp, nRm)) = A.pn_semi.size
    return PNSchurImplicitMatrix(
        A,
        V(undef, nLm*nRm),
        V(undef, nLm*nRm)
    )
end

function Base.size(A_schur::PNSchurImplicitMatrix, i)
    i < 1 && error("dimension index out of range")
    if i == 1 || i == 2
        ((nLp, _), (nRp, _)) = A_schur.A.pn_semi.size
        return nLp*nRp
    else
        return 1
    end
end
Base.size(A_schur::PNSchurImplicitMatrix) = (size(A_schur, 1), size(A_schur, 2))

function _update_D(A_schur::PNSchurImplicitMatrix{T}) where T
    # assemble D in A.D
    ((_, nLm), (_, nRm)) = A_schur.A.pn_semi.size
    tmp_m = @view(A_schur.A.tmp[1:nLm*nRm])
    tmp2_m = @view(A_schur.A.tmp2[1:nRm])


    fill!(A_schur.D, zero(T))
    for (ρmz, kmz, αz, γz) in zip(A_schur.A.pn_semi.ρm, A_schur.A.pn_semi.km, A_schur.A.α, A_schur.A.γ)
        bα = αz != 0
        bγ = any(i -> i != 0, γz)
        if bα || bγ
            fill!(tmp2_m, -αz)
            if bγ
                for (kmzi, γzi) in zip(kmz, γz)
                    axpy!(-γzi, kmzi.diag, tmp2_m)
                end
            end

            a = reshape(@view(tmp_m[:]), (nLm, nRm))
            b = reshape(@view(tmp2_m[:]), (1, nRm))
            c = reshape(@view(ρmz.diag[:]), (nLm, 1)) # without the view this allocates.. 
            # @show size.((a, b, c))
            mul!(a, c, b, true, false)
            axpy!(1.0, tmp_m, A_schur.D)
        end
    end
end

function mul!(C::AbstractVector, A::PNSchurImplicitMatrix, B::AbstractVector, α::Number, β::Number)
    ((nLp, nLm), (nRp, nRm)) = A.A.pn_semi.size

    Bp = reshape(@view(B[:]), (nLp, nRp))
    Cp = reshape(@view(C[:]), (nLp, nRp))

    rmul!(Cp, β)

    # tmp_pp = reshape(@view(A.A.tmp[1:nLp*nRp]), (nLp, nRp))
    # tmp_mp = reshape(@view(A.A.tmp[1:nLm*nRp]), (nLm, nRp))
    # tmp_pm = reshape(@view(A.A.tmp[1:nLp*nRm]), (nLp, nRm))
    # tmp_mm = reshape(@view(A.A.tmp[1:nLm*nRm]), (nLm, nRm))
    # tmp2_p = @view(A.A.tmp2[1:nRp])

    A_tmp_m = reshape(@view(A.tmp[1:nLm*nRm]), (nLm, nRm))

    fill!(A_tmp_m, zero(eltype(A_tmp_m)))
    _mul_pm!(A_tmp_m, A.A, Bp, 1.0)
    @view(A.tmp[1:nLm*nRm]) ./= A.D
    _mul_mp!(Cp, A.A, A_tmp_m, -α)
    _mul_pp!(Cp, A.A, Bp, α)
end

## EXPLICIT STUFF
struct PNExplicitMatrixP{T, V<:AbstractVector{T}, TA<:PNExplicitImplicitMatrix{T, V}} <: AbstractMatrix{T}
    A::TA
    b::V
end

function Base.getindex(A_schur::PNExplicitMatrixP{T}, I::Vararg{Int, 2}) where T
    return NaN
end

function pn_pnexplicitmatrixp(A::PNExplicitImplicitMatrix{T, V}) where {T, V}
    ((nLp, nLm), (nRp, nRm)) = A.pn_semi.size

    return PNExplicitMatrixP(
        A,
        V(undef, nLp)
    )
end

function Base.size(A::PNExplicitMatrixP, i)
    i < 1 && error("dimension index out of range")
    if i == 1 || i == 2
        ((nLp, _), (_, _)) = A.A.pn_semi.size
        return nLp
    else
        return 1
    end
end
Base.size(A::PNExplicitMatrixP) = (size(A, 1), size(A, 2))

function mul!(cp::AbstractVector, A::PNExplicitMatrixP, bp::AbstractVector, α::Number, β::Number)
    rmul!(cp, β)

    for (ρpz, αz) in zip(A.A.pn_semi.ρp, A.A.α)
        mul!(cp, ρpz, bp, α*αz, true)
    end
end

# struct PNExplicitMatrixM{T, V<:AbstractVector{T}, TA<:PNExplicitImplicitMatrix{T, V}} <: AbstractMatrix{T}
#     A::TA
# end

# function Base.getindex(A_schur::PNExplicitMatrixM{T}, I::Vararg{Int, 2}) where T
#     return NaN
# end

# function pn_pnexplicitmatrixm(A::PNExplicitImplicitMatrix{T, V}) where {T, V}
#     return PNExplicitMatrixM(
#         A
#     )
# end

# function Base.size(A::PNExplicitMatrixM, i)
#     i < 1 && error("dimension index out of range")
#     if i == 1 || i == 2
#         ((_, nLm), (_, nRm)) = A.A.pn_semi.size
#         return nLm*nRm
#     else
#         return 1
#     end
# end
# Base.size(A::PNExplicitMatrixM) = (size(A, 1), size(A, 2))

# function mul!(C::AbstractVector, A::PNExplicitMatrixM, B::AbstractVector, α::Number, β::Number)
#     ((nLp, nLm), (nRp, nRm)) = A.A.pn_semi.size

#     Bm = reshape(@view(B[:]), (nLm, nRm))
#     Cm = reshape(@view(C[:]), (nLm, nRm))

#     rmul!(Cm, β)
    
#     _mul_mm!(Cm, A.A, Bm, α)
# end