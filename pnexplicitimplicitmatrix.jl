import LinearAlgebra: mul!

# for less code..
macro parametric_type(type_name, fields...)
    # Generate type parameters as symbols T1, T2, ...
    field_types = [Symbol("T$f") for f in fields]
    
    # Generate the field definitions with their corresponding type parameters
    field_defs = [:( $(esc(field))::$(esc(field_type)) ) for (field, field_type) in zip(fields, field_types)]

    quote
        struct $(esc(type_name)){$((esc(field_type) for field_type in field_types)...)}
            $(field_defs...)
        end
    end
end

## Naming Convention: lowercase: vector values, uppercase: matrix valued
##

@parametric_type ZMatrix ρ k a c Tmp1 Tmp2

"""
    computes Y = sum_z ρ[z] * X * (a[z] * I + sum_i c[z][i] k[z][i]) * α + β * Y
"""
function LinearAlgebra.mul!(y::AbstractVector, (; ρ, k, a, c, Tmp1, Tmp2)::ZMatrix{R, K, A, C, T1, T2}, x::AbstractVector, α::Number, β::Number) where {R, K, A, C, T1, T2<:Diagonal}
    nL1, nL2 = size(first(ρ))
    nR1, nR2 = size(first(first(k)))
    rmul!(y, β)

    for (ρz, kz, az, cz) in zip(ρ, k, a, c)
        mul!(Tmp1, ρz, reshape(x, (nL2, nR1)), true, false)
        # first compute the sum of the kps and I (should be Diagonal)
        fill!(Tmp2.diag, az)
        for (kzi, czi) in zip(kz, cz)
            @assert kzi isa Diagonal
            axpy!(czi, kzi.diag, Tmp2.diag)
        end
        mul!(reshape(y, (nL1, nR2)), Tmp1, Tmp2, α, true)
    end
end

"""
    computes Y = sum_z ρ[z] * X * (a[z] * I + sum_i c[z][i] k[z][i]) * α + β * Y
"""
function LinearAlgebra.mul!(y::AbstractVector, (; ρ, k, a, c, Tmp1, Tmp2)::ZMatrix{R, K, A, C, T1, T2}, x::AbstractVector, α::Number, β::Number) where {R, K, A, C, T1, T2}
    nL1, nL2 = size(first(ρ))
    nR1, nR2 = size(first(first(k)))
    rmul!(y, β)

    for (ρz, kz, az, cz) in zip(ρ, k, a, c)
        mul!(Tmp1, ρz, reshape(x, (nL2, nR1)), true, false)
        # first compute the sum of the kps and I (should be Diagonal)
        fill!(Tmp2, 0.0)
        Tmp2 .+= az*I # lets see if this works...
        for (kzi, czi) in zip(kz, cz)
            axpy!(czi, kzi, Tmp2)
        end
        mul!(reshape(y, (nL1, nR2)), Tmp1, Tmp2, α, true)
    end
end

@parametric_type DMatrix l r b Tmp1

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

""" Constructs a view into tmp with matrix size (nL, nR)"""
function mat_view(tmp, nL, nR)
    return reshape(@view(tmp[1:nL*nR]), (nL, nR))
end

@parametric_type FullBlockMat ρp ρm ∇pm ∂p kp km Ωpm absΩp a b c tmp tmp2

function block_size(FBM::FullBlockMat)
    nLm, nLp = size(first(FBM.∇pm))
    nRm, nRp = size(first(FBM.Ωpm))
    return ((nLp, nLm), (nRp, nRm))
end

function Base.size(FBM::FullBlockMat)
    ((nLp, nLm), (nRp, nRm)) = block_size(FBM)
    n = nLp*nRp + nLm*nRm
    return (n, n)
end

function LinearAlgebra.mul!(y::AbstractVector, (;ρp, ρm, ∇pm, ∂p, kp, km, Ωpm, absΩp, a, b, c, tmp, tmp2)::FullBlockMat, x::AbstractVector, α::Number, β::Number)
    nLm, nLp = size(first(∇pm))
    nRm, nRp = size(first(Ωpm))

    np = nLp*nRp
    nm = nLm*nRm

    # the first writes into Yp and Ym have to use β, the others add to the previous
    # pp
    mul!(@view(y[1:np]), ZMatrix(ρp, kp, a, c, mat_view(tmp, nLp, nRp), Diagonal(@view(tmp2[1:nRp]))), @view(x[1:np]), α, β)
    mul!(@view(y[1:np]), DMatrix(∂p, absΩp, b, mat_view(tmp, nLp, nRp)), @view(x[1:np]), α, true)
    # mp
    mul!(@view(y[1:np]), DMatrix((transpose(∇pmd) for ∇pmd in ∇pm), Ωpm, b, mat_view(tmp, nLp, nRm)), @view(x[np+1:np+nm]), α, true)

    # mm
    mul!(@view(y[np+1:np+nm]), ZMatrix(ρm, km, (-a_ for a_ in a), ((-c_ for c_ in cz) for cz in c), mat_view(tmp, nLm, nRm), Diagonal(@view(tmp2[1:nRm]))), @view(x[np+1:np+nm]), α, β)
    # pm
    mul!(@view(y[np+1:np+nm]), DMatrix(∇pm, (transpose(Ωpmd) for Ωpmd in Ωpm), b, mat_view(tmp, nLm, nRp)), @view(x[1:np]), α, true)
end

## SCHUR STUFF
@parametric_type SchurBlockMat ρp ∇pm ∂p kp Ωpm absΩp D a b c tmp tmp2 tmp3

function mul!(y::AbstractVector, (;ρp, ∇pm, ∂p, kp, Ωpm, absΩp, D, a, b, c, tmp, tmp2, tmp3)::SchurBlockMat, x::AbstractVector, α::Number, β::Number)
    nLm, nLp = size(first(∇pm))
    nRm, nRp = size(first(Ωpm))

    # Xp = reshape(@view(x[:]), (nLp, nRp))
    # Yp = reshape(@view(y[:]), (nLp, nRp))

    rmul!(y, β)

    
    # this operator Cp = mp ∘ D-1 ∘ pm(Bp) is still quite sparse. maybe it is reasonable to construct it once before the krylov loop
    fill!(mat_view(tmp3, nLm, nRm), zero(eltype(tmp3)))
    # pm 
    # _mul_pm!(A_tmp_m, A.A, Bp, 1.0)
    mul!(tmp3, DMatrix(∇pm, (transpose(Ωpmd) for Ωpmd in Ωpm), b, mat_view(tmp, nLm, nRp)), x, true, false)
    
    # D^-1
    # ldiv!(D, @view(tmp3[1:nLm*nRm]))
    # @view(tmp3[1:nLm*nRm]) ./= D
    ldiv!(D, @view(tmp3[1:nLm*nRm]))
    
    # mp
    # _mul_mp!(Cp, A.A, A_tmp_m, -α)
    mul!(y, DMatrix((transpose(∇pmd) for ∇pmd in ∇pm), Ωpm, b, mat_view(tmp, nLp, nRm)), tmp3, -α, β)


    # _mul_pp!(Cp, A.A, Bp, α)
    mul!(y, ZMatrix(ρp, kp, a, c, mat_view(tmp, nLp, nRp), Diagonal(@view(tmp2[1:nRp]))), x, α, true)
    mul!(y, DMatrix(∂p, absΩp, b, mat_view(tmp, nLp, nRp)), x, α, true)
end

function block_size(SBM::SchurBlockMat)
    nLm, nLp = size(first(SBM.∇pm))
    nRm, nRp = size(first(SBM.Ωpm))
    return ((nLp, nLm), (nRp, nRm))
end

function Base.size(SBM::SchurBlockMat)
    ((nLp, _), (nRp, _)) = block_size(SBM)
    n = nLp*nRp
    return (n, n)
end


## EXPLICIT STUFF (this will be phased out.., legacy for the explicit solver)
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

# linker oberer block
function _mul_pp!(Cp, A::PNExplicitImplicitMatrix, Bp, α)
    # create views into the temporary variables
    ((nLp, nLm), (nRp, nRm)) = A.pn_semi.size
    mul!(Cp, ZMatrix(A.pn_semi.ρp, A.pn_semi.kp, A.α, A.γ, mat_view(A.tmp, nLp, nRp), Diagonal(@view(A.tmp2[1:nRp]))), Bp, α, true)
    mul!(Cp, DMatrix(A.pn_semi.∂p, A.pn_semi.absΩp, mat_view(A.tmp, nLp, nRp)), Bp, A.β[1]*α, true)
end

# rechter unterer block
function _mul_mm!(Cm, A::PNExplicitImplicitMatrix, Bm, α)
    ((nLp, nLm), (nRp, nRm)) = A.pn_semi.size
    mul!(Cm, ZMatrix(A.pn_semi.ρm, A.pn_semi.km, (-α for α in A.α), ((-γ for γ in γz) for γz in A.γ), mat_view(A.tmp, nLm, nRm), Diagonal(@view(A.tmp2[1:nRm]))), Bm, α, true)
end

#rechter oberer block
function _mul_mp!(Cp, A::PNExplicitImplicitMatrix, Bm, α)
    ((nLp, nLm), (nRp, nRm)) = A.pn_semi.size

    mul!(Cp, DMatrix((transpose(∇pmd) for ∇pmd in A.pn_semi.∇pm), A.pn_semi.Ωpm, mat_view(A.tmp, nLp, nRm)), Bm, A.β[1]*α, true)
end

#linker unterer block
function _mul_pm!(Cm, A::PNExplicitImplicitMatrix, Bp, α)
    ((nLp, nLm), (nRp, nRm)) = A.pn_semi.size

    mul!(Cm, DMatrix(A.pn_semi.∇pm, (transpose(Ωpmd) for Ωpmd in A.pn_semi.Ωpm), mat_view(A.tmp, nLm, nRp)), Bp, A.β[1]*α, true)
end

function LinearAlgebra.mul!(C::AbstractVector, A::PNExplicitImplicitMatrix, B::AbstractVector, α::Number, β::Number)
    ((nLp, nLm), (nRp, nRm)) = A.pn_semi.size
    np = nLp*nRp
    nm = nLm*nRm

    rmul!(C, β)

    Bp = reshape(@view(B[1:np]), (nLp, nRp))
    Bm = reshape(@view(B[np+1:np+nm]), (nLm, nRm))

    Cp = reshape(@view(C[1:np]), (nLp, nRp))
    Cm = reshape(@view(C[np+1:np+nm]), (nLm, nRm))

    # pp
    mul!(Cp, ZMatrix(A.pn_semi.ρp, A.pn_semi.kp, A.α, A.γ, mat_view(A.tmp, nLp, nRp), Diagonal(@view(A.tmp2[1:nRp]))), Bp, α, true)
    mul!(Cp, DMatrix(A.pn_semi.∂p, A.pn_semi.absΩp, mat_view(A.tmp, nLp, nRp)), Bp, A.β[1]*α, true)
    # mp
    mul!(Cp, DMatrix((transpose(∇pmd) for ∇pmd in A.pn_semi.∇pm), A.pn_semi.Ωpm, mat_view(A.tmp, nLp, nRm)), Bm, A.β[1]*α, true)

    # mm
    mul!(Cm, ZMatrix(A.pn_semi.ρm, A.pn_semi.km, (-α for α in A.α), ((-γ for γ in γz) for γz in A.γ), mat_view(A.tmp, nLm, nRm), Diagonal(@view(A.tmp2[1:nRm]))), Bm, α, true)
    # pm
    mul!(Cm, DMatrix(A.pn_semi.∇pm, (transpose(Ωpmd) for Ωpmd in A.pn_semi.Ωpm), mat_view(A.tmp, nLm, nRp)), Bp, A.β[1]*α, true)

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