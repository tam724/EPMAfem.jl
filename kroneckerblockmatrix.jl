using LinearAlgebra

"""
This matrix type implements the action of a matrix A when multiplied with a vector B
where A = ∑_i [Dpp_i ⊗ Epp_i   Dpm_i ⊗ Epm_i
               Dmp_i ⊗ Emp_i   Dmm_i ⊗ Emm_i]
"""
struct KroneckerBlockMatrix{N, Dpp_, Dpm_, Dmp_, Dmm_, Epp_, Epm_, Emp_, Emm_, T_}
    size::Tuple{Tuple{Int64, Int64}, Tuple{Int64, Int64}}
    D::Tuple{NTuple{N, Dpp_}, NTuple{N, Dpm_}, NTuple{N, Dmp_}, NTuple{N, Dmm_}}
    E::Tuple{NTuple{N, Epp_}, NTuple{N, Epm_}, NTuple{N, Emp_}, NTuple{N, Emm_}}
    tmp::T_
end

function KroneckerBlockMat((Dpp, Dpm, Dmp, Dmm), (Epp, Epm, Emp, Emm))
    ##TODO check size of arrays
    nDp, nDm = size(first(Dpm))
    nEp, nEm = size(first(Epm))

    return KroneckerBlockMatrix(((nDp, nDm), (nEp, nEm)), (Dpp, Dpm, Dmp, Dmm), (Epp, Epm, Emp, Emm), zeros(max(nEp, nEm)*max(nDp, nDm)))
end

import Base: size, eltype
function size(A::KroneckerBlockMatrix)
    ((nDp, nDm), (nEp, nEm)) = A.size
    return (nDp*nEp+nDm*nEm, nDp*nEp+nDm*nEm)
end

function size(A::KroneckerBlockMatrix, i)
    ((nDp, nDm), (nEp, nEm)) = A.size
    i<=0 && error("dimension index out of range")
    i<=2 && return nDp*nEp+nDm*nEm
    return 1
end

import LinearAlgebra: mul!
function mul!(C::AbstractVector, A::KroneckerBlockMatrix, B::AbstractVector, α::Number, β::Number)
    ((nDp, nDm), (nEp, nEm)) = A.size
    np = nDp*nEp
    nm = nDm*nEm
    
    rmul!(C, β)
    
    Bp = reshape(@view(B[1:np]), (nEp, nDp))
    Bm = reshape(@view(B[np+1:np+nm]), (nEm, nDm))

    Cp = reshape(@view(C[1:np]), (nEp, nDp))
    Cm = reshape(@view(C[np+1:np+nm]), (nEm, nDm))

    tmp_pp = reshape(@view(A.tmp[1:nEp*nDp]), (nEp, nDp))
    tmp_mp = reshape(@view(A.tmp[1:nEm*nDp]), (nEm, nDp))
    tmp_pm = reshape(@view(A.tmp[1:nEp*nDm]), (nEp, nDm))
    tmp_mm = reshape(@view(A.tmp[1:nEm*nDm]), (nEm, nDm))

    Dpp, Dpm, Dmp, Dmm = A.D
    Epp, Epm, Emp, Emm = A.E
    
    for (epp, dpp) in zip(Epp, Dpp)
        mul!(tmp_pp, Bp, transpose(dpp))
        mul!(Cp, epp, tmp_pp, α, true)
    end

    for (epm, dpm) in zip(Epm, Dpm)
        mul!(tmp_mp, Bm, transpose(dpm))
        mul!(Cp, epm, tmp_mp, α, true)
    end

    for (emp, dmp) in zip(Emp, Dmp)
        mul!(tmp_pm, Bp, transpose(dmp))
        mul!(Cm, emp, tmp_pm, α, true)
    end

    for (emm, dmm) in zip(Emm, Dmm)
        mul!(tmp_mm, Bm, transpose(dmm))
        mul!(Cm, emm, tmp_mm, α, true)
    end

    return C
end