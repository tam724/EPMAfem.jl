using LinearAlgebra
using BenchmarkTools
using StaticArrays
using CUDA

struct PNMatrix{NE, ND, Ni, Tρp, Tρm, T∂p, T∇pm, TIp, TIm, Tkp, Tkm, TabsΩp, TΩpm, Ttmp, Ttmp2}
    size::Tuple{Tuple{Int64, Int64}, Tuple{Int64, Int64}}
    ρp::SVector{NE, Tρp}
    ρm::SVector{NE, Tρm}
    ∂p::SVector{ND, T∂p}
    ∇pm::SVector{ND, T∇pm}

    α::MVector{NE, Float64}
    Ip::TIp
    Im::TIm

    γ::SVector{NE, MVector{Ni, Float64}}
    kp::SVector{NE, SVector{Ni, Tkp}}
    km::SVector{NE, SVector{Ni, Tkm}}

    β::MVector{1, Float64}
    absΩp::SVector{ND, TabsΩp}
    Ωpm::SVector{ND, TΩpm}
    tmp::Ttmp
    tmp2::Ttmp2
end

function assemble(A::PNMatrix)
    A_assembled = zeros(size(A))
    e_i = zeros(size(A, 2))
    for i in 1:size(A, 2)
        e_i[i] = 1.0
        mul!(@view(A_assembled[i, :]), A, e_i)
        e_i[i] = 0.0
    end
    return A_assembled
end

function assemble_mm(A::PNMatrix)
    ((nLp, nLm), (nRp, nRm)) = A.size
    A_assembled = zeros(nLm*nRm, nLm*nRm)
    e_i = zeros(nLm*nRm)
    Bm = reshape(e_i, (nLm, nRm))
    tmp_mm = reshape(@view(A.tmp[1:nLm*nRm]), (nLm, nRm))
    tmp2_m = @view(A.tmp2[1:nRm])
    for i in 1:nLm*nRm
        e_i[i] = 1.0
        Cm = reshape(@view(A_assembled[i, :]), (nLm, nRm))
        _mul_mm!(Cm, A, Bm, 1.0, tmp_mm, tmp2_m)
        e_i[i] = 0.0
    end
    return A_assembled
end

import Base: size
function size(A::PNMatrix, i)
    i < 1 && error("dimension index out of range")
    if i == 1 || i == 2
        ((nLp, nLm), (nRp, nRm)) = A.size
        return nLp*nRp + nLm*nRm
    else
        return 1
    end
end

function size(A::PNMatrix)
    return (size(A, 1), size(A, 2))
end

import Base: eltype
function eltype(A::PNMatrix)
    return eltype(first(A.ρp))
end

function cuda(A::PNMatrix, T=Float32)
    return PNMatrix(
        A.size,
        CUDA.CUSPARSE.CuSparseMatrixCOO{T}.(A.ρp),
        SVector([Diagonal(CuVector{T}(ρmz.diag)) for ρmz ∈ A.ρm]),

        CUDA.CUSPARSE.CuSparseMatrixCOO{T}.(A.∂p),
        CUDA.CUSPARSE.CuSparseMatrixCOO{T}.(A.∇pm),
        A.α,
        Diagonal(CuVector{T}(A.Ip.diag)),
        Diagonal(CuVector{T}(A.Im.diag)),

        A.γ,
        SVector([SVector([Diagonal(CuVector{Float64}(kpzi.diag)) for kpzi ∈ kpz]) for kpz ∈ A.kp]),
        SVector([SVector([Diagonal(CuVector{Float64}(kmzi.diag)) for kmzi ∈ kmz]) for kmz ∈ A.km]),

        A.β, # βpp, βpm, βmp
        CuMatrix{T}.(A.absΩp),
        CuMatrix{T}.(A.Ωpm),
        CuVector{T}(A.tmp),
        CuVector{T}(A.tmp2)
        )
end

# linker oberer block
function _mul_pp!(Cp, A, Bp, α, tmp_pp, tmp2_p)
    for (ρpz, kpz, αz, γz) in zip(A.ρp, A.kp, A.α, A.γ)
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

    for (∂pd, absΩpd) in zip(A.∂p, A.absΩp)
        bβ = A.β[1]*α != 0
        if bβ
            mul!(tmp_pp, ∂pd, Bp)
            # use βp
            mul!(Cp, tmp_pp, absΩpd, A.β[1]*α, true)
        end
    end
end

# rechter unterer block
function _mul_mm!(Cm, A, Bm, α, tmp_mm, tmp2_m)
    for (ρmz, kmz, αz, γz) in zip(A.ρm, A.km, A.α, A.γ)
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
function _mul_mp!(Cp, A, Bm, α, tmp_pm)
    for (∇pmd, Ωpmd) in zip(A.∇pm, A.Ωpm)
        bβ = A.β[1]*α != 0
        if bβ
            mul!(tmp_pm, transpose(∇pmd), Bm)
            # use βpm
            mul!(Cp, tmp_pm, Ωpmd, A.β[1]*α, true)
        end
    end
end

#linker unterer block
function _mul_pm!(Cm, A, Bp, α, tmp_mp)
    for (∇pmd, Ωpmd) in zip(A.∇pm, A.Ωpm)
        bβ = A.β[1]*α != 0
        if bβ
            mul!(tmp_mp, ∇pmd, Bp)
            # use βmp
            mul!(Cm, tmp_mp, transpose(Ωpmd), A.β[1]*α, true)
        end
    end
end

import LinearAlgebra: mul!
function mul!(C::AbstractVector, A::PNMatrix, B::AbstractVector, α::Number, β::Number)
    ((nLp, nLm), (nRp, nRm)) = A.size

    np = nLp*nRp
    nm = nLm*nRm

    rmul!(C, β)

    Bp = reshape(@view(B[1:np]), (nLp, nRp))
    Bm = reshape(@view(B[np+1:np+nm]), (nLm, nRm))

    Cp = reshape(@view(C[1:np]), (nLp, nRp))
    Cm = reshape(@view(C[np+1:np+nm]), (nLm, nRm))

    tmp_pp = reshape(@view(A.tmp[1:nLp*nRp]), (nLp, nRp))
    tmp_mp = reshape(@view(A.tmp[1:nLm*nRp]), (nLm, nRp))
    tmp_pm = reshape(@view(A.tmp[1:nLp*nRm]), (nLp, nRm))
    tmp_mm = reshape(@view(A.tmp[1:nLm*nRm]), (nLm, nRm))

    tmp2_p = @view(A.tmp2[1:nRp])
    tmp2_m = @view(A.tmp2[1:nRm])

    _mul_pp!(Cp, A, Bp, α, tmp_pp, tmp2_p)
    _mul_mm!(Cm, A, Bm, α, tmp_mm, tmp2_m)
    _mul_mp!(Cp, A, Bm, α, tmp_pm)
    _mul_pm!(Cm, A, Bp, α, tmp_mp)
end

## PNProblem

struct PNProblem{PNM, Tb}
    A::PNM # PNMatrix{...}

    b::Tb
    btmp::Tb
end

function cuda(pn_prob, T=Float32)
    return PNProblem(
        cuda(pn_prob.A, T),
        CuVector{T}(pn_prob.b),
        CuVector{T}(pn_prob.btmp),
    )
end

function _update_b(pn_prob, α)
    pn_prob.b .= α .* pn_prob.btmp
end

##

struct SchurPNProblem{Tprob, TA, Tb, Tfs}
    prob::Tprob

    A::TA
    b::Tb

    full_solution::Tfs
end

function schur_complement(pn_prob::PNProblem)
    ((nLp, nLm), (nRp, nRm)) = pn_prob.A.size

    return SchurPNProblem(
        pn_prob,
        SchurPNMatrix(pn_prob.A, similar(pn_prob.b, nLm*nRm), similar(pn_prob.b, nLm*nRm)),
        similar(pn_prob.b, nLp*nRp),
        similar(pn_prob.b, nLp*nRp + nLm*nRm)
    )
end

struct SchurPNMatrix{TA, TD, Ttmp}
    A::TA
    D::TD
    tmp::Ttmp
end

function size(A_schur::SchurPNMatrix, i)
    i < 1 && error("dimension index out of range")
    if i == 1 || i == 2
        ((nLp, _), (nRp, _)) = A_schur.A.size
        return nLp*nRp
    else
        return 1
    end
end

function size(A_schur::SchurPNMatrix)
    return (size(A_schur, 1), size(A_schur, 2))
end

function eltype(A_schur::SchurPNMatrix)
    return eltype(first(A_schur.A.ρp))
end

# rechter unterer block (assembliert)
function _update_D(A::SchurPNMatrix)
    # assemble D in A.D
    ((_, nLm), (_, nRm)) = A.A.size
    tmp_m = @view(A.A.tmp[1:nLm*nRm])
    tmp2_m = @view(A.A.tmp2[1:nRm])


    fill!(A.D, zero(eltype(A.D)))
    for (ρmz, kmz, αz, γz) in zip(A.A.ρm, A.A.km, A.A.α, A.A.γ)
        bα = αz != 0
        bγ = any(i -> i != 0, γz)
        if bα || bγ
            fill!(tmp2_m, -αz)
            if bγ
                for (kmzi, γzi) in zip(kmz, γz)
                    axpy!(-γzi, kmzi.diag, tmp2_m)
                end
            end
            # a = reshape(@view(tmp_m[:]), (nLm*nRm, 1))
            # b = reshape(@view(tmp2_m[:]), (nRm, 1))
            # c = reshape(@view(ρmz.diag[:]), (nLm, 1)) # without the view this allocates.. 
            # kron!(a, b, c)

            a = reshape(@view(tmp_m[:]), (nLm, nRm))
            b = reshape(@view(tmp2_m[:]), (1, nRm))
            c = reshape(@view(ρmz.diag[:]), (nLm, 1)) # without the view this allocates.. 
            # @show size.((a, b, c))
            mul!(a, c, b, true, false)
            # CUDA.@allowscalar kron!(tmp_m, tmp2_m, ρmz.diag)

            axpy!(1.0, tmp_m, A.D)
        end
    end
end

function _compute_schur_rhs(rhs_schur, A_schur::SchurPNMatrix, rhs)
    ((nLp, nLm), (nRp, nRm)) = A_schur.A.size

    np = nLp*nRp
    nm = nLm*nRm

    tmp_pm = reshape(@view(A_schur.A.tmp[1:nLp*nRm]), (nLp, nRm))

    rhsp = reshape(@view(rhs[1:np]), (nLp, nRp))
    rhsm = reshape(@view(rhs[np+1:np+nm]), (nLm, nRm))

    rhs_schurp = reshape(@view(rhs_schur[:]), (nLp, nRp))

    A_tmp_m = reshape(@view(A_schur.tmp[1:nLm*nRm]), (nLm, nRm))

    rhs_schurp .= rhsp
    @view(A_schur.tmp[1:nLm*nRm]) .= @view(rhs[np+1:np+nm]) ./ A_schur.D
    _mul_mp!(rhs_schurp, A_schur.A, A_tmp_m, -1.0, tmp_pm)
end

function _compute_full_solution_schur(full_solution, A_schur::SchurPNMatrix, rhs, sol_schur)
    ((nLp, nLm), (nRp, nRm)) = A_schur.A.size

    np = nLp*nRp
    nm = nLm*nRm

    full_p = @view(full_solution[1:np])
    full_m = @view(full_solution[np+1:np+nm])
    full_mm = reshape(full_m, (nLm, nRm))

    rhsp = reshape(@view(rhs[1:np]), (nLp, nRp))
    rhsm = reshape(@view(rhs[np+1:np+nm]), (nLm, nRm))

    tmp_mp = reshape(@view(A_schur.A.tmp[1:nLm*nRp]), (nLm, nRp))

    full_p .= sol_schur
    
    full_m .= @view(rhs[np+1:np+nm])
    _mul_pm!(full_mm, A_schur.A, reshape(@view(sol_schur[:]), (nLp, nRp)), -1.0, tmp_mp)
    full_m .= full_m ./ A_schur.D
end

function mul!(C::AbstractVector, A::SchurPNMatrix, B::AbstractVector, α::Number, β::Number)
    ((nLp, nLm), (nRp, nRm)) = A.A.size

    Bp = reshape(@view(B[:]), (nLp, nRp))
    Cp = reshape(@view(C[:]), (nLp, nRp))

    rmul!(Cp, β)

    tmp_pp = reshape(@view(A.A.tmp[1:nLp*nRp]), (nLp, nRp))
    tmp_mp = reshape(@view(A.A.tmp[1:nLm*nRp]), (nLm, nRp))
    tmp_pm = reshape(@view(A.A.tmp[1:nLp*nRm]), (nLp, nRm))
    # tmp_mm = reshape(@view(A.A.tmp[1:nLm*nRm]), (nLm, nRm))
    tmp2_p = @view(A.A.tmp2[1:nRp])

    A_tmp_m = reshape(@view(A.tmp[1:nLm*nRm]), (nLm, nRm))

    fill!(A_tmp_m, zero(eltype(A_tmp_m)))
    _mul_pm!(A_tmp_m, A.A, Bp, 1.0, tmp_mp)
    @view(A.tmp[1:nLm*nRm]) .= @view(A.tmp[1:nLm*nRm]) ./ A.D
    _mul_mp!(Cp, A.A, A_tmp_m, -α, tmp_pm)
    _mul_pp!(Cp, A.A, Bp, α, tmp_pp, tmp2_p)
end


### testing
# begin
    # nxp = 2000
    # nxm = 2100
    # nop = 2200
    # nom = 2300
# 
#     mat = PNMatrix(
#         ((nxp, nxm), (nop, nom)), # size
#         @SVector[rand(nxp, nxp), rand(nxp, nxp)], # ρp
#         @SVector[rand(nxm, nxm), rand(nxm, nxm)], # ρm
#         @SVector[rand(nxp, nxp)], # ∂p
#         @SVector[rand(nxm, nxp)], # ∇pm

#         @MVector[1.0, 1.0], # α
#         Diagonal(ones(nop)), # Ip
#         Diagonal(ones(nom)), # Im

#         @SVector[@MVector[1.0, 1.0], @MVector[1.0, 1.0]], # γ
#         @SVector[@SVector[rand(nop, nop), rand(nop, nop)], @SVector[rand(nop, nop), rand(nop, nop)]], # kp
#         @SVector[@SVector[rand(nom, nom), rand(nom, nom)], @SVector[rand(nom, nom), rand(nom, nom)]], # km

#         @MVector[1.0, 1.0, 1.0], # βpp, βpm, βmp
#         @SVector[rand(nop, nop)],  # absΩp
#         @SVector[rand(nom, nop)],  # Ωpm
#         zeros(max(nxp, nxm)*max(nop, nom))  # tmp
#     )
#     mat_cu = cuda(mat)


#     C = zeros(nxp*nop+nxm*nom)
#     B = rand(nxp*nop+nxm*nom)
#     C_cu = cu(C)
#     B_cu = cu(B)

#     @benchmark mul!($(C), $(mat), $(B), $(1.0), $(0.0))
#     @benchmark mul!($(C_cu), $(mat_cu), $(B_cu), $(1.0), $(0.0))

#     cu(1.0) |> typeof

#     mul!(C, mat, B, 1.0, 0.0)
#     mul!(C_cu, mat_cu, B_cu, 1.0, 0.0)

#     abs.((Vector(C_cu) .- C)./Vector(C_cu)) |> maximum

#     At = rand(500, 500)
#     Bt = rand(500, 500)
#     Ct = rand(500, 500)

#     @benchmark mul!(Ct, At, Bt, 1.0, 1.0)
#     @which mul!(Ct, At, Bt, 0.0, true)
# end