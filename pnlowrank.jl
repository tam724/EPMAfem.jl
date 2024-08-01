

struct PNProjectedSemidiscretization{T, V<:AbstractVector{T}, M<:AbstractMatrix{T}}
    max_rank::Int64

    Uranks::Vector{Int64}
    Vranks::Vector{Int64}

    UtρpU::Vector{V}
    UtρmU::Vector{V}
    Ut∂pU::Vector{V}
    Ut∇pmU::Vector{V}

    VtIpV::V
    VtImV::V

    VtkpV::Vector{Vector{V}}
    VtkmV::Vector{Vector{V}}

    VtabsΩpV::Vector{V}
    VtΩpmV::Vector{V}

    # # For source (g) and extraction (μ) we only discretize the even parts (the odd parts should always be zero.)
    gxU::Vector{V}
    gΩV::Vector{V}

    μxU::Vector{V}
    μΩV::Vector{V}

    buf::V
end

function pn_projectedsemidiscretization(pn_semi::PNSemidiscretization{T, V, M}, max_rank) where {T, V, M}
    ((nLp, nLm), (nRp, nRm)) = pn_semi.size
    return PNProjectedSemidiscretization{T, V, M}(
        max_rank,
        [0, 0],
        [0, 0],

        [V(undef, 2*max_rank*2*max_rank) for _ in pn_semi.ρp],
        [V(undef, 2*max_rank*2*max_rank) for _ in pn_semi.ρm],
        [V(undef, 2*max_rank*2*max_rank) for _ in pn_semi.∂p],
        [V(undef, 2*max_rank*2*max_rank) for _ in pn_semi.∇pm],

        V(undef, 2*max_rank*2*max_rank),
        V(undef, 2*max_rank*2*max_rank),

        [[V(undef, 2*max_rank*2*max_rank) for _ in kpz] for kpz in pn_semi.kp],
        [[V(undef, 2*max_rank*2*max_rank) for _ in kmz] for kmz in pn_semi.km],

        [V(undef, 2*max_rank*2*max_rank) for _ in pn_semi.absΩp],
        [V(undef, 2*max_rank*2*max_rank) for _ in pn_semi.Ωpm],
        
        [V(undef, 2*max_rank*1) for _ in pn_semi.gx],
        [V(undef, 1*2*max_rank) for _ in pn_semi.gΩ],

        [V(undef, 2*max_rank*1) for _ in pn_semi.μx],
        [V(undef, 1*2*max_rank) for _ in pn_semi.μΩ],
        
        V(undef, 2*max_rank * max(nLp, nLm, nRp, nRm))

    )
end

function U_views(pn_proj_semi::PNProjectedSemidiscretization)
    rankp, rankm = pn_proj_semi.Uranks[1], pn_proj_semi.Uranks[2]

    UtρpU = (reshape(@view(UtρpzU[1:rankp*rankp]), (rankp, rankp)) for UtρpzU in pn_proj_semi.UtρpU)
    UtρmU = (reshape(@view(UtρmzU[1:rankm*rankm]), (rankm, rankm)) for UtρmzU in pn_proj_semi.UtρmU)
    Ut∂pU = (reshape(@view(Ut∂pdU[1:rankp*rankp]), (rankp, rankp)) for Ut∂pdU in pn_proj_semi.Ut∂pU)
    Ut∇pmU = (reshape(@view(Ut∇pmdU[1:rankm*rankp]), (rankm, rankp)) for Ut∇pmdU in pn_proj_semi.Ut∇pmU)
    gxU = (reshape(@view(gxjU[1:rankp]), (rankp, 1)) for gxjU in pn_proj_semi.gxU)
    μxU = (reshape(@view(μxjU[1:rankp]), (rankp, 1)) for μxjU in pn_proj_semi.μxU)
    return UtρpU, UtρmU, Ut∂pU, Ut∇pmU, gxU, μxU
end

function gxU_view(pn_proj_semi::PNProjectedSemidiscretization, gj)
    rankp, rankm = pn_proj_semi.Uranks[1], pn_proj_semi.Uranks[2]
    return reshape(@view(pn_proj_semi.gxU[gj][1:rankp]), (rankp, 1))
end

function V_views(pn_proj_semi::PNProjectedSemidiscretization)
    rankp, rankm = pn_proj_semi.Vranks[1], pn_proj_semi.Vranks[2]

    VtIpV = reshape(@view(pn_proj_semi.VtIpV[1:rankp*rankp]), (rankp, rankp))
    VtImV = reshape(@view(pn_proj_semi.VtImV[1:rankm*rankm]), (rankm, rankm))
    VtkpV = ((reshape(@view(VtkpziV[1:rankp*rankp]), (rankp, rankp)) for VtkpziV in VtkpzV) for VtkpzV in pn_proj_semi.VtkpV)
    VtkmV = ((reshape(@view(VtkmziV[1:rankm*rankm]), (rankm, rankm)) for VtkmziV in VtkmzV) for VtkmzV in pn_proj_semi.VtkmV)
    VtabsΩpV = (reshape(@view(VtabsΩpdV[1:rankp*rankp]), (rankp, rankp)) for VtabsΩpdV in pn_proj_semi.VtabsΩpV)
    VtΩpmV = (reshape(@view(VtΩpmdV[1:rankm*rankp]), (rankm, rankp)) for VtΩpmdV in pn_proj_semi.VtΩpmV)
    gΩV = (reshape(@view(gΩkV[1:rankp]), (1, rankp)) for gΩkV in pn_proj_semi.gΩV)
    μΩV = (reshape(@view(μΩkV[1:rankp]), (1, rankp)) for μΩkV in pn_proj_semi.μΩV)
    return VtIpV, VtImV, VtkpV, VtkmV, VtabsΩpV, VtΩpmV, gΩV, μΩV
end

function gΩV_view(pn_proj_semi::PNProjectedSemidiscretization, gk)
    rankp, rankm = pn_proj_semi.Vranks[1], pn_proj_semi.Vranks[2]
    return reshape(@view(pn_proj_semi.gΩV[gk][1:rankp]), (1, rankp))
end

function mul_buf!(Y, A, B, C, buf)
    buf_mat = reshape(@view(buf[1:size(A, 1)*size(B, 2)]), (size(A, 1), size(B, 2)))
    mul!(buf_mat, A, B, true, false)
    mul!(Y, buf_mat, C, true, false)
end

function mul_buf!(Y, A, B::Diagonal, C, buf)
    buf_mat = reshape(@view(buf[1:size(A, 1)*size(B, 2)]), (size(A, 1), size(B, 2)))
    # mul!(buf_mat, A, B, true, false)
    buf_mat .= A .* transpose(B.diag)
    mul!(Y, buf_mat, C, true, false)
end

function update_Vt!(pn_proj_semi, pn_semi, (; Vtp, Vtm))
    rankp = size(Vtp, 1)
    rankm = size(Vtm, 1)
    pn_proj_semi.Vranks[1] = rankp
    pn_proj_semi.Vranks[2] = rankm

    VtIpV, VtImV, VtkpV, VtkmV, VtabsΩpV, VtΩpmV, gΩV, μΩV = V_views(pn_proj_semi)

    buf = pn_proj_semi.buf

    # TODO: make inplace matrix multiplications

    # VtIpV .= Vtp*pn_semi.Ip*transpose(Vtp)
    mul_buf!(VtIpV, Vtp, pn_semi.Ip, transpose(Vtp), buf)
    # VtImV .= Vtm*pn_semi.Im*transpose(Vtm)
    mul_buf!(VtImV, Vtm, pn_semi.Im, transpose(Vtm), buf)

    for (VtkpzV, kpz) in zip(VtkpV, pn_semi.kp)
        for (Vtkpzi, kpzi) in zip(VtkpzV, kpz)
            # Vtkpzi .= Vtp*kpzi*transpose(Vtp)
            mul_buf!(Vtkpzi, Vtp, kpzi, transpose(Vtp), buf)
        end
    end

    for (VtkmzV, kmz) in zip(VtkmV, pn_semi.km)
        for (Vtkmzi, kmzi) in zip(VtkmzV, kmz)
            # Vtkmzi .= Vtm*kmzi*transpose(Vtm)
            mul_buf!(Vtkmzi, Vtm, kmzi, transpose(Vtm), buf)
        end
    end

    for (VtabsΩpdV, absΩpd) in zip(VtabsΩpV, pn_semi.absΩp)
        # VtabsΩpdV .= Vtp*absΩpd*transpose(Vtp)
        mul_buf!(VtabsΩpdV, Vtp, absΩpd, transpose(Vtp), buf)
    end

    for (VtΩpmdV, Ωpmd) in zip(VtΩpmV, pn_semi.Ωpm)
        # VtΩpmdV .= Vtm*Ωpmd*transpose(Vtp)
        mul_buf!(VtΩpmdV, Vtm, Ωpmd, transpose(Vtp), buf)
    end

    for (gΩkV, gΩk) in zip(gΩV, pn_semi.gΩ)
        mul!(gΩkV, gΩk, transpose(Vtp))
    end

    for (μΩkV, μΩk) in zip(μΩV, pn_semi.μΩ)
        mul!(μΩkV, μΩk, transpose(Vtp))
    end
    return
end

function update_U!(pn_proj_semi, pn_semi, (; Up, Um))
    rankp = size(Up, 2)
    rankm = size(Um, 2)
    pn_proj_semi.Uranks[1] = rankp
    pn_proj_semi.Uranks[2] = rankm

    UtρpU, UtρmU, Ut∂pU, Ut∇pmU, gxU, μxU = U_views(pn_proj_semi)
    
    buf = pn_proj_semi.buf

    for (UtρpzU, ρpz) in zip(UtρpU, pn_semi.ρp)
        # UtρpzU .= transpose(Up)*ρpz*Up
        mul_buf!(UtρpzU, transpose(Up), ρpz, Up, buf)
    end

    for (UtρmzU, ρmz) in zip(UtρmU, pn_semi.ρm)
        # UtρmzU .= transpose(Um)*ρmz*Um
        mul_buf!(UtρmzU, transpose(Um), ρmz, Um, buf)
    end

    for (Ut∂pdU, ∂pd) in zip(Ut∂pU, pn_semi.∂p)
        # Ut∂pdU .= transpose(Up)*∂pd*Up
        mul_buf!(Ut∂pdU, transpose(Up), ∂pd, Up, buf)

    end

    for (Ut∇pmdU, ∇pmd) in zip(Ut∇pmU, pn_semi.∇pm)
        # Ut∇pmdU .= transpose(Um)*∇pmd*Up
        mul_buf!(Ut∇pmdU, transpose(Um), ∇pmd, Up, buf)
    end

    for (gxjU, gxj) in zip(gxU, pn_semi.gx)
        mul!(gxjU, transpose(Up), gxj)
    end

    for (μxjU, μxj) in zip(μxU, pn_semi.μx)
        mul!(μxjU, transpose(Up), μxj)
    end
    return
end