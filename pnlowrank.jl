struct ProjectedPNProblem{T, V<:AbstractVector{T}}
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

    gxpU::V
    gΩpV::V

    μxpU::V
    μΩpV::V

    buf::V
end

function pn_projectedproblem(pn_eq, discrete_model, max_rank)
    (_, (nLp, nLm), (nRp, nRm)) = problem.model.n_basis
    VT = vec_type(discrete_model)
    return ProjectedPNProblem(
        max_rank,
        [0, 0],
        [0, 0],

        [VT(undef, 2*max_rank*2*max_rank) for _ in 1:number_of_elements(pn_eq)],
        [VT(undef, 2*max_rank*2*max_rank) for _ in 1:number_of_elements(pn_eq)],
        [VT(undef, 2*max_rank*2*max_rank) for _ in 1:number_of_dimensions(space(discrete_model))],
        [VT(undef, 2*max_rank*2*max_rank) for _ in 1:number_of_dimensions(space(discrete_model))],

        VT(undef, 2*max_rank*2*max_rank),
        VT(undef, 2*max_rank*2*max_rank),

        [[VT(undef, 2*max_rank*2*max_rank) for _ in 1:number_of_scatterings(pn_eq)] for kpz in 1:number_of_elements(pn_eq)],
        [[VT(undef, 2*max_rank*2*max_rank) for _ in 1:number_of_scatterings(pn_eq)] for kmz in 1:number_of_elements(pn_eq)],

        [VT(undef, 2*max_rank*2*max_rank) for _ in 1:number_of_dimensions(space(discrete_model))],
        [VT(undef, 2*max_rank*2*max_rank) for _ in 1:number_of_dimensions(space(discrete_model))],
        
        VT(undef, 2*max_rank*1),
        VT(undef, 1*2*max_rank),

        VT(undef, 2*max_rank*1),
        VT(undef, 1*2*max_rank),
        
        VT(undef, 2*max_rank * max(nLp, nLm, nRp, nRm))
    )
end

function U_views(proj_problem::ProjectedPNProblem)
    rankp, rankm = proj_problem.Uranks[1], proj_problem.Uranks[2]

    UtρpU = (reshape(@view(UtρpzU[1:rankp*rankp]), (rankp, rankp)) for UtρpzU in proj_problem.UtρpU)
    UtρmU = (reshape(@view(UtρmzU[1:rankm*rankm]), (rankm, rankm)) for UtρmzU in proj_problem.UtρmU)
    Ut∂pU = (reshape(@view(Ut∂pdU[1:rankp*rankp]), (rankp, rankp)) for Ut∂pdU in proj_problem.Ut∂pU)
    Ut∇pmU = (reshape(@view(Ut∇pmdU[1:rankm*rankp]), (rankm, rankp)) for Ut∇pmdU in proj_problem.Ut∇pmU)
    gxpU = reshape(@view(proj_problem.gxpU[1:rankp]), (rankp, 1))
    μxpU = reshape(@view(proj_problem.μxpU[1:rankp]), (rankp, 1))
    return UtρpU, UtρmU, Ut∂pU, Ut∇pmU, gxpU, μxpU
end

function gxpU_view(proj_problem::ProjectedPNProblem)
    rankp, rankm = proj_problem.Uranks[1], proj_problem.Uranks[2]
    return reshape(@view(proj_problem.gxpU[1:rankp]), (rankp, 1))
end

function V_views(proj_problem::ProjectedPNProblem)
    rankp, rankm = proj_problem.Vranks[1], proj_problem.Vranks[2]

    VtIpV = reshape(@view(proj_problem.VtIpV[1:rankp*rankp]), (rankp, rankp))
    VtImV = reshape(@view(proj_problem.VtImV[1:rankm*rankm]), (rankm, rankm))
    VtkpV = ((reshape(@view(VtkpziV[1:rankp*rankp]), (rankp, rankp)) for VtkpziV in VtkpzV) for VtkpzV in proj_problem.VtkpV)
    VtkmV = ((reshape(@view(VtkmziV[1:rankm*rankm]), (rankm, rankm)) for VtkmziV in VtkmzV) for VtkmzV in proj_problem.VtkmV)
    VtabsΩpV = (reshape(@view(VtabsΩpdV[1:rankp*rankp]), (rankp, rankp)) for VtabsΩpdV in proj_problem.VtabsΩpV)
    VtΩpmV = (reshape(@view(VtΩpmdV[1:rankm*rankp]), (rankm, rankp)) for VtΩpmdV in proj_problem.VtΩpmV)
    gΩpV = reshape(@view(proj_problem.gΩpV[1:rankp]), (1, rankp))
    μΩpV = reshape(@view(proj_problem.μΩpV[1:rankp]), (1, rankp))
    return VtIpV, VtImV, VtkpV, VtkmV, VtabsΩpV, VtΩpmV, gΩpV, μΩpV
end

function gΩpV_view(proj_problem::ProjectedPNProblem)
    rankp, rankm = proj_problem.Vranks[1], proj_problem.Vranks[2]
    return reshape(@view(proj_problem.gΩpV[1:rankp]), (1, rankp))
end

# computes Y = A*B*C
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

function update_Vt!(proj_problem, problem, (; Vtp, Vtm))
    rankp = size(Vtp, 1)
    rankm = size(Vtm, 1)
    proj_problem.Vranks[1] = rankp
    proj_problem.Vranks[2] = rankm

    VtIpV, VtImV, VtkpV, VtkmV, VtabsΩpV, VtΩpmV, gΩpV, μΩpV = V_views(proj_problem)

    buf = proj_problem.buf

    # VtIpV .= Vtp*pn_semi.Ip*transpose(Vtp)
    mul_buf!(VtIpV, Vtp, problem.Ip, transpose(Vtp), buf)
    # VtImV .= Vtm*pn_semi.Im*transpose(Vtm)
    mul_buf!(VtImV, Vtm, problem.Im, transpose(Vtm), buf)

    for (VtkpzV, kpz) in zip(VtkpV, problem.kp)
        for (Vtkpzi, kpzi) in zip(VtkpzV, kpz)
            # Vtkpzi .= Vtp*kpzi*transpose(Vtp)
            mul_buf!(Vtkpzi, Vtp, kpzi, transpose(Vtp), buf)
        end
    end

    for (VtkmzV, kmz) in zip(VtkmV, problem.km)
        for (Vtkmzi, kmzi) in zip(VtkmzV, kmz)
            # Vtkmzi .= Vtm*kmzi*transpose(Vtm)
            mul_buf!(Vtkmzi, Vtm, kmzi, transpose(Vtm), buf)
        end
    end

    for (VtabsΩpdV, absΩpd) in zip(VtabsΩpV, problem.absΩp)
        # VtabsΩpdV .= Vtp*absΩpd*transpose(Vtp)
        mul_buf!(VtabsΩpdV, Vtp, absΩpd, transpose(Vtp), buf)
    end

    for (VtΩpmdV, Ωpmd) in zip(VtΩpmV, problem.Ωpm)
        # VtΩpmdV .= Vtm*Ωpmd*transpose(Vtp)
        mul_buf!(VtΩpmdV, Vtm, Ωpmd, transpose(Vtp), buf)
    end

    mul!(gΩpV, problem.gΩp, transpose(Vtp))
    mul!(μΩpV, problem.μΩp, transpose(Vtp))
    return
end

function update_U!(proj_problem, problem, (; Up, Um))
    rankp = size(Up, 2)
    rankm = size(Um, 2)
    proj_problem.Uranks[1] = rankp
    proj_problem.Uranks[2] = rankm

    UtρpU, UtρmU, Ut∂pU, Ut∇pmU, gxpU, μxpU = U_views(proj_problem)
    
    buf = proj_problem.buf

    for (UtρpzU, ρpz) in zip(UtρpU, problem.ρp)
        # UtρpzU .= transpose(Up)*ρpz*Up
        mul_buf!(UtρpzU, transpose(Up), ρpz, Up, buf)
    end

    for (UtρmzU, ρmz) in zip(UtρmU, problem.ρm)
        # UtρmzU .= transpose(Um)*ρmz*Um
        mul_buf!(UtρmzU, transpose(Um), ρmz, Um, buf)
    end

    for (Ut∂pdU, ∂pd) in zip(Ut∂pU, problem.∂p)
        # Ut∂pdU .= transpose(Up)*∂pd*Up
        mul_buf!(Ut∂pdU, transpose(Up), ∂pd, Up, buf)

    end

    for (Ut∇pmdU, ∇pmd) in zip(Ut∇pmU, problem.∇pm)
        # Ut∇pmdU .= transpose(Um)*∇pmd*Up
        mul_buf!(Ut∇pmdU, transpose(Um), ∇pmd, Up, buf)
    end

    mul!(gxpU, transpose(Up), problem.gxp)
    mul!(μxpU, transpose(Up), problem.μxp)
    return
end