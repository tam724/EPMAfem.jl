

struct PNProjectedSemidiscretization{T, V<:AbstractVector{T}, M<:AbstractMatrix{T}}
    max_rank::Int64

    UtρpU::Vector{M}
    UtρmU::Vector{M}
    Ut∂pU::Vector{M}
    Ut∇pmU::Vector{M}

    VtIpV::M
    VtImV::M

    VtkpV::Vector{Vector{M}}
    VtkmV::Vector{Vector{M}}

    VtabsΩpV::Vector{M}
    VtΩpmV::Vector{M}

    # # For source (g) and extraction (μ) we only discretize the even parts (the odd parts should always be zero.)
    gxU::Vector{M}
    gΩV::Vector{M}

    μxU::Vector{M}
    μΩV::Vector{M}

    # ρ_to_ρp::SM
    # ρ_to_ρm::Diagonal{T, V}
    #temp::V
end

function pn_projectedsemidiscretization(pn_semi::PNSemidiscretization{T, V, M}, max_rank) where {T, V, M}
    ((nLp, nLm), (nRp, nRm)) = pn_semi.size
    return PNProjectedSemidiscretization{T, V, M}(
        max_rank,
        [M(undef, 2*max_rank, 2*max_rank) for _ in pn_semi.ρp],
        [M(undef, 2*max_rank, 2*max_rank) for _ in pn_semi.ρm],
        [M(undef, 2*max_rank, 2*max_rank) for _ in pn_semi.∂p],
        [M(undef, 2*max_rank, 2*max_rank) for _ in pn_semi.∇pm],

        M(undef, 2*max_rank,  2*max_rank),
        M(undef, 2*max_rank,  2*max_rank),

        [[M(undef, 2*max_rank, 2*max_rank) for _ in kpz] for kpz in pn_semi.kp],
        [[M(undef, 2*max_rank, 2*max_rank) for _ in kmz] for kmz in pn_semi.km],

        [M(undef, 2*max_rank, 2*max_rank) for _ in pn_semi.absΩp],
        [M(undef, 2*max_rank, 2*max_rank) for _ in pn_semi.Ωpm],
        
        [M(undef, 2*max_rank, 1) for _ in pn_semi.gx],
        [M(undef, 1, 2*max_rank) for _ in pn_semi.gΩ],

        [M(undef, 2*max_rank, 1) for _ in pn_semi.μx],
        [M(undef, 1, 2*max_rank) for _ in pn_semi.μΩ]
        
        #V(undef, 2*max_rank * max(nLp, nLm, nRp, nRm))

    )
end

function update_Vt!(pn_proj_semi, pn_semi, (; Vtp, Vtm))
    rankp = size(Vtp, 1)
    rankm = size(Vtm, 1)
    ((nLp, nLm), (nRp, nRm)) = pn_semi.size

    # TODO: make inplace matrix multiplications

    pn_proj_semi.VtIpV .= Vtp*pn_semi.Ip*transpose(Vtp)
    pn_proj_semi.VtImV .= Vtm*pn_semi.Im*transpose(Vtm)

    for e in 1:number_of_elements(equations(pn_semi))
        for i in 1:number_of_scatterings(equations(pn_semi))
            pn_proj_semi.VtkpV[e][i] .= Vtp*pn_semi.kp[e][i]*transpose(Vtp)
            pn_proj_semi.VtkmV[e][i] .= Vtm*pn_semi.km[e][i]*transpose(Vtm)
        end
    end

    for d in 1:length(pn_semi.absΩp)
        pn_proj_semi.VtabsΩpV[d] .= Vtp*pn_semi.absΩp[d]*transpose(Vtp)
        pn_proj_semi.VtΩpmV[d] .= Vtm*pn_semi.Ωpm[d]*transpose(Vtp)
    end

    for k in 1:length(pn_semi.gΩ)
        pn_proj_semi.gΩV[k] .= pn_semi.gΩ[k] * transpose(Vtp)
    end

    for k in 1:length(pn_semi.μΩ)
        pn_proj_semi.μΩV[k] .= pn_semi.μΩ[k] * transpose(Vtp)
    end
end

function update_U!(pn_proj_semi, pn_semi, (; Up, Um))
    rankp = size(Up, 2)
    rankm = size(Um, 2)

    ((nLp, nLm), (nRp, nRm)) = pn_semi.size
    # temp = reshape(@view(pn_proj_semi.temp[1:rankp*nLp]), )
    for e in 1:number_of_elements(equations(pn_semi))
        pn_proj_semi.UtρpU[e] .= transpose(Up)*pn_semi.ρp[e]*Up
        pn_proj_semi.UtρmU[e] .= transpose(Um)*pn_semi.ρm[e]*Um
    end

    for d in 1:length(pn_semi.∂p)
        pn_proj_semi.Ut∂pU[d] .=  transpose(Up)*pn_semi.∂p[d]*Up
        pn_proj_semi.Ut∇pmU[d] .=  transpose(Um)*pn_semi.∇pm[d]*Up
    end

    for j in 1:length(pn_semi.gx)
        pn_proj_semi.gxU[j] .= transpose(Up) * pn_semi.gx[j]
    end

    for j in 1:length(pn_semi.μx)
        pn_proj_semi.μxU[j] .= transpose(Up) * pn_semi.μx[j]
    end
end
