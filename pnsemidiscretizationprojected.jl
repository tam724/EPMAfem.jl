

struct PNProjectedSemidiscretization{T, V<:AbstractVector{T}, M<:AbstractMatrix{T}}
    max_rank::Int64

    ρpU::Vector{M}
    ρmU::Vector{M}
    ∂pU::Vector{M}
    ∇pmU::Vector{M}

    # Basically these two should never be used.. they are diagonal anyways..
    # IpV::Diagonal{M, V}
    # ImV::Diagonal{M, V}

    kpV::Vector{Vector{Diagonal{T, V}}}
    kmV::Vector{Vector{Diagonal{T, V}}}

    absΩpV::Vector{M}
    ΩpmV::Vector{M}

    # # For source (g) and extraction (μ) we only discretize the even parts (the odd parts should always be zero.)
    # gx::Vector{SM}
    # gΩ::Vector{M}

    # μx::Vector{M}
    # μΩ::Vector{M}

    # ρ_to_ρp::SM
    # ρ_to_ρm::Diagonal{T, V}
    temp::V
end

function pn_projectedsemidiscretization(pn_semi::PNSemidiscretization{T, V, M}, max_rank) where {T, V, M}
    ((nLp, nLm), (nRp, nRm)) = pn_semi.size
    return PNProjectedSemidiscretization(
        max_rank,
        [M(undef, 2*max_rank, 2*max_rank) for _ in pn_semi.ρp],
        [M(undef, 2*max_rank, 2*max_rank) for _ in pn_semi.ρm],
        [M(undef, 2*max_rank, 2*max_rank) for _ in pn_semi.∂pU],
        [M(undef, 2*max_rank, 2*max_rank) for _ in pn_semi.∇pmU],

        [[M(undef, 2*max_rank, 2*max_rank) for _ in kpz] for kpz in pn_semi.kp],
        [[M(undef, 2*max_rank, 2*max_rank) for _ in kmz] for kmz in pn_semi.km],

        [M(undef, 2*max_rank, 2*max_rank) for _ in pn_semi.absΩp],
        [M(undef, 2*max_rank, 2*max_rank) for _ in pn_semi.ΩpmV],
        V(undef, 2*max_rank * max(nLp, nLm, nRp, nRm))
    )
end

function update_Vt!(pn_proj_semi, pn_semi, Vt)
    rankp = size(Vt[1], 1)
    rankm = size(Vt[2], 1)
    ((nLp, nLm), (nRp, nRm)) = pn_semi.size

    
end

function update_U!(pn_proj_semi, pn_semi, U)
    rankp = size(U[1], 2)
    rankm = size(U[2], 2)

    ((nLp, nLm), (nRp, nRm)) = pn_semi.size
    temp = reshape(@view(pn_proj_semi.temp[1:rankp*nLp]), )
    for (ρpVz, ρpz) in zip(pn_proj_semi.ρpV, pn_semi.ρp)
        mul!(temp, U[1], ρpz)
        mul!(ρpVz, temp, U[1])
    end


