@concrete struct Rank1DiscretePNVector{co} <: AbstractDiscretePNVector{co}
    model
    bϵ

    # might be moded to gpu
    bxp
    bΩp
end

struct DiscretePNVector{co} <: AbstractDiscretePNVector{co}
    model

    weights
    bϵ

    #(might be moved to gpu)
    bxp
    bΩp
end


function assemble_rhs_p!(b, rhs::DiscretePNVector{false}, i, Δ; bxp=rhs.bxp, bΩp=rhs.bΩp)
    fill!(b, zero(eltype(b)))

    nLp = length(first(bxp))
    nRp = length(first(bΩp))

    bp = reshape(@view(b[1:nLp*nRp]), (nLp, nRp))
    # bp = pview(b, rhs.model)
    for (ϵi, bϵi) in zip(1:size(rhs.weights, 1), rhs.bϵ)
        for (xi, bxpi) in zip(1:size(rhs.weights, 2), bxp)
            for (Ωi, bΩpi) in zip(1:size(rhs.weights, 3), bΩp)
                bϵ2 = bϵi[i]
                bxpi_mat = reshape(@view(bxpi[:]), (length(bxpi), 1))
                bΩpi_mat = reshape(@view(bΩpi[:]), (1, length(bΩpi)))
                mul!(bp, bxpi_mat, bΩpi_mat, rhs.weights[ϵi, xi, Ωi]*bϵ2*Δ, 1.0)
            end
        end
    end
end

function assemble_rhs_p_midpoint!(b, rhs::DiscretePNVector{true}, i, Δ; bxp=rhs.bxp, bΩp=rhs.bΩp)
    fill!(b, zero(eltype(b)))

    nLp = length(first(bxp))
    nRp = length(first(bΩp))

    bp = reshape(@view(b[1:nLp*nRp]), (nLp, nRp))
    # bp = pview(b, rhs.model)
    for (ϵi, bϵi) in zip(1:size(rhs.weights, 1), rhs.bϵ)
        for (xi, bxpi) in zip(1:size(rhs.weights, 2), bxp)
            for (Ωi, bΩpi) in zip(1:size(rhs.weights, 3), bΩp)
                bϵ2 = 0.5*(bϵi[i] + bϵi[i+1])
                bxpi_mat = reshape(@view(bxpi[:]), (length(bxpi), 1))
                bΩpi_mat = reshape(@view(bΩpi[:]), (1, length(bΩpi)))
                mul!(bp, bxpi_mat, bΩpi_mat, rhs.weights[ϵi, xi, Ωi]*bϵ2*Δ, 1.0)
            end
        end
    end
end

function assemble_rhs_p!(b, rhs::Rank1DiscretePNRHS, i, Δ; bxp=rhs.bxp, bΩp=rhs.bΩp)
    fill!(b, zero(eltype(b)))

    nLp = length(bxp)
    nRp = length(bΩp)

    bp = reshape(@view(b[1:nLp*nRp]), (nLp, nRp))
    # bp = pview(b, rhs.model)

    bϵ2 = rhs.bϵ[i]
    bxp_mat = reshape(@view(bxp[:]), (length(bxp), 1))
    bΩp_mat = reshape(@view(bΩp[:]), (1, length(bΩp)))
    mul!(bp, bxp_mat, bΩp_mat, bϵ2*Δ, 1.0)
end

function assemble_rhs_p_midpoint!(b, rhs::Rank1DiscretePNRHS, i, Δ; bxp=rhs.bxp, bΩp=rhs.bΩp)

    fill!(b, zero(eltype(b)))

    nLp = length(bxp)
    nRp = length(bΩp)

    bp = reshape(@view(b[1:nLp*nRp]), (nLp, nRp))
    # bp = pview(b, rhs.model)

    bϵ2 = 0.5*(rhs.bϵ[i] + rhs.bϵ[i+1])
    bxp_mat = reshape(@view(bxp[:]), (length(bxp), 1))
    bΩp_mat = reshape(@view(bΩp[:]), (1, length(bΩp)))
    mul!(bp, bxp_mat, bΩp_mat, bϵ2*Δ, 1.0)
end
