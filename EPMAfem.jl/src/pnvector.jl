@concrete struct Rank1DiscretePNVector{co} <: AbstractDiscretePNVector{co}
    model
    bϵ

    # might be moved to gpu
    bxp
    bΩp
end

@concrete struct DiscretePNVector{co} <: AbstractDiscretePNVector{co}
    model

    weights
    bϵ

    #(might be moved to gpu)
    bxp
    bΩp
end

@concrete struct ArrayOfRank1DiscretePNVector{co} <: AbstractDiscretePNVector{co}
    model
    bϵs

    # might be moved to gpu
    bxps
    bΩps
end

function Base.size(arr_r1::ArrayOfRank1DiscretePNVector)
    return length(arr_r1.bϵs), length(arr_r1.bxps), length(arr_r1.bΩps)
end

function Base.getindex(arr_r1::ArrayOfRank1DiscretePNVector{co}, i, j, k) where co
    return Rank1DiscretePNVector{co}(arr_r1.model, arr_r1.bϵs[i], arr_r1.bxps[j], arr_r1.bΩps[k])
end

function weight_array_of_r1(weights, arr_r1::ArrayOfRank1DiscretePNVector{co}) where co
    return DiscretePNVector{co}(arr_r1.model, weights, arr_r1.bϵs, arr_r1.bxps, arr_r1.bΩps)
end
@concrete struct VecOfRank1DiscretePNVector{co} <: AbstractDiscretePNVector{co}
    model

    bϵs

    #(might be moved to gpu)
    bxps
    bΩps
end

function Base.size(vec::VecOfRank1DiscretePNVector)
    @assert length(vec.bϵs) == length(vec.bxps) == length(vec.bΩps)
    return length(vec.bϵs)
end

function Base.getindex(vec::VecOfRank1DiscretePNVector{co}, i) where co
    return Rank1DiscretePNVector{co}(vec.model, vec.bϵs[i], vec.bxps[i], vec.bΩps[i])
end 

function (b::Rank1DiscretePNVector{true})(it::NonAdjointIterator)
    Δϵ = step(b.model.energy_model)
    integral = 0.0
    for (ϵ, i_ϵ) in it
        ψp = pview(current_solution(it.solver), it.system.model)
        integral += Δϵ * b.bϵ[i_ϵ]*dot(b.bxp, ψp * b.bΩp)   
    end
    return integral
end

function (b::ArrayOfRank1DiscretePNVector{true})(it::NonAdjointIterator)
    Δϵ = step(b.model.energy_model)
    integral = zeros(size(b))
    for (ϵ, i_ϵ) in it
        ψp = pview(current_solution(it.solver), it.system.model)

        for i in 1:length(b.bϵs)
            for j in 1:length(b.bxps)
                for k in 1:length(b.bΩps)
                    integral[i, j, k] += Δϵ * b.bϵs[i][i_ϵ]*dot(b.bxps[j], ψp * b.bΩps[k])   
                end
            end
        end
    end
    return integral
end

function (b::VecOfRank1DiscretePNVector{true})(it::NonAdjointIterator)
    Δϵ = step(b.model.energy_model)
    integral = zeros(size(b))
    for (ϵ, i_ϵ) in it
        ψp = pview(current_solution(it.solver), it.system.model)

        for i in 1:length(b.bϵs)
            integral[i] += Δϵ * b.bϵs[i][i_ϵ]*dot(b.bxps[i], ψp * b.bΩps[i])   
        end
    end
    return integral
end

function (b::Rank1DiscretePNVector{false})(it::AdjointIterator)
    Δϵ = step(b.model.energy_model)
    integral = 0.0
    for (ϵ, i_ϵ) in it
        ψp = pview(current_solution(it.solver), it.system.model)

        if i_ϵ != 1 # (where ψp is initialized to 0 anyways..)
            integral += Δϵ * 0.5 * (b.bϵ[i_ϵ] + b.bϵ[i_ϵ-1])*dot(b.bxp, ψp * b.bΩp)
        end  
    end
    return integral
end

function (b::ArrayOfRank1DiscretePNVector{false})(it::AdjointIterator)
    Δϵ = step(b.model.energy_model)
    integral = zeros(size(b))
    for (ϵ, i_ϵ) in it
        ψp = pview(current_solution(it.solver), it.system.model)

        for i in 1:length(b.bϵs)
            for j in 1:length(b.bxps)
                for k in 1:length(b.bΩps)
                    if i_ϵ != 1 # (where ψp is initialized to 0 anyways..)
                        integral[i, j, k] += Δϵ * 0.5 * (b.bϵs[i][i_ϵ] + b.bϵs[i][i_ϵ-1])*dot(b.bxps[j], ψp * b.bΩps[k])
                    end  
                end
            end
        end
    end
    return integral
end

function (b::VecOfRank1DiscretePNVector{false})(it::AdjointIterator)
    Δϵ = step(b.model.energy_model)
    integral = zeros(size(b))
    for (ϵ, i_ϵ) in it
        ψp = pview(current_solution(it.solver), it.system.model)

        for i in 1:length(b.bϵs)
            if i_ϵ != 1 # (where ψp is initialized to 0 anyways..)
                integral[i, j, k] += Δϵ * 0.5 * (b.bϵs[i][i_ϵ] + b.bϵs[i][i_ϵ-1])*dot(b.bxps[i], ψp * b.bΩps[i])
            end  
        end
    end
    return integral
end

function assemble_rhs_p!(b, rhs::DiscretePNVector{true}, i, Δ; bxp=rhs.bxp, bΩp=rhs.bΩp)
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

function assemble_rhs_p_midpoint!(b, rhs::DiscretePNVector{false}, i, Δ; bxp=rhs.bxp, bΩp=rhs.bΩp)
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

function assemble_rhs_p!(b, rhs::Rank1DiscretePNVector{true}, i, Δ; bxp=rhs.bxp, bΩp=rhs.bΩp)
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

function assemble_rhs_p_midpoint!(b, rhs::Rank1DiscretePNVector{false}, i, Δ; bxp=rhs.bxp, bΩp=rhs.bΩp)
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
