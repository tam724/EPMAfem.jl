@concrete struct Rank1DiscretePNVector{co} <: AbstractDiscretePNVector{co}
    model
    arch
    bϵ

    # might be moved to gpu
    bxp
    bΩp
end

@concrete struct WeightedArrayOfRank1PNVector{co} <: AbstractDiscretePNVector{co}
    weights
    arr_r1
end

function weighted(weights, arr_r1::Array{<:Rank1DiscretePNVector{co}}) where co
    @assert size(weights) == size(arr_r1)
    return WeightedArrayOfRank1PNVector{co}(weights, arr_r1)
end

function (b::Rank1DiscretePNVector{true})(it::NonAdjointIterator)
    Δϵ = step(energy_model(b.model))

    T = base_type(b.arch)
    buf = allocate_vec(b.arch, length(b.bxp))
    integral = zero(T)

    for (ϵ, i_ϵ) in it
        ψp = pview(current_solution(it.system), it.system.problem.model)
        integral += Δϵ * b.bϵ[i_ϵ]*dot_buf(b.bxp, ψp, b.bΩp, buf)   
    end
    return integral
end

function (b::Rank1DiscretePNVector{false})(it::AdjointIterator)
    Δϵ = step(energy_model(b.model))

    T = base_type(b.arch)
    buf = allocate_vec(b.arch, length(b.bxp))
    integral = zero(T)

    for (ϵ, i_ϵ) in it
        ψp = pview(current_solution(it.system), it.system.problem.model)

        if i_ϵ != 1 # (where ψp is initialized to 0 anyways..)
            integral += Δϵ * T(0.5) * (b.bϵ[i_ϵ] + b.bϵ[i_ϵ-1])*dot_buf(b.bxp, ψp, b.bΩp, buf)
        end  
    end
    return integral
end

function (b_arr::Array{<:Rank1DiscretePNVector{true}})(it::NonAdjointIterator)
    problem = it.system.problem
    Δϵ = step(energy_model(problem.model))

    T = base_type(problem.arch)
    (_, (nxp, _), (_, _)) = n_basis(problem)
    buf = allocate_vec(problem.arch, nxp)
    integral = zeros(T, size(b_arr))

    for (ϵ, i_ϵ) in it
        ψp = pview(current_solution(it.system), it.system.problem.model)

        for i in eachindex(b_arr) # this could be made more efficient by reusing unique (=== !) bxp, bΩp etc.
            integral[i] += Δϵ * b_arr[i].bϵ[i_ϵ]*dot_buf(b_arr[i].bxp, ψp, b_arr[i].bΩp, buf)   
        end
    end
    return integral
end

function (b_arr::Array{<:Rank1DiscretePNVector{false}})(it::AdjointIterator)
    problem = it.system.problem
    Δϵ = step(energy_model(problem.model))

    T = base_type(problem.arch)
    (_, (nxp, _), (_, _)) = n_basis(problem)
    buf = allocate_vec(problem.arch, nxp)

    integral = zeros(T, size(b_arr))

    for (ϵ, i_ϵ) in it
        ψp = pview(current_solution(it.system), it.system.problem.model)
        for i in eachindex(b_arr) # this could be made more efficient by reusing unique (=== !) bxp, bΩp etc.
            if i_ϵ != 1 # (where ψp is initialized to 0 anyways..)
                integral[i] += Δϵ * T(0.5) * (b_arr[i].bϵ[i_ϵ] + b_arr[i].bϵ[i_ϵ-1])*dot_buf(b_arr[i].bxp, ψp, b_arr[i].bΩp, buf)
            end  
        end
    end
    return integral
end

function assemble_rhs!(b, rhs::WeightedArrayOfRank1PNVector{true}, i_ϵ, Δ, sym)
    fill!(b, zero(eltype(b)))

    nLp = length(first(rhs.arr_r1).bxp)
    nRp = length(first(rhs.arr_r1).bΩp)

    bp = reshape(@view(b[1:nLp*nRp]), (nLp, nRp))
    for i in eachindex(rhs.weights, rhs.arr_r1)
        bϵ2 = rhs.arr_r1[i].bϵ[i_ϵ]
        bxpi_mat = reshape(rhs.arr_r1[i].bxp, (length(rhs.arr_r1[i].bxp), 1))
        bΩpi_mat = reshape(rhs.arr_r1[i].bΩp, (1, length(rhs.arr_r1[i].bΩp)))
        mul!(bp, bxpi_mat, bΩpi_mat, rhs.weights[i]*bϵ2*Δ, true)
    end

    # sym is not used since we only assemble the p part of b here
end

function assemble_rhs_midpoint!(b, rhs::WeightedArrayOfRank1PNVector{false}, i_ϵ, Δ, sym)
    fill!(b, zero(eltype(b)))

    nLp = length(first(rhs.arr_r1).bxp)
    nRp = length(first(rhs.arr_r1).bΩp)

    bp = reshape(@view(b[1:nLp*nRp]), (nLp, nRp))
    for i in eachindex(rhs.weights, rhs.arr_r1)
        bϵ2 = 0.5*(rhs.arr_r1[i].bϵ[i_ϵ] + rhs.arr_r1[i].bϵ[i_ϵ+1])
        # bϵ2 = rhs.arr_r1[i].bϵ[i_ϵ]
        bxpi_mat = reshape(rhs.arr_r1[i].bxp, (length(rhs.arr_r1[i].bxp), 1))
        bΩpi_mat = reshape(rhs.arr_r1[i].bΩp, (1, length(rhs.arr_r1[i].bΩp)))
        mul!(bp, bxpi_mat, bΩpi_mat, rhs.weights[i]*bϵ2*Δ, true)
    end
    
    # sym is not used since we only assemble the p part of b here
end

function assemble_rhs!(b, rhs::Rank1DiscretePNVector{true}, i, Δ, sym)
    fill!(b, zero(eltype(b)))

    nLp = length(bxp)
    nRp = length(bΩp)

    bp = reshape(@view(b[1:nLp*nRp]), (nLp, nRp))
    # bp = pview(b, rhs.model)

    bϵ2 = rhs.bϵ[i]
    bxp_mat = reshape(@view(bxp[:]), (length(bxp), 1))
    bΩp_mat = reshape(@view(bΩp[:]), (1, length(bΩp)))
    mul!(bp, bxp_mat, bΩp_mat, bϵ2*Δ, true)
    # sym is not used since we only assemble the p part of b here
end

function assemble_rhs_midpoint!(b, rhs::Rank1DiscretePNVector{false}, i, Δ, sym)
    fill!(b, zero(eltype(b)))

    nLp = length(bxp)
    nRp = length(bΩp)

    bp = reshape(@view(b[1:nLp*nRp]), (nLp, nRp))
    # bp = pview(b, rhs.model)

    bϵ2 = 0.5*(rhs.bϵ[i] + rhs.bϵ[i+1])
    bxp_mat = reshape(@view(bxp[:]), (length(bxp), 1))
    bΩp_mat = reshape(@view(bΩp[:]), (1, length(bΩp)))
    mul!(bp, bxp_mat, bΩp_mat, bϵ2*Δ, true)
    # sym is not used since we only assemble the p part of b here
end