@concrete struct Rank1DiscretePNVector <: AbstractDiscretePNVector
    adjoint::Bool
    model
    arch
    bϵ
    # might be moved to gpu
    bxp
    bΩp
end

_is_adjoint_vector(b::Rank1DiscretePNVector) = b.adjoint

@concrete struct WeightedArrayOfRank1PNVector <: AbstractDiscretePNVector
    weights
    arr_r1
end

function weighted(weights, arr_r1::Array{<:Rank1DiscretePNVector})
    @assert size(weights) == size(arr_r1)
    if !all(r1 -> _is_adjoint_vector(first(arr_r1)) == _is_adjoint_vector(r1), arr_r1) @warn "creating a weighted array of vectors with different adjoint properties" end
    return WeightedArrayOfRank1PNVector(weights, arr_r1)
end

function solve_and_integrate_nonadjoint(b::Rank1DiscretePNVector, it::AbstractDiscretePNSolution)
    @assert _is_adjoint_vector(b) && !_is_adjoint_solution(it)
    model = b.model
    arch = b.arch

    Δϵ = step(energy_model(model))

    T = base_type(arch)
    (_, (nxp, _), (nΩp, _)) = n_basis(model)
    buf = allocate_vec(arch, nΩp)
    integral = zero(T)

    # trapezoidal rule, if ψp[end] == 0 and b.bϵ[1] == 0 (we require this for dual consistency)
    for (idx, ψ) in it
        ψp = pview(ψ, model)
        integral += Δϵ * b.bϵ[idx]*dot_buf(b.bxp, ψp, b.bΩp, buf)
    end
    return integral
end

function solve_and_integrate_adjoint(b::Rank1DiscretePNVector, it::AbstractDiscretePNSolution)
    @assert !_is_adjoint_vector(b) && _is_adjoint_solution(it)
    model = b.model
    arch = b.arch
    Δϵ = step(energy_model(model))

    T = base_type(arch)
    (_, (nxp, _), (nΩp, _)) = n_basis(model)
    buf = allocate_vec(arch, nΩp)
    integral = zero(T)

    for (idx, ψ) in it
        if is_first(idx) continue end # (where ψp is initialized to 0 anyways..)
        ψp = pview(ψ, model)
        integral += Δϵ * T(0.5) * (b.bϵ[plus½(idx)] + b.bϵ[minus½(idx)])*dot_buf(b.bxp, ψp, b.bΩp, buf)
    end
    return integral
end

function ideal_index_order(b_arr::Array{<:Rank1DiscretePNVector})
    d = Dict{Int64, Vector{Int64}}()
    for i in eachindex(b_arr)
        key_or_nothing = findfirst(((key, val), ) -> b_arr[key].bxp === b_arr[i].bxp, ((k, v) for (k, v) in d))
        push!(get!(Vector{typeof(i)}, d, isnothing(key_or_nothing) ? i : key_or_nothing), i)
    end
    d2 = Dict{Int64, Dict{Int64, Vector{Int64}}}()
    for (k, v) in d
        d2[k] = Dict()
        for i in eachindex(v)
            key_or_nothing = findfirst(((key, val), ) -> b_arr[key].bΩp === b_arr[v[i]].bΩp, ((k, v) for (k, v) in d2[k]))
            push!(get!(Vector{typeof(i)}, d2[k], isnothing(key_or_nothing) ? v[i] : key_or_nothing), v[i])
        end
    end
    return d2
end

function solve_and_integrate_nonadjoint!(res, b_arr::Array{<:Rank1DiscretePNVector}, it::AbstractDiscretePNSolution)
    @assert all(_is_adjoint_vector, b_arr) && !_is_adjoint_solution(it)
    model = first(b_arr).model
    arch = first(b_arr).arch

    Δϵ = step(energy_model(model))

    T = base_type(arch)
    (_, (nxp, _), (nΩp, _)) = n_basis(model)
    buf = allocate_vec(arch, nΩp)

    idx_order = ideal_index_order(b_arr)

    for (idx, ψ) in it
        ψp = pview(ψ, model)

        for (x_base, x_rem) in idx_order
            mul!(transpose(buf), transpose(b_arr[x_base].bxp), ψp)
            for (Ω_base, Ωx_rem) in x_rem
                bufbuf = dot(b_arr[Ω_base].bΩp, buf)
                for i in Ωx_rem
                    res[i] += Δϵ * b_arr[i].bϵ[idx] * bufbuf
                end
            end
        end

        # for i in eachindex(b_arr) # this could be made more efficient by reusing unique (=== !) bxp, bΩp etc.
        #     res[i] += Δϵ * b_arr[i].bϵ[idx]*dot_buf(b_arr[i].bxp, ψp, b_arr[i].bΩp, buf)   
        # end
    end
    return res
end

function solve_and_integrate_adjoint!(res, b_arr::Array{<:Rank1DiscretePNVector}, it::AbstractDiscretePNSolution)
    @assert all(!_is_adjoint_vector, b_arr) && _is_adjoint_solution(it)
    model = first(b_arr).model
    arch = first(b_arr).arch

    Δϵ = step(energy_model(model))

    T = base_type(arch)
    (_, (nxp, _), (nΩp, _)) = n_basis(model)
    buf = allocate_vec(arch, nΩp)

    idx_order = ideal_index_order(b_arr)

    for (idx, ψ) in it
        if is_first(idx) continue end # (where ψp is initialized to 0 anyways..)
        ψp = pview(ψ, model)

        for (x_base, x_rem) in idx_order
            mul!(buf', b_arr[x_base].bxp', ψp)
            for (Ω_base, Ωx_rem) in x_rem
                bufbuf = dot(b_arr[Ω_base].bΩp, buf)
                for i in Ωx_rem
                    res[i] += Δϵ * T(0.5) * (b_arr[i].bϵ[plus½(idx)] + b_arr[i].bϵ[minus½(idx)])*bufbuf
                end
            end
        end

        # for i in eachindex(b_arr) # this could be made more efficient by reusing unique (=== !) bxp, bΩp etc.
        #     res[i] += Δϵ * T(0.5) * (b_arr[i].bϵ[plus½(idx)] + b_arr[i].bϵ[minus½(idx)])*dot_buf(b_arr[i].bxp, ψp, b_arr[i].bΩp, buf)
        # end  
    end
    return res
end

function assemble_rhs!(b, rhs::WeightedArrayOfRank1PNVector, idx, Δ, sym)
    @assert idx.adjoint == true
    fill!(b, zero(eltype(b)))

    nLp = length(first(rhs.arr_r1).bxp)
    nRp = length(first(rhs.arr_r1).bΩp)

    bp = reshape(@view(b[1:nLp*nRp]), (nLp, nRp))
    for i in eachindex(rhs.weights, rhs.arr_r1)
        bϵ2 = rhs.arr_r1[i].bϵ[plus½(idx)]
        bxpi_mat = reshape(rhs.arr_r1[i].bxp, (length(rhs.arr_r1[i].bxp), 1))
        bΩpi_mat = reshape(rhs.arr_r1[i].bΩp, (1, length(rhs.arr_r1[i].bΩp)))
        mul!(bp, bxpi_mat, bΩpi_mat, rhs.weights[i]*bϵ2*Δ, true)
    end

    # sym is not used since we only assemble the p part of b here
end

# update the rhs for the nonadjoint step from i to i-1 (at the midpoint of the energy interval)
function assemble_rhs_midpoint!(b, rhs::WeightedArrayOfRank1PNVector, idx, Δ, sym)
    @assert idx.adjoint == false
    fill!(b, zero(eltype(b)))

    nLp = length(first(rhs.arr_r1).bxp)
    nRp = length(first(rhs.arr_r1).bΩp)

    bp = reshape(@view(b[1:nLp*nRp]), (nLp, nRp))
    for i in eachindex(rhs.weights, rhs.arr_r1)
        bϵ2 = 0.5*(rhs.arr_r1[i].bϵ[idx] + rhs.arr_r1[i].bϵ[minus1(idx)])
        # bϵ2 = rhs.arr_r1[i].bϵ[i_ϵ]
        bxpi_mat = reshape(rhs.arr_r1[i].bxp, (length(rhs.arr_r1[i].bxp), 1))
        bΩpi_mat = reshape(rhs.arr_r1[i].bΩp, (1, length(rhs.arr_r1[i].bΩp)))
        mul!(bp, bxpi_mat, bΩpi_mat, rhs.weights[i]*bϵ2*Δ, true)
    end
    
    # sym is not used since we only assemble the p part of b here
end

function assemble_rhs!(b, rhs::Rank1DiscretePNVector, idx, Δ, sym)
    @assert idx.adjoint == true
    fill!(b, zero(eltype(b)))

    nLp = length(rhs.bxp)
    nRp = length(rhs.bΩp)

    bp = reshape(@view(b[1:nLp*nRp]), (nLp, nRp))
    # bp = pview(b, rhs.model)

    bϵ2 = rhs.bϵ[plus½(idx)]
    bxp_mat = reshape(@view(rhs.bxp[:]), (length(rhs.bxp), 1))
    bΩp_mat = reshape(@view(rhs.bΩp[:]), (1, length(rhs.bΩp)))
    mul!(bp, bxp_mat, bΩp_mat, bϵ2*Δ, true)
    # sym is not used since we only assemble the p part of b here
end

# update the rhs for the nonadjoint step from i to i-1 (at the midpoint of the energy interval)
function assemble_rhs_midpoint!(b, rhs::Rank1DiscretePNVector, idx, Δ, sym)
    @assert idx.adjoint == false
    fill!(b, zero(eltype(b)))

    nLp = length(rhs.bxp)
    nRp = length(rhs.bΩp)

    bp = reshape(@view(b[1:nLp*nRp]), (nLp, nRp))
    # bp = pview(b, rhs.model)

    bϵ2 = 0.5*(rhs.bϵ[idx] + rhs.bϵ[minus1(idx)])
    bxp_mat = reshape(@view(rhs.bxp[:]), (length(rhs.bxp), 1))
    bΩp_mat = reshape(@view(rhs.bΩp[:]), (1, length(rhs.bΩp)))
    mul!(bp, bxp_mat, bΩp_mat, bϵ2*Δ, true)
    # sym is not used since we only assemble the p part of b here
end