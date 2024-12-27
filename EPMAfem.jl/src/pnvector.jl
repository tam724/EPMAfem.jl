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

function assemble_rhs!(b, rhs::DiscretePNVector{true}, i, Δ, sym; bxp=rhs.bxp, bΩp=rhs.bΩp)
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
    # sym is not used since we only assemble the p part of b here
end

function assemble_rhs_midpoint!(b, rhs::DiscretePNVector{false}, i, Δ, sym; bxp=rhs.bxp, bΩp=rhs.bΩp)
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
    # sym is not used since we only assemble the p part of b here
end

function assemble_rhs!(b, rhs::Rank1DiscretePNVector{true}, i, Δ, sym; bxp=rhs.bxp, bΩp=rhs.bΩp)
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

function assemble_rhs_midpoint!(b, rhs::Rank1DiscretePNVector{false}, i, Δ, sym; bxp=rhs.bxp, bΩp=rhs.bΩp)
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

@concrete struct ArrayOfTangentDiscretePNVector <: AbstractDiscretePNVector{true}
    ρp_tangent
    ρm_tangent

    cached_solution
end

function tangent(it::AdjointIterator)
    cached = saveall(it)

    ρp_tangent = copy(it.system.ρp_tens.skeleton)
    ρm_tangent = copy(it.system.ρm_tens.skeleton)
    return ArrayOfTangentDiscretePNVector(ρp_tangent, ρm_tangent, cached)
end

function Base.getindex(arr::ArrayOfTangentDiscretePNVector, i_e, i_x)
    discrete_system = arr.cached_solution.it.system
    onehot = zeros(num_free_dofs(SpaceModels.material(space(discrete_system.model))))
    onehot[i_x] = 1.0
    Sparse3Tensor._project!(arr.ρp_tangent, discrete_system.ρp_tens, onehot)
    Sparse3Tensor._project!(arr.ρm_tangent, discrete_system.ρm_tens, onehot)
    return TangentDiscretePNVector(arr, i_x, i_e)
    # return DiscretePNVector(arr.ρp_tangent[i], arr.ρm_tangent[i], arr.cached_solution)
end

function (arr::ArrayOfTangentDiscretePNVector)(it::NonAdjointIterator)
    discrete_system = arr.cached_solution.it.system
    Δϵ = step(energy(discrete_system.model))

    ρs_adjoint = [zeros(num_free_dofs(SpaceModels.material(space(discrete_system.model)))) for _ in 1:length(discrete_system.ρp)]
    skip_initial = true
    for (ϵ, i_ϵ) in it
        if skip_initial
            skip_initial = false
            continue
        end
        Φp = pview(current_solution(it.solver), discrete_system.model)
        Φm = mview(current_solution(it.solver), discrete_system.model)

        Λ_im2p = pview(arr.cached_solution[i_ϵ], discrete_system.model)
        Λ_im2m = mview(arr.cached_solution[i_ϵ], discrete_system.model)
        Λ_ip2p = pview(arr.cached_solution[i_ϵ+1], discrete_system.model)
        Λ_ip2m = mview(arr.cached_solution[i_ϵ+1], discrete_system.model)

        for i_e in 1:1:length(discrete_system.ρp)
            s_i = discrete_system.s[i_e, i_ϵ]
            τ_i = discrete_system.τ[i_e, i_ϵ]
            for i_m in 1:size(Λ_im2p, 2)
                σ_i = sum(discrete_system.σ[i_e, i, i_ϵ] * discrete_system.kp[i_e][i].diag[i_m] for i in 1:size(discrete_system.σ, 2))
                Λp = s_i / Δϵ * (Λ_ip2p[:, i_m] .- Λ_im2p[:, i_m]) .+ (τ_i - σ_i) * 0.5 * (Λ_ip2p[:, i_m] .+ Λ_im2p[:, i_m])
                Sparse3Tensor.contract!(ρs_adjoint[i_e], discrete_system.ρp_tens2, Λp, @view(Φp[:, i_m]), Δϵ, 1.0)
            end
            for i_m in 1:size(Λ_im2m, 2)
                σ_i = sum(discrete_system.σ[i_e, i, i_ϵ] * discrete_system.km[i_e][i].diag[i_m] for i in 1:size(discrete_system.σ, 2))
                Λm = s_i / Δϵ * (Λ_ip2m[:, i_m] .- Λ_im2m[:, i_m]) .+ (τ_i - σ_i) * 0.5 * (Λ_ip2m[:, i_m] .+ Λ_im2m[:, i_m])
                Sparse3Tensor.contract!(ρs_adjoint[i_e], discrete_system.ρm_tens2, Λm, @view(Φm[:, i_m]), Δϵ, 1.0)
            end
        end
    end
    return ρs_adjoint
end

@concrete struct TangentDiscretePNVector <: AbstractDiscretePNVector{true}
    parent
    i_x
    i_e
end

function assemble_rhs!(b, rhs::TangentDiscretePNVector, i, Δ, sym)
    discrete_system = rhs.parent.cached_solution.it.system
    Δϵ = step(discrete_system.model.energy_model)

    fill!(b, zero(eltype(b)))

    n_basis = number_of_basis_functions(discrete_system.model)
    nxp, nxm, nΩp, nΩm = n_basis.x.p, n_basis.x.m, n_basis.Ω.p, n_basis.Ω.m
    np = nxp*nΩp
    nm = nxm*nΩm

    bp = @view(b[1:np])
    bm = @view(b[np+1:np+nm])

    si = discrete_system.s[rhs.i_e, i]
    τi = discrete_system.τ[rhs.i_e, i]
    σi = discrete_system.σ[rhs.i_e, :, i]

    Λ_im2 = rhs.parent.cached_solution[i]
    Λ_ip2 = rhs.parent.cached_solution[i+1]

    tmp = zeros(max(np, nm))
    tmp2 = zeros(max(nΩp, nΩm))

    a = [(si/Δϵ + τi*0.5)]
    c = [(-σi.*0.5)]
    γ = sym ? -1 : 1

    mul!(bp, ZMatrix([rhs.parent.ρp_tangent], discrete_system.Ip, [discrete_system.kp[rhs.i_e]], a, c, mat_view(tmp, nxp, nΩp), Diagonal(@view(tmp2[1:nΩp]))), @view(Λ_ip2[1:np]), Δ, 0.0)
    mul!(bm, ZMatrix([rhs.parent.ρm_tangent], discrete_system.Im, [discrete_system.km[rhs.i_e]], a, c, mat_view(tmp, nxm, nΩm), Diagonal(@view(tmp2[1:nΩm]))), @view(Λ_ip2[np+1:np+nm]), γ*Δ, 0.0)

    a = [(-si/Δϵ + τi*0.5)]
    c = [(-σi.*0.5)]
    mul!(bp, ZMatrix([rhs.parent.ρp_tangent], discrete_system.Ip, [discrete_system.kp[rhs.i_e]], a, c, mat_view(tmp, nxp, nΩp), Diagonal(@view(tmp2[1:nΩp]))), @view(Λ_im2[1:np]), Δ, 1.0)
    mul!(bm, ZMatrix([rhs.parent.ρm_tangent], discrete_system.Im, [discrete_system.km[rhs.i_e]], a, c, mat_view(tmp, nxm, nΩm), Diagonal(@view(tmp2[1:nΩm]))), @view(Λ_im2[np+1:np+nm]), γ*Δ, 1.0)
end
