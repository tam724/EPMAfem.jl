@concrete struct Rank1DiscretePNVector{co} <: AbstractDiscretePNVector{co}
    model
    arch
    bϵ

    # might be moved to gpu
    bxp
    bΩp
end

@concrete struct DiscretePNVector{co} <: AbstractDiscretePNVector{co}
    model
    arch

    weights
    bϵ

    #(might be moved to gpu)
    bxp
    bΩp
end

@concrete struct ArrayOfRank1DiscretePNVector{co} <: AbstractDiscretePNVector{co}
    model
    arch
    bϵs

    # might be moved to gpu
    bxps
    bΩps
end

function Base.size(arr_r1::ArrayOfRank1DiscretePNVector)
    return length(arr_r1.bϵs), length(arr_r1.bxps), length(arr_r1.bΩps)
end

function Base.getindex(arr_r1::ArrayOfRank1DiscretePNVector{co}, i, j, k) where co
    return Rank1DiscretePNVector{co}(arr_r1.model, arr_r1.arch, arr_r1.bϵs[i], arr_r1.bxps[j], arr_r1.bΩps[k])
end

function weight_array_of_r1(weights, arr_r1::ArrayOfRank1DiscretePNVector{co}) where co
    return DiscretePNVector{co}(arr_r1.model, arr_r1.arch, weights, arr_r1.bϵs, arr_r1.bxps, arr_r1.bΩps)
end
@concrete struct VecOfRank1DiscretePNVector{co} <: AbstractDiscretePNVector{co}
    model
    arch

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
    return Rank1DiscretePNVector{co}(vec.model, vec.arch, vec.bϵs[i], vec.bxps[i], vec.bΩps[i])
end 

function (b::Rank1DiscretePNVector{true})(it::NonAdjointIterator)
    Δϵ = step(energy_model(b.model))

    T = base_type(b.arch)
    buf = allocate_vec(b.arch, length(b.bxp))
    integral = zero(T)

    for (ϵ, i_ϵ) in it
        ψp = pview(current_solution(it.solver), it.system.model)
        integral += Δϵ * b.bϵ[i_ϵ]*dot_buf(b.bxp, ψp, b.bΩp, buf)   
    end
    return integral
end

function (b::ArrayOfRank1DiscretePNVector{true})(it::NonAdjointIterator)
    Δϵ = step(energy_model(b.model))

    T = base_type(b.arch)
    buf = allocate_vec(b.arch, length(b.bxps |> first))
    integral = zeros(T, size(b))

    for (ϵ, i_ϵ) in it
        ψp = pview(current_solution(it.solver), it.system.model)

        for i in 1:length(b.bϵs)
            for j in 1:length(b.bxps)
                for k in 1:length(b.bΩps)
                    integral[i, j, k] += Δϵ * b.bϵs[i][i_ϵ]*dot_buf(b.bxps[j], ψp, b.bΩps[k], buf)   
                end
            end
        end
    end
    return integral
end

function (b::VecOfRank1DiscretePNVector{true})(it::NonAdjointIterator)
    Δϵ = step(energy_model(b.model))

    T = base_type(b.arch)
    buf = allocate_vec(b.arch, length(b.bxps |> first))
    integral = zeros(T, size(b))

    for (ϵ, i_ϵ) in it
        # @show i_ϵ
        ψp = pview(current_solution(it.solver), it.system.model)

        for i in 1:length(b.bϵs)
            integral[i] += Δϵ * b.bϵs[i][i_ϵ]*dot_buf(b.bxps[i], ψp, b.bΩps[i], buf)   
        end
    end
    return integral
end

function (b::Rank1DiscretePNVector{false})(it::AdjointIterator)
    Δϵ = step(energy_model(b.model))

    T = base_type(b.arch)
    buf = allocate_vec(b.arch, length(b.bxp))
    integral = zero(T)

    for (ϵ, i_ϵ) in it
        ψp = pview(current_solution(it.solver), it.system.model)

        if i_ϵ != 1 # (where ψp is initialized to 0 anyways..)
            integral += Δϵ * T(0.5) * (b.bϵ[i_ϵ] + b.bϵ[i_ϵ-1])*dot_buf(b.bxp, ψp, b.bΩp, buf)
        end  
    end
    return integral
end

function (b::ArrayOfRank1DiscretePNVector{false})(it::AdjointIterator)
    Δϵ = step(energy_model(b.model))

    T = base_type(b.arch)
    buf = allocate_vec(b.arch, length(b.bxps |> first))
    integral = zeros(T, size(b))

    for (ϵ, i_ϵ) in it
        ψp = pview(current_solution(it.solver), it.system.model)

        for i in 1:length(b.bϵs)
            for j in 1:length(b.bxps)
                for k in 1:length(b.bΩps)
                    if i_ϵ != 1 # (where ψp is initialized to 0 anyways..)
                        integral[i, j, k] += Δϵ * 0.5 * (b.bϵs[i][i_ϵ] + b.bϵs[i][i_ϵ-1])*dot_buf(b.bxps[j], ψp, b.bΩps[k], buf)
                    end  
                end
            end
        end
    end
    return integral
end

function (b::VecOfRank1DiscretePNVector{false})(it::AdjointIterator)
    Δϵ = step(energy_model(b.model))

    T = base_type(b.arch)
    buf = allocate_vec(b.arch, length(b.bxps |> first))
    integral = zeros(T, size(b))

    for (ϵ, i_ϵ) in it
        ψp = pview(current_solution(it.solver), it.system.model)

        for i in 1:length(b.bϵs)
            if i_ϵ != 1 # (where ψp is initialized to 0 anyways..)
                integral[i] += Δϵ * T(0.5) * (b.bϵs[i][i_ϵ] + b.bϵs[i][i_ϵ-1])*dot_buf(b.bxps[i], ψp, b.bΩps[i], buf)
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
                mul!(bp, bxpi_mat, bΩpi_mat, rhs.weights[ϵi, xi, Ωi]*bϵ2*Δ, true)
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
                mul!(bp, bxpi_mat, bΩpi_mat, rhs.weights[ϵi, xi, Ωi]*bϵ2*Δ, true)
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
    mul!(bp, bxp_mat, bΩp_mat, bϵ2*Δ, true)
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
    mul!(bp, bxp_mat, bΩp_mat, bϵ2*Δ, true)
end

@concrete struct ArrayOfTangentDiscretePNVector <: AbstractDiscretePNVector{true}
    ρp_tangent
    ρm_tangent

    cached_solution
end

function tangent(it::AdjointIterator)
    cached = saveall(it)

    ρp_tangent = it.system.ρp_tens.skeleton |> it.system.arch
    ρm_tangent = it.system.ρm_tens.skeleton |> it.system.arch
    return ArrayOfTangentDiscretePNVector(ρp_tangent, ρm_tangent, cached)
end

function Base.getindex(arr::ArrayOfTangentDiscretePNVector, i_e, i_x, Δ=true)
    discrete_system = arr.cached_solution.it.system
    # T = base_type(architecture(discrete_system.model))
    # VT = vec_type(architecture(discrete_system.model))

    # cv(x) = convert_to_architecture(architecture(discrete_system.model), x)

    onehot = zeros(num_free_dofs(SpaceModels.material(space_model(discrete_system.model))))
    onehot[i_x] = Δ
    Sparse3Tensor.project!(discrete_system.ρp_tens, onehot)
    Sparse3Tensor.project!(discrete_system.ρm_tens, onehot)

    copyto!(nonzeros(arr.ρp_tangent), nonzeros(discrete_system.ρp_tens.skeleton))
    copyto!(nonzeros(arr.ρm_tangent), nonzeros(discrete_system.ρm_tens.skeleton))
    return TangentDiscretePNVector(arr, i_x, i_e)
    # return DiscretePNVector(arr.ρp_tangent[i], arr.ρm_tangent[i], arr.cached_solution)
end

function (arr::ArrayOfTangentDiscretePNVector)(it::NonAdjointIterator)
    discrete_system = arr.cached_solution.it.system
    arch = discrete_system.arch

    T = base_type(arch)
    # cv_Int(x) = convert_to_architecture(Int64, architecture(discrete_system.model), x)

    Δϵ = T(step(energy_model(discrete_system.model)))

    isp, jsp = arch.(Int64, Sparse3Tensor.get_ijs(discrete_system.ρp_tens))
    ism, jsm = arch.(Int64, Sparse3Tensor.get_ijs(discrete_system.ρm_tens))

    (nϵ, (nxp, nxm), (nΩp, nΩm)) = n_basis(discrete_system.model)

    Λtemp = allocate_vec(arch, max(nxp*nΩp, nxm*nΩm))
    Λtempp = reshape(@view(Λtemp[1:nxp*nΩp]), (nxp, nΩp))
    Λtempm = reshape(@view(Λtemp[1:nxm*nΩm]), (nxm, nΩm))

    σtemp = allocate_vec(arch, max(nΩp, nΩm))
    σtempp = Diagonal(@view(σtemp[1:nΩp]))
    σtempm = Diagonal(@view(σtemp[1:nΩm]))

    ΛpΦp = [allocate_vec(arch, length(isp)) for _ in 1:length(discrete_system.ρp)]
    ΛmΦm = [allocate_vec(arch, length(ism)) for _ in 1:length(discrete_system.ρp)]

    skip_initial = true
    write_initial = false
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

            # σp_i = similar(discrete_system.kp[i_e][1])
            rmul!(σtempp.diag, false)
            for i in 1:size(discrete_system.σ, 2)
                σtempp.diag .+= discrete_system.σ[i_e, i, i_ϵ] .* discrete_system.kp[i_e][i].diag
            end

            mul!(Λtempp, Λ_ip2p, σtempp, -T(0.5), false)
            mul!(Λtempp, Λ_im2p, σtempp, -T(0.5), true)
            Λtempp .+= (s_i / Δϵ + T(0.5) * τ_i) .* Λ_ip2p .+ (-s_i / Δϵ + T(0.5) * τ_i) .* Λ_im2p

            Sparse3Tensor.special_matmul!(ΛpΦp[i_e], isp, jsp, Λtempp, Φp, Δϵ, write_initial)

            # σm_i = similar(discrete_system.km[i_e][1])
            rmul!(σtempm.diag, false)
            for i in 1:size(discrete_system.σ, 2)
                σtempm.diag .+= discrete_system.σ[i_e, i, i_ϵ] .* discrete_system.km[i_e][i].diag
            end

            mul!(Λtempm, Λ_ip2m, σtempm, -T(0.5), false)
            mul!(Λtempm, Λ_im2m, σtempm, -T(0.5), true)
            Λtempm .+= (s_i / Δϵ + T(0.5) * τ_i) .* Λ_ip2m .+ (-s_i / Δϵ + T(0.5) * τ_i) .* Λ_im2m

            Sparse3Tensor.special_matmul!(ΛmΦm[i_e], ism, jsm, Λtempm, Φm, Δϵ, write_initial)
        end
        write_initial = true
    end
    ρs_adjoint = [zeros(num_free_dofs(SpaceModels.material(space_model(discrete_system.model)))) for _ in 1:length(discrete_system.ρp)]
    for i_e in 1:length(discrete_system.ρp)
        Sparse3Tensor.contract!(ρs_adjoint[i_e], discrete_system.ρp_tens, ΛpΦp[i_e] |> collect, true, true)
        Sparse3Tensor.contract!(ρs_adjoint[i_e], discrete_system.ρm_tens, ΛmΦm[i_e] |> collect, true, true)
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
    arch = discrete_system.arch

    T = base_type(arch)
    Δϵ = T(step(energy_model(discrete_system.model)))

    fill!(b, zero(eltype(b)))

    (nϵ, (nxp, nxm), (nΩp, nΩm)) = n_basis(discrete_system.model)
    np = nxp*nΩp
    nm = nxm*nΩm

    bp = @view(b[1:np])
    bm = @view(b[np+1:np+nm])

    si = discrete_system.s[rhs.i_e, i]
    τi = discrete_system.τ[rhs.i_e, i]
    σi = discrete_system.σ[rhs.i_e, :, i]

    Λ_im2 = rhs.parent.cached_solution[i]
    Λ_ip2 = rhs.parent.cached_solution[i+1]

    #TODO: move temporary allocations somwhere else
    tmp = allocate_vec(arch, max(np, nm))
    tmp2 = allocate_vec(arch, max(nΩp, nΩm))

    a = [(si/Δϵ + τi*0.5)]
    c = [(-σi.*0.5)]
    γ = sym ? -1 : 1

    mul!(bp, ZMatrix([rhs.parent.ρp_tangent], discrete_system.Ip, [discrete_system.kp[rhs.i_e]], a, c, mat_view(tmp, nxp, nΩp), Diagonal(@view(tmp2[1:nΩp]))), @view(Λ_ip2[1:np]), Δ, false)
    mul!(bm, ZMatrix([rhs.parent.ρm_tangent], discrete_system.Im, [discrete_system.km[rhs.i_e]], a, c, mat_view(tmp, nxm, nΩm), Diagonal(@view(tmp2[1:nΩm]))), @view(Λ_ip2[np+1:np+nm]), γ*Δ, false)

    a = [(-si/Δϵ + τi*0.5)]
    c = [(-σi.*0.5)]
    mul!(bp, ZMatrix([rhs.parent.ρp_tangent], discrete_system.Ip, [discrete_system.kp[rhs.i_e]], a, c, mat_view(tmp, nxp, nΩp), Diagonal(@view(tmp2[1:nΩp]))), @view(Λ_im2[1:np]), Δ, true)
    mul!(bm, ZMatrix([rhs.parent.ρm_tangent], discrete_system.Im, [discrete_system.km[rhs.i_e]], a, c, mat_view(tmp, nxm, nΩm), Diagonal(@view(tmp2[1:nΩm]))), @view(Λ_im2[np+1:np+nm]), γ*Δ, true)
end
