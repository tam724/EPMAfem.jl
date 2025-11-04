@concrete terse struct TangentDiscretePNVector{U} <: AbstractDiscretePNVector
    adjoint::Bool
    updatable_problem_or_vector::U
    parameters
    cached_solution
    parameter_index
end

function tangent(upd_problem::UpdatableDiscretePNProblem, ψ::AbstractDiscretePNSolution, ρs)
    # this basically creates the "PNVectors" \dot{a}(\cdot{}, ψ) or \dot{a}(ψ, \cdot{})
    # if ψ is an adjoint solution, this is an nonadjoint vector, is ψ is a nonadjoint solution, this is an adjoint vector 
    (n_e, n_cells) = n_parameters(upd_problem)
    return [TangentDiscretePNVector(_is_adjoint_solution(ψ), upd_problem, ρs, ψ, (i, j)) for i in 1:n_e, j in 1:n_cells]
end

_is_adjoint_vector(b::TangentDiscretePNVector) = b.adjoint

function initialize_integration(b::Array{<:TangentDiscretePNVector{<:UpdatableDiscretePNProblem}})
    if !allunique(b) @warn "Duplicate TangentDiscretePNVector instances detected in the input array. Computed tangents are aliased" end
    upd_problem = first(b).updatable_problem_or_vector
    problem = upd_problem.problem
    arch = problem.arch
    T = base_type(arch)
    (nϵ, (nxp, nxm), (nΩp, nΩm)) = n_basis(problem.model)
    (; nd, ne, nσ) = n_sums(problem)

    isp, jsp = arch.(Int64, Sparse3Tensor.get_ijs(upd_problem.ρp_tens))
    ism, jsm = arch.(Int64, Sparse3Tensor.get_ijs(upd_problem.ρm_tens))
    Λtemp = allocate_vec(arch, max(nxp*nΩp, nxm*nΩm))
    σtemp = allocate_vec(arch, max(nΩp, nΩm))
    ΛpΦp = [allocate_vec(arch, length(isp)) for _ in 1:ne]
    ΛmΦm = [allocate_vec(arch, length(ism)) for _ in 1:ne]
    for ΛpΦpi in ΛpΦp
        fill!(ΛpΦpi, zero(eltype(ΛpΦpi)))
    end
    for ΛmΦmi in ΛmΦm
        fill!(ΛmΦmi, zero(eltype(ΛmΦmi)))
    end
    cache = (; isp, jsp, ism, jsm, Λtemp, σtemp, ΛpΦp, ΛmΦm)
    return PNVectorIntegrator(b, cache)
end

function finalize_integration((; b, cache)::PNVectorIntegrator{<:Array{<:TangentDiscretePNVector{<:UpdatableDiscretePNProblem}}})
    upd_problem = first(b).updatable_problem_or_vector
    problem = upd_problem.problem
    (; ΛpΦp, ΛmΦm) = cache
    (; nd, ne, nσ) = n_sums(problem)

    # first we collect the adjoint for all parameters
    ρs_adjoint = zeros(n_parameters(upd_problem))
    for i_e in 1:ne
        Sparse3Tensor.contract!(@view(ρs_adjoint[i_e, :]), upd_problem.ρp_tens, ΛpΦp[i_e] |> collect, true, true)
        Sparse3Tensor.contract!(@view(ρs_adjoint[i_e, :]), upd_problem.ρm_tens, ΛmΦm[i_e] |> collect, true, true)
    end
    integral = zeros(size(b))
    for i in eachindex(b)
        integral[i] = ρs_adjoint[b[i].parameter_index |> CartesianIndex]
    end
    return integral
end

function ((; b, cache)::PNVectorIntegrator{<:Array{<:TangentDiscretePNVector{<:UpdatableDiscretePNProblem}}})(idx, ψ)
    if !_is_adjoint_vector(b)
        throw(ErrorException("not implemented yet"))
    end
    (; isp, jsp, ism, jsm, Λtemp, σtemp, ΛpΦp, ΛmΦm) = cache

    upd_problem = first(b).updatable_problem_or_vector
    problem = upd_problem.problem
    Ip, Im, kp, km, absΩp, Ωpm = direction_matrices(problem)

    T = base_type(architecture(problem))

    Δϵ = T(step(energy_model(problem.model)))

    (nϵ, (nxp, nxm), (nΩp, nΩm)) = n_basis(problem.model)
    (; nd, ne, nσ) = n_sums(problem)

    Λtempp = reshape(@view(Λtemp[1:nxp*nΩp]), (nxp, nΩp))
    Λtempm = reshape(@view(Λtemp[1:nxm*nΩm]), (nxm, nΩm))
    σtempp = Diagonal(@view(σtemp[1:nΩp]))
    σtempm = Diagonal(@view(σtemp[1:nΩm]))

    Φp, Φm = pmview(ψ, problem.model)
    Λ = first(b).cached_solution
    Λ_im2p, Λ_im2m = pmview(Λ[minus½(idx)], problem.model)
    Λ_ip2p, Λ_ip2m = pmview(Λ[plus½(idx)], problem.model)

    for i_e in 1:ne
        s_i = problem.s[i_e, idx]
        τ_i = problem.τ[i_e, idx]

        σtempp.diag .= τ_i
        for i in 1:size(problem.σ, 2)
            σtempp.diag .-= problem.σ[i_e, i, idx] .* kp[i_e][i].diag
        end

        mul!(Λtempp, Λ_ip2p, σtempp, T(0.5), false)
        mul!(Λtempp, Λ_im2p, σtempp, T(0.5), true)
        Λtempp .+= (s_i / Δϵ) .* (Λ_ip2p .- Λ_im2p)

        Sparse3Tensor.special_matmul!(ΛpΦp[i_e], isp, jsp, Λtempp, Φp, Δϵ, true)

        σtempm.diag .= τ_i
        for i in 1:size(problem.σ, 2)
            σtempm.diag .-= problem.σ[i_e, i, idx] .* km[i_e][i].diag
        end

        mul!(Λtempm, Λ_ip2m, σtempm, T(0.5), false)
        mul!(Λtempm, Λ_im2m, σtempm, T(0.5), true)
        Λtempm .+= (s_i / Δϵ) .* (Λ_ip2m .- Λ_im2m)

        Sparse3Tensor.special_matmul!(ΛmΦm[i_e], ism, jsm, Λtempm, Φm, Δϵ, true)
    end
end

# assembly
function initialize_assembly(b::TangentDiscretePNVector{<:UpdatableDiscretePNProblem})
    upd_problem = b.updatable_problem_or_vector
    problem = upd_problem.problem
    (n_elem, n_cells) = n_parameters(upd_problem)

    arch = architecture(problem)
    T = base_type(arch)
    ρp_tangent = [similar(upd_problem.ρp_tens.skeleton) |> arch for _ in 1:n_elem]
    ρm_tangent = [similar(upd_problem.ρm_tens.skeleton) |> arch for _ in 1:n_elem]

    onehot = zeros(n_cells)
    element_index, cell_index = b.parameter_index
    for i in 1:n_elem
        if i == element_index onehot[cell_index] = 1.0 end
        Sparse3Tensor.project!(upd_problem.ρp_tens, onehot)
        Sparse3Tensor.project!(upd_problem.ρm_tens, onehot)
        copyto!(nonzeros(ρp_tangent[i]), nonzeros(upd_problem.ρp_tens.skeleton))
        copyto!(nonzeros(ρm_tangent[i]), nonzeros(upd_problem.ρm_tens.skeleton))
        if i == element_index onehot[cell_index] = 0.0 end
    end

    (_, (nxp, nxm), (nΩp, nΩm)) = n_basis(problem.model)
    np, nm = nxp*nΩp, nxm*nΩm
    tmp = allocate_vec(arch, max(np, nm))
    tmp2 = allocate_vec(arch, max(nΩp, nΩm))

    (nd, ne, nσ) = n_sums(problem)
    a = Vector{T}(undef, ne)
    c = [Vector{T}(undef, nσ) for _ in 1:ne]

    cache = (; ρp_tangent, ρm_tangent, a, c, tmp, tmp2)

    return PNVectorAssembler(b, cache)
end

function assemble_at!(rhs, (; b, cache)::PNVectorAssembler{<:TangentDiscretePNVector{<:UpdatableDiscretePNProblem}}, idx, Δ, sym, β=false)
    if _is_adjoint_solution(b.cached_solution) != true
        throw(ErrorException("not implemented yet"))
    end
    upd_problem = b.updatable_problem_or_vector
    problem = upd_problem.problem

    T = base_type(architecture(problem))
    Δϵ = T(step(energy_model(problem.model)))

    rhsp, rhsm = pmview(rhs, problem.model)
    
    # si = b.problem.s[rhs.i_e, idx]
    # τi = b.problem.τ[rhs.i_e, idx]
    # σi = @view(b.problem.σ[rhs.i_e, :, idx])

    (_, (nxp, nxm), (nΩp, nΩm)) = n_basis(problem.model)
    (nd, ne, nσ) = n_sums(problem)
    Ip, Im, kp, km, absΩp, Ωpm = direction_matrices(problem)


    for ie in 1:ne
        cache.a[ie] = problem.s[ie, idx]/Δϵ + problem.τ[ie, idx]*0.5
        for iσ in 1:nσ
            cache.c[ie][iσ] = -problem.σ[ie, iσ, idx]*0.5
        end
    end
    γ = sym ? -1 : 1

    Λp⁻½, Λm⁻½ = pmview(b.cached_solution[minus½(idx)], problem.model)
    Λp⁺½, Λm⁺½ = pmview(b.cached_solution[plus½(idx)], problem.model)

    mul!(@view(rhsp[:]), ZMatrix2(cache.ρp_tangent, Ip, kp, cache.a, cache.c, mat_view(cache.tmp, nxp, nΩp), Diagonal(@view(cache.tmp2[1:nΩp]))), @view(Λp⁺½[:]), Δ, β)
    mul!(@view(rhsm[:]), ZMatrix2(cache.ρm_tangent, Im, km, cache.a, cache.c, mat_view(cache.tmp, nxm, nΩm), Diagonal(@view(cache.tmp2[1:nΩm]))), @view(Λm⁺½[:]), γ*Δ, β)

    for ie in 1:ne
        cache.a[ie] = -problem.s[ie, idx]/Δϵ + problem.τ[ie, idx]*0.5
        for iσ in 1:nσ
            cache.c[ie][iσ] = -problem.σ[ie, iσ, idx]*0.5
        end
    end
    mul!(@view(rhsp[:]), ZMatrix2(cache.ρp_tangent, Ip, kp, cache.a, cache.c, mat_view(cache.tmp, nxp, nΩp), Diagonal(@view(cache.tmp2[1:nΩp]))), @view(Λp⁻½[:]), Δ, true)
    mul!(@view(rhsm[:]), ZMatrix2(cache.ρm_tangent, Im, km, cache.a, cache.c, mat_view(cache.tmp, nxm, nΩm), Diagonal(@view(cache.tmp2[1:nΩm]))), @view(Λm⁻½[:]), γ*Δ, true)
end

## NOW VECTOR
function tangent(upd_vector::UpdatableRank1DiscretePNVector, ρs)
    # this basically creates the "PNVectors" \dot{b}(ψ)
    (n_e, n_cells) = n_parameters(upd_vector)
    return [TangentDiscretePNVector(_is_adjoint_vector(upd_vector.vector), upd_vector, ρs, nothing, (i, j))  for i in 1:n_e, j in 1:n_cells]
end

function initialize_integration(b::TangentDiscretePNVector{<:UpdatableRank1DiscretePNVector})
    arch = b.updatable_problem_or_vector.vector.arch
    bxp_adjoint = allocate_vec(arch, length(b.updatable_problem_or_vector.vector.bxp))
    fill!(bxp_adjoint, zero(eltype(bxp_adjoint)))
    cache = (bxp_adjoint = bxp_adjoint, )
    return PNVectorIntegrator(b, cache)
end

function finalize_integration((; b, cache)::PNVectorIntegrator{<:TangentDiscretePNVector{<:UpdatableRank1DiscretePNVector}})
    if b.parameter_index[1] == b.updatable_problem_or_vector.element_index
        # this is inefficient.. 
        ρ_adjoint = zeros(n_parameters(b.updatable_problem_or_vector))
        update_bxp_adjoint!(ρ_adjoint, b.updatable_problem_or_vector.bxp_updater, cache.bxp_adjoint, b.parameters)
        return ρ_adjoint[b.parameter_index...]
    else
        return 0.0
    end
end

function ((; b, cache)::PNVectorIntegrator{<:TangentDiscretePNVector{<:UpdatableRank1DiscretePNVector}})(idx, ψ)
    upd_vector = b.updatable_problem_or_vector
    vector = upd_vector.vector
    model = vector.model
    ψp = pview(ψ, model)
    Δϵ = step(energy_model(model))
    T = base_type(vector.arch)
    if idx.adjoint
        @assert !_is_adjoint_vector(vector)
        bϵ2 = T(0.5) * (vector.bϵ[minus½(idx)] + vector.bϵ[plus½(idx)])
    else
        @assert _is_adjoint_vector(b)
        bϵ2 = vector.bϵ[idx]
    end
    mul!(cache.bxp_adjoint, ψp, vector.bΩ.p, Δϵ * bϵ2, true)
end

## for array valued integrations

function initialize_integration(b::Array{<:TangentDiscretePNVector{<:UpdatableRank1DiscretePNVector}})
    if !allunique(b) @warn "Duplicate TangentDiscretePNVector instances detected in the input array. Computed tangents are aliased" end
    arch = first(b).updatable_problem_or_vector.vector.arch
    idx_order = ideal_index_order([bi.updatable_problem_or_vector.vector for bi in b])
    bxp_adjoints = Dict((i => allocate_vec(arch, length(first(b).updatable_problem_or_vector.vector.bxp)) for i in keys(idx_order)))
    for (i, bxp_adjoint) in bxp_adjoints
        fill!(bxp_adjoint, zero(eltype(bxp_adjoint)))
    end
    cache = (idx_order=idx_order, bxp_adjoints=bxp_adjoints)
    return PNVectorIntegrator(b, cache)
end

function finalize_integration((; b, cache)::PNVectorIntegrator{<:Array{<:TangentDiscretePNVector{<:UpdatableRank1DiscretePNVector}}})
    ρ_adjoints = zeros(n_parameters(first(b).updatable_problem_or_vector))
    for (x_base, bxp_adjoint) in cache.bxp_adjoints
        update_bxp_adjoint!(ρ_adjoints, b[x_base].updatable_problem_or_vector.bxp_updater, bxp_adjoint, b[x_base].parameters)
    end
    # this feels silly..
    integral = zeros(size(b))
    for (i, b) in enumerate(b)
        integral[i] += ρ_adjoints[b.parameter_index...]
    end
    return integral
end

function ((; b, cache)::PNVectorIntegrator{<:Array{<:TangentDiscretePNVector{<:UpdatableRank1DiscretePNVector}}})(idx, ψ)
    model = first(b).updatable_problem_or_vector.vector.model
    arch = first(b).updatable_problem_or_vector.vector.arch
    ψp = pview(ψ, model)
    Δϵ = step(energy_model(model))
    T = base_type(arch)
    for (x_base, x_rem) in cache.idx_order
        bpx_adjoint = cache.bxp_adjoints[x_base]
        for (Ω_base, Ωx_rem) in x_rem
            bΩp_i = b[Ω_base].updatable_problem_or_vector.vector.bΩp
            for (ϵ_base, ϵΩx_rem) in Ωx_rem
                bϵ_i = b[ϵ_base].updatable_problem_or_vector.vector.bϵ
                if idx.adjoint
                    @assert !_is_adjoint_vector(b[ϵ_base])
                    Δbϵ_i = Δϵ * T(0.5) * (bϵ_i[minus½(idx)] + bϵ_i[plus½(idx)])
                else
                    @assert _is_adjoint_vector(b[ϵ_base])
                    Δbϵ_i = Δϵ * bϵ_i[idx]
                end
                # the remaining vectors share the same bxp, bΩp and bϵ -> collectively compute all tangents (looping over tangent indices happens in finalize)
                mul!(bpx_adjoint, ψp, bΩp_i, Δbϵ_i, true)
                # for i in ϵΩx_rem ... end
            end
        end
    end
end
