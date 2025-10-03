@concrete terse struct PNProbe <: AbstractDiscretePNVector
    model
    arch
    x
    Ω
    ϵ
end

function PNProbe(model, arch; x=nothing, ϵ=nothing, Ω=nothing)
    return PNProbe(model, arch, x, Ω, ϵ)
end

function initialize_integration(b::PNProbe)
    (nϵ, (nxp, nxm), (nΩp, nΩm)) = n_basis(b.model)

    integral = (p=zeros(isnothing(b.x) ? nxp : 1, isnothing(b.Ω) ? nΩp : 1, isnothing(b.ϵ) ? nϵ : 1)|> b.arch,
                m=zeros(isnothing(b.x) ? nxm : 1, isnothing(b.Ω) ? nΩm : 1, isnothing(b.ϵ) ? nϵ : 1) |> b.arch)
    bases = (
         x = isnothing(b.x) ? nothing : SpaceModels.eval_basis(space_model(b.model), b.x) |> b.arch,
         Ω = isnothing(b.Ω) ? nothing : SphericalHarmonicsModels.eval_basis(direction_model(b.model), b.Ω) |> b.arch,
         ϵ = isnothing(b.ϵ) ? nothing : energy_eval_basis(energy_model(b.model), b.ϵ)
    )

    return PNVectorIntegrator(b, (integral=integral, bases=bases))
end

function ((; b, cache)::PNVectorIntegrator{<:PNProbe})(idx, ψ)
    ψp, ψm = pmview(ψ, b.model)
    Δϵ = step(energy_model(b.model))
    T = base_type(b.arch)
    bases = cache.bases

    if isnothing(b.x)
        if isnothing(b.Ω)
            if isnothing(b.ϵ)
                cache.integral.p[:, :, idx.i] .+= ψp 
                cache.integral.m[:, :, idx.i] .+= ψm
            else
                bϵ2 = idx.adjoint ? T(Δϵ*0.5) * (bases.ϵ[minus½(idx)] + bases.ϵ[plus½(idx)]) : T(Δϵ)*bases.ϵ[idx]
                cache.integral.p[:, :, 1] .+= bϵ2 .* ψp
                cache.integral.m[:, :, 1] .+= bϵ2 .* ψm
            end
        else
            if isnothing(b.ϵ)
                mul!(@view(cache.integral.p[:, 1, idx.i]), ψp, bases.Ω.p, true, true)
                mul!(@view(cache.integral.m[:, 1, idx.i]), ψm, bases.Ω.m, true, true)
            else
                bϵ2 = idx.adjoint ? T(Δϵ*0.5) * (bases.ϵ[minus½(idx)] + bases.ϵ[plus½(idx)]) : T(Δϵ)*bases.ϵ[idx]
                mul!(@view(cache.integral.p[:, 1, 1]), ψp, bases.Ω.p, bϵ2, true)
                mul!(@view(cache.integral.m[:, 1, 1]), ψm, bases.Ω.m, bϵ2, true)
            end
        end
    else
        if isnothing(b.Ω)
            if isnothing(b.ϵ)
                cache.integral.p[1:1, :, idx.i] .+= transpose(bases.x.p)*ψp
                cache.integral.m[1:1, :, idx.i] .+= transpose(bases.x.m)*ψm
            else
                bϵ2 = idx.adjoint ? T(Δϵ*0.5) * (bases.ϵ[minus½(idx)] + bases.ϵ[plus½(idx)]) : T(Δϵ)*bases.ϵ[idx]
                cache.integral.p[1:1, :, 1] .+= transpose(bases.x.p)*ψp.*bϵ2
                cache.integral.m[1:1, :, 1] .+= transpose(bases.x.m)*ψm.*bϵ2
            end
        else
            if isnothing(b.ϵ)
                # mul!(@view(cache.integral.p[1, 1, idx.i]), transpose(bases.x.p) * ψp, bases.Ω.p, true, true)
                CUDA.@allowscalar cache.integral.p[1, 1, idx.i] += dot(transpose(bases.x.p) * ψp, bases.Ω.p)
                CUDA.@allowscalar cache.integral.m[1, 1, idx.i] += dot(transpose(bases.x.m) * ψm, bases.Ω.m)
            else
                bϵ2 = idx.adjoint ? T(Δϵ*0.5) * (bases.ϵ[minus½(idx)] + bases.ϵ[plus½(idx)]) : T(Δϵ)*bases.ϵ[idx]
                CUDA.@allowscalar cache.integral.p[1, 1, 1] += dot(transpose(bases.x.p) * ψp, bases.Ω.p)*bϵ2
                CUDA.@allowscalar cache.integral.m[1, 1, 1] += dot(transpose(bases.x.m) * ψm, bases.Ω.m)*bϵ2
            end
        end
    end
    return nothing
end

function finalize_integration((; b, cache)::PNVectorIntegrator{<:PNProbe})
    if isnothing(b.x)
        if isnothing(b.Ω)
            if isnothing(b.ϵ)
                return cache.integral
            else
                return (p=dropdims(cache.integral.p; dims=3), m=dropdims(cache.integral.m; dims=3))
            end
        else
            if isnothing(b.ϵ)
                return (p=dropdims(cache.integral.p; dims=2)|> collect, m=dropdims(cache.integral.m; dims=2)|> collect)
            else
                return (p=dropdims(cache.integral.p; dims=(2, 3))|>collect, m=dropdims(cache.integral.m; dims=(2, 3))|>collect)
            end
        end
    else
        if isnothing(b.Ω)
            if isnothing(b.ϵ)
                return (p=dropdims(cache.integral.p; dims=1) |> collect, m=dropdims(cache.integral.m; dims=1) |> collect)
            else
                return (p=dropdims(cache.integral.p; dims=(1, 3)) |> collect, m=dropdims(cache.integral.m; dims=(1, 3)) |> collect)
            end
        else
            if isnothing(b.ϵ)
                return dropdims(cache.integral.p + cache.integral.m; dims=(1, 2)) |> collect
            else
                return only(cache.integral.p |> collect) + only(cache.integral.m |> collect)
            end
        end
    end
end

function (p::PNProbe)(ψ::AbstractDiscretePNSolution)
    return solve_and_integrate(p, ψ)
end

function interpolable(p::PNProbe, ψ::AbstractDiscretePNSolution)
    integral = solve_and_integrate(p, ψ)
    if isnothing(p.ϵ) error("Not yet implemented") end
    if isnothing(p.Ω) && !isnothing(p.x) return SphericalHarmonicsModels.interpolable(integral, direction_model(p.model)) end
    if isnothing(p.x) && !isnothing(p.Ω) return SpaceModels.interpolable(integral, space_model(p.model)) end
    return integral
end

# @concrete terse struct PNBoundaryProbe <: AbstractDiscretePNVector
#     model
#     arch
#     x
#     Ω
#     ϵ
# end

# function PNBoundaryProbe(model, arch; x=nothing, ϵ=nothing, Ω=nothing)
#     return PNProbe(model, arch, x, Ω, ϵ)
# end

# function initialize_integration(b::PNProbe)
#     (nϵ, (nxp, nxm), (nΩp, nΩm)) = n_basis(b.model)

#     integral = (p=zeros(isnothing(b.x) ? nxp : 1, isnothing(b.Ω) ? nΩp : 1, isnothing(b.ϵ) ? nϵ : 1)|> b.arch,
#                 m=zeros(isnothing(b.x) ? nxm : 1, isnothing(b.Ω) ? nΩm : 1, isnothing(b.ϵ) ? nϵ : 1) |> b.arch)
#     bases = (
#          x = isnothing(b.x) ? nothing : SpaceModels.eval_basis(space_model(b.model), b.x) |> b.arch,
#          Ω = isnothing(b.Ω) ? nothing : SphericalHarmonicsModels.eval_basis(direction_model(b.model), b.Ω) |> b.arch,
#          ϵ = isnothing(b.ϵ) ? nothing : energy_eval_basis(energy_model(b.model), b.ϵ)
#     )

#     return PNVectorIntegrator(b, (integral=integral, bases=bases))
# end

# function ((; b, cache)::PNVectorIntegrator{<:PNProbe})(idx, ψ)
#     ψp, ψm = pmview(ψ, b.model)
#     Δϵ = step(energy_model(b.model))
#     T = base_type(b.arch)
#     bases = cache.bases

#     if isnothing(b.x)
#         if isnothing(b.Ω)
#             if isnothing(b.ϵ)
#                 cache.integral.p[:, :, idx.i] .+= ψp 
#                 cache.integral.m[:, :, idx.i] .+= ψm
#             else
#                 bϵ2 = idx.adjoint ? T(Δϵ*0.5) * (bases.bϵ[minus½(idx)] + bases.bϵ[plus½(idx)]) : T(Δϵ)*bases.ϵ[idx]
#                 cache.integral.p[:, :, 1] .+= bϵ2 .* ψp
#                 cache.integral.m[:, :, 1] .+= bϵ2 .* ψm
#             end
#         else
#             if isnothing(b.ϵ)
#                 mul!(@view(cache.integral.p[:, 1, idx.i]), ψp, bases.Ω.p, true, true)
#                 mul!(@view(cache.integral.m[:, 1, idx.i]), ψm, bases.Ω.m, true, true)
#             else
#                 bϵ2 = idx.adjoint ? T(Δϵ*0.5) * (bases.ϵ[minus½(idx)] + bases.ϵ[plus½(idx)]) : T(Δϵ)*bases.ϵ[idx]
#                 mul!(@view(cache.integral.p[:, 1, 1]), ψp, bases.Ω.p, bϵ2, true)
#                 mul!(@view(cache.integral.m[:, 1, 1]), ψm, bases.Ω.m, bϵ2, true)
#             end
#         end
#     else
#         if isnothing(b.Ω)
#             if isnothing(b.ϵ)
#                 cache.integral.p[1:1, :, idx.i] .+= transpose(bases.x.p)*ψp
#                 cache.integral.m[1:1, :, idx.i] .+= transpose(bases.x.m)*ψm
#             else
#                 bϵ2 = idx.adjoint ? T(Δϵ*0.5) * (bases.ϵ[minus½(idx)] + bases.ϵ[plus½(idx)]) : T(Δϵ)*bases.ϵ[idx]
#                 cache.integral.p[1:1, :, 1] .+= transpose(bases.x.p)*ψp.*bϵ2
#                 cache.integral.m[1:1, :, 1] .+= transpose(bases.x.m)*ψm.*bϵ2
#             end
#         else
#             if isnothing(b.ϵ)
#                 # mul!(@view(cache.integral.p[1, 1, idx.i]), transpose(bases.x.p) * ψp, bases.Ω.p, true, true)
#                 CUDA.@allowscalar cache.integral.p[1, 1, idx.i] += dot(transpose(bases.x.p) * ψp, bases.Ω.p)
#                 CUDA.@allowscalar cache.integral.m[1, 1, idx.i] += dot(transpose(bases.x.m) * ψm, bases.Ω.m)
#             else
#                 bϵ2 = idx.adjoint ? T(Δϵ*0.5) * (bases.ϵ[minus½(idx)] + bases.ϵ[plus½(idx)]) : T(Δϵ)*bases.ϵ[idx]
#                 CUDA.@allowscalar cache.integral.p[1, 1, 1] += dot(transpose(bases.x.p) * ψp, bases.Ω.p)*bϵ2
#                 CUDA.@allowscalar cache.integral.m[1, 1, 1] += dot(transpose(bases.x.m) * ψm, bases.Ω.m)*bϵ2
#             end
#         end
#     end
#     return nothing
# end

# function finalize_integration((; b, cache)::PNVectorIntegrator{<:PNProbe})
#     if isnothing(b.x)
#         if isnothing(b.Ω)
#             if isnothing(b.ϵ)
#                 return cache.integral
#             else
#                 return (p=dropdims(cache.integral.p; dims=3), m=dropdims(cache.integral.m; dims=3))
#             end
#         else
#             if isnothing(b.ϵ)
#                 return (p=dropdims(cache.integral.p; dims=2)|> collect, m=dropdims(cache.integral.m; dims=2)|> collect)
#             else
#                 return (p=dropdims(cache.integral.p; dims=(2, 3))|>collect, m=dropdims(cache.integral.m; dims=(2, 3))|>collect)
#             end
#         end
#     else
#         if isnothing(b.Ω)
#             if isnothing(b.ϵ)
#                 return (p=dropdims(cache.integral.p; dims=1) |> collect, m=dropdims(cache.integral.m; dims=1) |> collect)
#             else
#                 return (p=dropdims(cache.integral.p; dims=(1, 3)) |> collect, m=dropdims(cache.integral.m; dims=(1, 3)) |> collect)
#             end
#         else
#             if isnothing(b.ϵ)
#                 return dropdims(cache.integral.p + cache.integral.m; dims=(1, 2)) |> collect
#             else
#                 return (p=dropdims(cache.integral.p), m=dropdims(cache.integral.m))
#             end
#         end
#     end
# end

# function (p::PNProbe)(ψ::AbstractDiscretePNSolution)
#     return solve_and_integrate(p, ψ)
# end
