@concrete terse struct Rank1DiscretePNVector <: AbstractDiscretePNVector
    adjoint::Bool
    model
    arch
    bϵ
    # might be moved to gpu
    bxp
    bΩp
end

Base.show(io::IO, v::Rank1DiscretePNVector) = print(io, "Rank1DiscretePNVector [adj = $(v.adjoint)]")
Base.show(io::IO, ::MIME"text/plain", v::Rank1DiscretePNVector) = show(io, v)

_is_adjoint_vector(b::Rank1DiscretePNVector) = b.adjoint

function initialize_integration(b::Rank1DiscretePNVector)
    (_, (_, _), (nΩp, _)) = n_basis(b.model)
    buf = allocate_vec(b.arch, nΩp)
    cache = (integral = [0.0], buf = buf)
    return PNVectorIntegrator(b, cache)
end

function finalize_integration((; b, cache)::PNVectorIntegrator{<:Rank1DiscretePNVector})
    return cache.integral[1]
end

function ((; b, cache)::PNVectorIntegrator{<:Rank1DiscretePNVector})(idx, ψ)
    ψp = pview(ψ, b.model)
    Δϵ = step(energy_model(b.model))
    T = base_type(b.arch)
    if idx.adjoint
        @assert !_is_adjoint_vector(b)
        bϵ2 = T(0.5) * (b.bϵ[minus½(idx)] + b.bϵ[plus½(idx)])
    else
        @assert _is_adjoint_vector(b)
        bϵ2 = b.bϵ[idx]
    end
    # CUDA does not like this :(
    # mul!(transpose(cache.buf), transpose(b.bxp), ψp) AT = BT * C <=> A = CT * B
    mul!(cache.buf, transpose(ψp), b.bxp)
    cache.integral[1] += Δϵ * bϵ2 * dot(cache.buf, b.bΩp)
    return nothing
end

function initialize_assembly(b::Rank1DiscretePNVector)
    return PNVectorAssembler(b, nothing)
end

# if β isa Number we add to rhs
function assemble_at!(rhs, (; b)::PNVectorAssembler{<:Rank1DiscretePNVector}, idx, Δ, sym, β=false)
    bp, bm = pmview(rhs, b.model)
    T = base_type(b.arch)
    if idx.adjoint #you are assembling the rhs at the half step
        @assert !_is_adjoint_vector(b)
        bϵ2 = T(0.5)*(b.bϵ[minus½(idx)] + b.bϵ[plus½(idx)])
    else
        @assert _is_adjoint_vector(b)
        bϵ2 = b.bϵ[idx]
    end
    mul!(bp, b.bxp, transpose(b.bΩp), bϵ2*Δ, β)
    my_rmul!(bm, β) # * -1 if sym
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

## for array valued integrations
function initialize_integration(b::Array{<:Rank1DiscretePNVector})
    (_, (_, _), (nΩp, _)) = n_basis(first(b).model)
    buf = allocate_vec(first(b).arch, nΩp)
    idx_order = ideal_index_order(b)
    cache = (integral = zeros(size(b)), idx_order = idx_order, buf = buf)
    return PNVectorIntegrator(b, cache)
end

function finalize_integration((; b, cache)::PNVectorIntegrator{<:Array{<:Rank1DiscretePNVector}})
    return cache.integral
end

function ((; b, cache)::PNVectorIntegrator{<:Array{<:Rank1DiscretePNVector}})(idx, ψ)
    ψp = pview(ψ, first(b).model)
    Δϵ = step(energy_model(first(b).model))
    T = base_type(first(b).arch)
    for (x_base, x_rem) in cache.idx_order
        # CUDA does not like this :(
        # mul!(transpose(cache.buf), transpose(b[x_base].bxp), ψp) AT = BT * C <=> A = CT * B
        mul!(cache.buf, transpose(ψp), b[x_base].bxp)
        for (Ω_base, Ωx_rem) in x_rem
            bufbuf = dot(b[Ω_base].bΩp, cache.buf)
            for i in Ωx_rem
                if idx.adjoint
                    @assert !_is_adjoint_vector(b[i])
                    cache.integral[i] += Δϵ * T(0.5) * (b[i].bϵ[plus½(idx)] + b[i].bϵ[minus½(idx)])*bufbuf
                else
                    @assert _is_adjoint_vector(b[i])
                    cache.integral[i] += Δϵ * b[i].bϵ[idx] * bufbuf
                end
            end
        end
    end
end
@concrete struct SumOfAbstractDiscretePNVector <: AbstractDiscretePNVector
    weights
    vecs
end

_is_adjoint_vector(b::SumOfAbstractDiscretePNVector) = all(bi -> _is_adjoint_vector(bi) == _is_adjoint_vector(first(b.vecs)), b.vecs) ? _is_adjoint_vector(first(b.vecs)) : throw(ArgumentError("Vectors have different adjoint properties"))

function weighted(weights, vecs::Array{<:AbstractDiscretePNVector})
    @assert size(weights) == size(vecs)
    if !all(r1 -> _is_adjoint_vector(first(vecs)) == _is_adjoint_vector(r1), vecs) throw(ErrorException("cannot create a sum of vectors with different adjoint properties")) end
    return SumOfAbstractDiscretePNVector(weights, vecs)
end

function initialize_assembly(b::SumOfAbstractDiscretePNVector)
    cache = [initialize_assembly(b.vecs[i]) for i in eachindex(b.vecs)]
    return PNVectorAssembler(b, cache)
end

function assemble_at!(rhs, (; b, cache)::PNVectorAssembler{<:SumOfAbstractDiscretePNVector}, idx, Δ, sym, β=false)
    for i in eachindex(b.vecs)
        assemble_at!(rhs, cache[i], idx, Δ*b.weights[i], sym, β)
        β = true
    end
end

function initialize_integration(b::SumOfAbstractDiscretePNVector)
    cache = [initialize_integration(v) for v in b.vecs]
    return PNVectorIntegrator(b, cache)
end

function ((; b, cache)::PNVectorIntegrator{<:SumOfAbstractDiscretePNVector})(idx, ψ)
    for i in eachindex(b.vecs, cache)
        cache[i](idx, ψ)
    end
end

function finalize_integration((; b, cache)::PNVectorIntegrator{<:SumOfAbstractDiscretePNVector})
    return sum(b.weights[i] * finalize_integration(cache[i]) for i in eachindex(b.weights, b.vecs))
end

# for fun
Base.:*(a::Real, b::AbstractDiscretePNVector) = weighted([a], [b])
Base.:*(b::AbstractDiscretePNVector, a::Real) = a * b
Base.:+(a::AbstractDiscretePNVector, b::AbstractDiscretePNVector) = weighted([1.0, 1.0], [a, b])
Base.:+(a::AbstractDiscretePNVector, b::SumOfAbstractDiscretePNVector) = weighted([1.0, b.weights...], [a, b.vecs...])
Base.:+(a::SumOfAbstractDiscretePNVector, b::AbstractDiscretePNVector) = b + a
Base.:+(a::SumOfAbstractDiscretePNVector, b::SumOfAbstractDiscretePNVector) = weighted([a.weights..., b.weights...], [a.vecs..., b.vecs...])
