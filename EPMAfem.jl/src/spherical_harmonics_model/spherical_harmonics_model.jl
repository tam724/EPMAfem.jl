function is_viable(m::SphericalHarmonic, ND)
    dims_m_should_be_even = filter(d -> d ∉ dimensions(ND), dimensions())
    for dim in dims_m_should_be_even
        if is_odd_in(m, dim)
            return false
        end
    end
    return true
end

function get_all_viable_harmonics_up_to(N, ND)
    all_moments = (SphericalHarmonic(l, k) for l in 0:N for k in -l:l)
    viable_moments = [m for m in all_moments if is_viable(m, ND)]
    return viable_moments
end

abstract type AbstractSphericalHarmonicsModel{ND} end
@concrete struct EOSphericalHarmonicsModel{ND} <: AbstractSphericalHarmonicsModel{ND}
    N
    num_dofs
    sh_index
    sh_cache
end

function EOSphericalHarmonicsModel(N, ND)
    viable_moments = get_all_viable_harmonics_up_to(N, ND)
    sort!(viable_moments, lt=isless_evenodd)

    # compute the index to evaluate using SphericalHarmonics.jl
    sh_index_even = [findall(m_ -> m_ == (degree(m), order(m)), SphericalHarmonics.ML(0:N)) |> only for m in viable_moments if is_even(m)]
    sh_index_odd = [findall(m_ -> m_ == (degree(m), order(m)), SphericalHarmonics.ML(0:N)) |> only for m in viable_moments if is_odd(m)]

    sh_index = ComponentVector(even=sh_index_even, odd=sh_index_odd)

    # compute the number of even and odd basis functions
    num_dofs_even = length(sh_index_even)
    num_dofs_odd = length(sh_index_odd)
    num_dofs = (even=num_dofs_even, odd=num_dofs_odd)

    # compute the SphericalHarmonics.jl cache
    sh_cache = SphericalHarmonics.cache(Float64, N, SHType=SphericalHarmonics.RealHarmonics())
    return EOSphericalHarmonicsModel{ND}(N, num_dofs, sh_index, sh_cache)
end

function even(model::EOSphericalHarmonicsModel)
    return model.sh_index.even
end

function odd(model::EOSphericalHarmonicsModel)
    return model.sh_index.odd
end

@concrete struct EEEOSphericalHarmonicsModel{ND} <: AbstractSphericalHarmonicsModel{ND}
    N
    num_dofs
    sh_index
    sh_cache
end

function EEEOSphericalHarmonicsModel(N, ND)
    viable_moments = get_all_viable_harmonics_up_to(N, ND)
    sort!(viable_moments, lt=isless_eeevenodd)

    # compute the index to evaluate using SphericalHarmonics.jl
    sh_index_eee = [findall(m_ -> m_ == (degree(m), order(m)), SphericalHarmonics.ML(0:N)) |> only for m in viable_moments if get_eee(m) == EEEO.eee]
    sh_index_eoo = [findall(m_ -> m_ == (degree(m), order(m)), SphericalHarmonics.ML(0:N)) |> only for m in viable_moments if get_eee(m) == EEEO.eoo]
    sh_index_oeo = [findall(m_ -> m_ == (degree(m), order(m)), SphericalHarmonics.ML(0:N)) |> only for m in viable_moments if get_eee(m) == EEEO.oeo]
    sh_index_ooe = [findall(m_ -> m_ == (degree(m), order(m)), SphericalHarmonics.ML(0:N)) |> only for m in viable_moments if get_eee(m) == EEEO.ooe]

    sh_index_oee = [findall(m_ -> m_ == (degree(m), order(m)), SphericalHarmonics.ML(0:N)) |> only for m in viable_moments if get_eee(m) == EEEO.oee]
    sh_index_eoe = [findall(m_ -> m_ == (degree(m), order(m)), SphericalHarmonics.ML(0:N)) |> only for m in viable_moments if get_eee(m) == EEEO.eoe]
    sh_index_eeo = [findall(m_ -> m_ == (degree(m), order(m)), SphericalHarmonics.ML(0:N)) |> only for m in viable_moments if get_eee(m) == EEEO.eeo]
    sh_index_ooo = [findall(m_ -> m_ == (degree(m), order(m)), SphericalHarmonics.ML(0:N)) |> only for m in viable_moments if get_eee(m) == EEEO.ooo]

    sh_index = ComponentVector(eee=sh_index_eee, eoo=sh_index_eoo, oeo=sh_index_oeo, ooe=sh_index_ooe, oee=sh_index_oee, eoe=sh_index_eoe, eeo=sh_index_eeo, ooo=sh_index_ooo)

    # compute the number of even and odd basis functions
    num_dofs_even = count(is_even.(viable_moments))
    num_dofs_odd = count(is_odd.(viable_moments))
    num_dofs = (even=num_dofs_even, odd=num_dofs_odd)

    # compute the SphericalHarmonics.jl cache
    sh_cache = SphericalHarmonics.cache(Float64, N, SHType=SphericalHarmonics.RealHarmonics())
    return EEEOSphericalHarmonicsModel{ND}(N, num_dofs, sh_index, sh_cache)
end

max_degree(model::AbstractSphericalHarmonicsModel) = model.N

function get_basis_harmonics(model::EOSphericalHarmonicsModel{ND}) where ND
    viable_moments = get_all_viable_harmonics_up_to(max_degree(model), ND)
    sort!(viable_moments, lt=isless_evenodd)
    return viable_moments
end

function even(model::EEEOSphericalHarmonicsModel)
    return @view(model.sh_index[(:eee, :eoo, :oeo, :ooe)])
end

function odd(model::EEEOSphericalHarmonicsModel)
    return @view(model.sh_index[(:oee, :eoe, :ooe, :ooo)])
end

function get_basis_harmonics(model::EEEOSphericalHarmonicsModel{ND}) where ND
    viable_moments = get_all_viable_harmonics_up_to(max_degree(model), ND)
    sort!(viable_moments, lt=isless_eeevenodd)
    return viable_moments
end

function num_dofs(model::EOSphericalHarmonicsModel)
    return model.num_dofs.even + model.num_dofs.odd
end

function num_dofs(model::EEEOSphericalHarmonicsModel)
    return model.num_dofs.even + model.num_dofs.odd
end

function _eval_basis_functions_cache!(model::AbstractSphericalHarmonicsModel, Ω::VectorValue{3})
    # TODO (check): we mirror x and y to fit the definition on wikipedia https://en.wikipedia.org/wiki/Spherical_harmonics
    θ, ϕ = unitsphere_cartesian_to_spherical(VectorValue(Ω[1], -Ω[2], -Ω[3]))
    SphericalHarmonics.computePlmcostheta!(model.sh_cache, θ)
    SphericalHarmonics.computeYlm!(model.sh_cache, θ, ϕ)
end


function _eval_basis_functions!(model::AbstractSphericalHarmonicsModel, Ω::VectorValue{3}, idx)
    _eval_basis_functions_cache!(model, Ω)
    return @view(model.sh_cache.Y[idx])
end

function _eval_basis_functions!(model::AbstractSphericalHarmonicsModel, Ω::VectorValue{3}, idx1, idx2)
    _eval_basis_functions_cache!(model, Ω)
    return @view(model.sh_cache.Y[idx1]), @view(model.sh_cache.Y[idx2])
end

function eval_basis_functions!(model::AbstractSphericalHarmonicsModel{ND}, Ω::VectorValue{ND}, idx=model.sh_index) where ND
    _eval_basis_functions!(model, extend_3D(Ω), idx)
end

function eval_basis_functions!(model::AbstractSphericalHarmonicsModel{ND1}, Ω::VectorValue{ND2}, idx=model.sh_index) where {ND1, ND2}
    @warn "spherical harmonics basis of dimension $ND1 evaluated with direction of dimension $ND2"
    _eval_basis_functions!(model, extend_3D(Ω), idx)
end



