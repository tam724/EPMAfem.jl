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
dimensionality(::AbstractSphericalHarmonicsModel{ND}) where {ND} = dimensionality_type(ND)

function findSHML_index(m, N)
    return findall(m_ -> m_ == (degree(m), order(m)), SphericalHarmonics.ML(0:N)) |> only
end

@concrete struct EOSphericalHarmonicsModel{ND} <: AbstractSphericalHarmonicsModel{ND}
    N
    num_dofs
    sh_index
    sh_cache
end

function EOSphericalHarmonicsModel(N, ND)
    _XD = dimensionality_type(ND)
    viable_moments = get_all_viable_harmonics_up_to(N, _XD)
    sort!(viable_moments, lt=isless_evenodd)

    # compute the index to evaluate using SphericalHarmonics.jl
    sh_index_even = [findSHML_index(m, N) for m in viable_moments if is_even(m)]
    sh_index_odd = [findSHML_index(m, N) for m in viable_moments if is_odd(m)]

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

function get_basis_harmonics(model::EOSphericalHarmonicsModel{ND}) where ND
    viable_moments = get_all_viable_harmonics_up_to(max_degree(model), dimensionality_type(ND))
    sort!(viable_moments, lt=isless_evenodd)
    return viable_moments
end

@concrete struct EEEOSphericalHarmonicsModel{ND} <: AbstractSphericalHarmonicsModel{ND}
    N
    num_dofs
    sh_index
    sh_cache
end

function EEEOSphericalHarmonicsModel(N, ND)
    _XD = dimensionality_type(ND)
    viable_moments = get_all_viable_harmonics_up_to(N, _XD)
    sort!(viable_moments, lt=isless_eeevenodd)

    # compute the index to evaluate using SphericalHarmonics.jl
    sh_index_eee = [findSHML_index(m, N) for m in viable_moments if get_eee(m) == EEEO.eee]
    sh_index_eoo = [findSHML_index(m, N) for m in viable_moments if get_eee(m) == EEEO.eoo]
    sh_index_oeo = [findSHML_index(m, N) for m in viable_moments if get_eee(m) == EEEO.oeo]
    sh_index_ooe = [findSHML_index(m, N) for m in viable_moments if get_eee(m) == EEEO.ooe]

    sh_index_oee = [findSHML_index(m, N) for m in viable_moments if get_eee(m) == EEEO.oee]
    sh_index_eoe = [findSHML_index(m, N) for m in viable_moments if get_eee(m) == EEEO.eoe]
    sh_index_eeo = [findSHML_index(m, N) for m in viable_moments if get_eee(m) == EEEO.eeo]
    sh_index_ooo = [findSHML_index(m, N) for m in viable_moments if get_eee(m) == EEEO.ooo]

    sh_index = ComponentVector(eee=sh_index_eee, eoo=sh_index_eoo, oeo=sh_index_oeo, ooe=sh_index_ooe, oee=sh_index_oee, eoe=sh_index_eoe, eeo=sh_index_eeo, ooo=sh_index_ooo)

    # compute the number of even and odd basis functions
    num_dofs_even = count(is_even.(viable_moments))
    num_dofs_odd = count(is_odd.(viable_moments))
    num_dofs = (even=num_dofs_even, odd=num_dofs_odd)

    # compute the SphericalHarmonics.jl cache
    sh_cache = SphericalHarmonics.cache(Float64, N, SHType=SphericalHarmonics.RealHarmonics())
    return EEEOSphericalHarmonicsModel{ND}(N, num_dofs, sh_index, sh_cache)
end

function get_basis_harmonics(model::EEEOSphericalHarmonicsModel{ND}) where ND
    viable_moments = get_all_viable_harmonics_up_to(max_degree(model), dimensionality_type(ND))
    sort!(viable_moments, lt=isless_eeevenodd)
    return viable_moments
end

max_degree(model::AbstractSphericalHarmonicsModel) = model.N

function even(model::EEEOSphericalHarmonicsModel)
    return @view(model.sh_index[(:eee, :eoo, :oeo, :ooe)])
end

function odd(model::EEEOSphericalHarmonicsModel)
    return @view(model.sh_index[(:oee, :eoe, :eeo, :ooo)])
end

function get_indices_∫S²absΩuv(model::EEEOSphericalHarmonicsModel{1})
    list = ((:eee, :eee), )
    return tuple(((getproperty(model.sh_index, l[1]).indices[1], getproperty(model.sh_index, l[2]).indices[1]) for l in list)...)
end

function get_indices_∫S²absΩuv(model::EEEOSphericalHarmonicsModel{2})
    list = ((:eee, :eee), (:ooe, :ooe))
    return tuple(((getproperty(model.sh_index, l[1]).indices[1], getproperty(model.sh_index, l[2]).indices[1]) for l in list)...)
end

function get_indices_∫S²absΩuv(model::EEEOSphericalHarmonicsModel{3})
    list = ((:eee, :eee), (:eoo, :eoo), (:oeo, :oeo), (:ooe, :ooe))
    return tuple(((getproperty(model.sh_index, l[1]).indices[1], getproperty(model.sh_index, l[2]).indices[1]) for l in list)...)
end

function get_indices_∫S²Ωuv(model::EEEOSphericalHarmonicsModel{1}, ::Z)
    n_even = n_basis(model).p
    list = ((:oee, :eee), )
    return tuple(((getproperty(model.sh_index, l[1]).indices[1].-n_even, getproperty(model.sh_index, l[2]).indices[1]) for l in list)...)
end

function get_indices_∫S²Ωuv(model::EEEOSphericalHarmonicsModel{2}, ::Z)
    n_even = n_basis(model).p
    list = ((:oee, :eee), (:eoe, :ooe))
    return tuple(((getproperty(model.sh_index, l[1]).indices[1].-n_even, getproperty(model.sh_index, l[2]).indices[1]) for l in list)...)
end

function get_indices_∫S²Ωuv(model::EEEOSphericalHarmonicsModel{2}, ::X)
    n_even = n_basis(model).p
    list = ((:eoe, :eee), (:oee, :ooe))
    return tuple(((getproperty(model.sh_index, l[1]).indices[1].-n_even, getproperty(model.sh_index, l[2]).indices[1]) for l in list)...)
end

function get_indices_∫S²Ωuv(model::EEEOSphericalHarmonicsModel{3}, ::Z)
    n_even = n_basis(model).p
    list = ((:oee, :eee), (:ooo, :eoo), (:eeo, :oeo), (:eoe, :ooe))
    return tuple(((getproperty(model.sh_index, l[1]).indices[1].-n_even, getproperty(model.sh_index, l[2]).indices[1]) for l in list)...)
end

function get_indices_∫S²Ωuv(model::EEEOSphericalHarmonicsModel{3}, ::X)
    n_even = n_basis(model).p
    list = ((:eoe, :eee), (:eeo, :eoo), (:ooo, :oeo), (:oee, :ooe))
    return tuple(((getproperty(model.sh_index, l[1]).indices[1].-n_even, getproperty(model.sh_index, l[2]).indices[1]) for l in list)...)
end

function get_indices_∫S²Ωuv(model::EEEOSphericalHarmonicsModel{3}, ::Y)
    n_even = n_basis(model).p
    list = ((:eeo, :eee), (:eoe, :eoo), (:oee, :oeo), (:ooo, :ooe))
    return tuple(((getproperty(model.sh_index, l[1]).indices[1].-n_even, getproperty(model.sh_index, l[2]).indices[1]) for l in list)...)
end

function num_dofs(model::EOSphericalHarmonicsModel)
    return model.num_dofs.even + model.num_dofs.odd
end

function num_dofs(model::EEEOSphericalHarmonicsModel)
    return model.num_dofs.even + model.num_dofs.odd
end

function n_basis(model::AbstractSphericalHarmonicsModel)
    return (p=model.num_dofs.even, m=model.num_dofs.odd)
end

function _eval_basis_functions_cache!(model::AbstractSphericalHarmonicsModel, Ω::VectorValue{3})
    # TODO (check): we mirror x and y to fit the definition on wikipedia https://en.wikipedia.org/wiki/Spherical_harmonics
    θ, ϕ = unitsphere_cartesian_to_spherical(VectorValue(Ωz(Ω), -Ωx(Ω), -Ωy(Ω)))
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

#dirac basis evaluation
function eval_basis(model, Ω::VectorValue{D}) where D
    eval_basis_functions!(model, Ω)
    (p=collect(model.sh_cache.Y[even(model)]), m=collect(model.sh_cache.Y[odd(model)]))
end

#integrated basis functions
function eval_basis(model, h::Function)
    (p=assemble_linear(∫S²_hv(h), model, even(model)), m=assemble_linear(∫S²_hv(h), model, odd(model)))
end


function interpolable(b, model)
    function interpolant(Ω)
        eval_basis_functions!(model, Ω)
        return dot(b.p, model.sh_cache.Y[even(model)]) + dot(b.m, model.sh_cache.Y[odd(model)])
    end
    return interpolant
end


