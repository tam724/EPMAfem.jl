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
    cache = get_cache(N)
    all_moments = (SphericalHarmonic(l, k, cache) for l in 0:N for k in -l:l)
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
    moments
end

function EOSphericalHarmonicsModel(N, ND)
    _XD = dimensionality_type(ND)
    viable_moments = get_all_viable_harmonics_up_to(N, _XD)
    sort!(viable_moments, lt=isless_evenodd)

    # compute the index to evaluate using SphericalHarmonics.jl
    even_moments = [m for m in viable_moments if is_even(m)]
    odd_moments = [m for m in viable_moments if is_odd(m)]

    moments = ComponentVector(even=even_moments, odd=odd_moments)

    # compute the number of even and odd basis functions
    num_dofs_even = length(even_moments)
    num_dofs_odd = length(odd_moments)
    num_dofs = (even=num_dofs_even, odd=num_dofs_odd)

    return EOSphericalHarmonicsModel{Dimensions.dimensionality_int(ND)}(N, num_dofs, moments)
end

function even(model::EOSphericalHarmonicsModel)
    return model.moments.even
end

function odd(model::EOSphericalHarmonicsModel)
    return model.moments.odd
end

function get_basis_harmonics(model::EOSphericalHarmonicsModel{ND}) where ND
    # viable_moments = get_all_viable_harmonics_up_to(max_degree(model), dimensionality_type(ND))
    # sort!(viable_moments, lt=isless_evenodd)
    return model.moments
end

@concrete struct EEEOSphericalHarmonicsModel{ND} <: AbstractSphericalHarmonicsModel{ND}
    N
    num_dofs
    moments
end

function EEEOSphericalHarmonicsModel(N, ND)
    _XD = dimensionality_type(ND)
    viable_moments = get_all_viable_harmonics_up_to(N, _XD)
    sort!(viable_moments, lt=isless_eeevenodd)

    # compute the index to evaluate using SphericalHarmonics.jl
    moments_eee = [m for m in viable_moments if get_eee(m) == EEEO.eee]
    moments_eoo = [m for m in viable_moments if get_eee(m) == EEEO.eoo]
    moments_oeo = [m for m in viable_moments if get_eee(m) == EEEO.oeo]
    moments_ooe = [m for m in viable_moments if get_eee(m) == EEEO.ooe]

    moments_oee = [m for m in viable_moments if get_eee(m) == EEEO.oee]
    moments_eoe = [m for m in viable_moments if get_eee(m) == EEEO.eoe]
    moments_eeo = [m for m in viable_moments if get_eee(m) == EEEO.eeo]
    moments_ooo = [m for m in viable_moments if get_eee(m) == EEEO.ooo]

    moments = ComponentVector(eee=moments_eee, eoo=moments_eoo, oeo=moments_oeo, ooe=moments_ooe, oee=moments_oee, eoe=moments_eoe, eeo=moments_eeo, ooo=moments_ooo)

    # compute the number of even and odd basis functions
    num_dofs_even = count(is_even.(viable_moments))
    num_dofs_odd = count(is_odd.(viable_moments))
    num_dofs = (even=num_dofs_even, odd=num_dofs_odd)

    return EEEOSphericalHarmonicsModel{Dimensions.dimensionality_int(ND)}(N, num_dofs, moments)
end

function get_basis_harmonics(model::EEEOSphericalHarmonicsModel{ND}) where ND
    return model.moments
end

max_degree(model::AbstractSphericalHarmonicsModel) = model.N

function even(model::EEEOSphericalHarmonicsModel)
    return @view(model.moments[(:eee, :eoo, :oeo, :ooe)])
end

function odd(model::EEEOSphericalHarmonicsModel)
    return @view(model.moments[(:oee, :eoe, :eeo, :ooo)])
end

function even_in(model::AbstractSphericalHarmonicsModel{ND}, n::VectorValue) where ND
    return [m for m in model.moments if is_even_in(m, n)]
    # _XD = dimensionality_type(ND)
    # viable_moments = get_all_viable_harmonics_up_to(model.N, _XD)
    # sort!(viable_moments, lt=isless_eeevenodd)
    # sh_index_even_in = [findSHML_index(m, model.N) for m in viable_moments if is_even_in(m, n)]
    # return sh_index_even_in
end 

function odd_in(model::AbstractSphericalHarmonicsModel{ND}, n::VectorValue) where ND
    return [m for m in model.moments if is_odd_in(m, n)]
    # _XD = dimensionality_type(ND)
    # viable_moments = get_all_viable_harmonics_up_to(model.N, _XD)
    # sort!(viable_moments, lt=isless_eeevenodd)
    # sh_index_even_in = [findSHML_index(m, model.N) for m in viable_moments if is_odd_in(m, n)]
    # return sh_index_even_in
end 

function get_indices_∫S²absΩuv(model::EEEOSphericalHarmonicsModel{1})
    list = ((:eee, :eee), )
    return tuple(((getproperty(model.moments, l[1]).indices[1], getproperty(model.moments, l[2]).indices[1]) for l in list)...)
end

function get_indices_∫S²absΩuv(model::EEEOSphericalHarmonicsModel{2})
    list = ((:eee, :eee), (:ooe, :ooe))
    return tuple(((getproperty(model.moments, l[1]).indices[1], getproperty(model.moments, l[2]).indices[1]) for l in list)...)
end

function get_indices_∫S²absΩuv(model::EEEOSphericalHarmonicsModel{3})
    list = ((:eee, :eee), (:eoo, :eoo), (:oeo, :oeo), (:ooe, :ooe))
    return tuple(((getproperty(model.moments, l[1]).indices[1], getproperty(model.moments, l[2]).indices[1]) for l in list)...)
end

function get_indices_∫S²Ωuv(model::EEEOSphericalHarmonicsModel{1}, ::Z)
    n_even = n_basis(model).p
    list = ((:oee, :eee), )
    return tuple(((getproperty(model.moments, l[1]).indices[1].-n_even, getproperty(model.moments, l[2]).indices[1]) for l in list)...)
end

function get_indices_∫S²Ωuv(model::EEEOSphericalHarmonicsModel{2}, ::Z)
    n_even = n_basis(model).p
    list = ((:oee, :eee), (:eoe, :ooe))
    return tuple(((getproperty(model.moments, l[1]).indices[1].-n_even, getproperty(model.moments, l[2]).indices[1]) for l in list)...)
end

function get_indices_∫S²Ωuv(model::EEEOSphericalHarmonicsModel{2}, ::X)
    n_even = n_basis(model).p
    list = ((:eoe, :eee), (:oee, :ooe))
    return tuple(((getproperty(model.moments, l[1]).indices[1].-n_even, getproperty(model.moments, l[2]).indices[1]) for l in list)...)
end

function get_indices_∫S²Ωuv(model::EEEOSphericalHarmonicsModel{3}, ::Z)
    n_even = n_basis(model).p
    list = ((:oee, :eee), (:ooo, :eoo), (:eeo, :oeo), (:eoe, :ooe))
    return tuple(((getproperty(model.moments, l[1]).indices[1].-n_even, getproperty(model.moments, l[2]).indices[1]) for l in list)...)
end

function get_indices_∫S²Ωuv(model::EEEOSphericalHarmonicsModel{3}, ::X)
    n_even = n_basis(model).p
    list = ((:eoe, :eee), (:eeo, :eoo), (:ooo, :oeo), (:oee, :ooe))
    return tuple(((getproperty(model.moments, l[1]).indices[1].-n_even, getproperty(model.moments, l[2]).indices[1]) for l in list)...)
end

function get_indices_∫S²Ωuv(model::EEEOSphericalHarmonicsModel{3}, ::Y)
    n_even = n_basis(model).p
    list = ((:eeo, :eee), (:eoe, :eoo), (:oee, :oeo), (:ooo, :ooe))
    return tuple(((getproperty(model.moments, l[1]).indices[1].-n_even, getproperty(model.moments, l[2]).indices[1]) for l in list)...)
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

# should not be needed anymore
# function _eval_basis_functions_cache!(model::AbstractSphericalHarmonicsModel, Ω::VectorValue{3})
#     # TODO (check): we mirror x and y to fit the definition on wikipedia https://en.wikipedia.org/wiki/Spherical_harmonics
#     θ, ϕ = unitsphere_cartesian_to_spherical(VectorValue(Ωz(Ω), -Ωx(Ω), -Ωy(Ω)))
#     SphericalHarmonics.computePlmcostheta!(model.sh_cache, θ)
#     SphericalHarmonics.computeYlm!(model.sh_cache, θ, ϕ)
# end

function _eval_basis_functions!(::AbstractSphericalHarmonicsModel, Ω::VectorValue{3}, idx)
    return idx(Ω)
    # _eval_basis_functions_cache!(model, Ω)
    # return @view(model.sh_cache.Y[idx])
end

function _eval_basis_functions!(::AbstractSphericalHarmonicsModel, Ω::VectorValue{3}, idx1, idx2)
    y1 = idx1(Ω)
    cache = first(idx1).cache
    y2 = zeros(length(idx2))
    for (i, sh) in enumerate(idx2)
        y2[i] = cache.Y[(degree(sh), order(sh))]
    end
    return y1, y2
    # _eval_basis_functions_cache!(model, Ω)
    # return @view(model.sh_cache.Y[idx1]), @view(model.sh_cache.Y[idx2])
end

function eval_basis_functions!(model::AbstractSphericalHarmonicsModel{ND}, Ω::VectorValue{ND}, idx...=model.moments) where ND
    _eval_basis_functions!(model, extend_3D(Ω), idx...)
end

function eval_basis_functions!(model::AbstractSphericalHarmonicsModel{ND1}, Ω::VectorValue{ND2}, idx...=model.moments) where {ND1, ND2}
    @warn "spherical harmonics basis of dimension $ND1 evaluated with direction of dimension $ND2"
    _eval_basis_functions!(model, extend_3D(Ω), idx...)
end

#dirac basis evaluation
function eval_basis(model::AbstractSphericalHarmonicsModel, Ω::VectorValue)
    p_vals, m_vals = eval_basis_functions!(model, Ω, even(model), odd(model))
    (p=p_vals, m=m_vals)
end

#integrated basis functions
function eval_basis(model, h::Function)
    (p=assemble_linear(∫S²_hv(h), model, even(model)), m=assemble_linear(∫S²_hv(h), model, odd(model)))
end

function interpolable(b, model)
    function interpolant(Ω)
        if hasproperty(b, :m) # if not we assume its zero
            Y_even, Y_odd = eval_basis_functions!(model, Ω, even(model), odd(model))
            return dot(b.p, Y_even) + dot(b.m, Y_odd)
        elseif hasproperty(b, :p) # if not we assume that b = b.p
            Y_even = eval_basis_functions!(model, Ω, even(model))
            return dot(b.p, Y_even)
        else
            Y_even = eval_basis_functions!(model, Ω, even(model))
            return dot(b, Y_even)
        end
    end
    return interpolant
end


