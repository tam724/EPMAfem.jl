abstract type AbstractHarmonicsModel{D, ND} end
Dimensions.dimensionality(::AbstractHarmonicsModel{D, ND}) where {D, ND} = dimensionality(ND)

@concrete struct EOHarmonicsModel{D, ND, EO} <: AbstractHarmonicsModel{D, ND}
    N::Int64
    moments
end

EOSphericalHarmonicsModel(N, ND, eo=:EO) = EOHarmonicsModel(3, N, ND, eo, spherical_harmonics)
EOCircularHarmonicsModel(N, ND, eo=:EO) = EOHarmonicsModel(2, N, ND, eo, circular_harmonics)

function EOHarmonicsModel(D, N, ND, eo, harmonics)
    viable_moments = harmonics(N, dimensionality(ND))
    sort!(viable_moments, lt=isless_evenodd)

    even_moments = [m for m in viable_moments if is_even(m)]
    odd_moments = [m for m in viable_moments if is_odd(m)]

    moments = ComponentVector(even=even_moments, odd=odd_moments)

    return EOHarmonicsModel{D, Dimensions.dimensionality_int(ND), eo}(N, moments)
end

even(model::EOHarmonicsModel) = model.moments.even
odd(model::EOHarmonicsModel) = model.moments.odd

plus(model::EOHarmonicsModel{D, ND, :EO}) where {D, ND} = model.moments.even
plus(model::EOHarmonicsModel{D, ND, :OE}) where {D, ND} = model.moments.odd

minus(model::EOHarmonicsModel{D, ND, :EO}) where {D, ND} = model.moments.odd
minus(model::EOHarmonicsModel{D, ND, :OE}) where {D, ND} = model.moments.even

eo(::EOHarmonicsModel{D, ND, EO}) where {D, ND, EO} = EO

function get_basis_harmonics(model::EOHarmonicsModel)
    return model.moments
end

@concrete struct EEEOSphericalHarmonicsModel{ND} <: AbstractHarmonicsModel{3, ND}
    N
    moments
end

function EEEOSphericalHarmonicsModel(N, ND)
    _XD = dimensionality(ND)
    viable_moments = spherical_harmonics(N, _XD)
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

    return EEEOSphericalHarmonicsModel{Dimensions.dimensionality_int(ND)}(N, moments)
end

function get_basis_harmonics(model::EEEOSphericalHarmonicsModel{ND}) where ND
    return model.moments
end

max_degree(model::AbstractHarmonicsModel) = model.N

function even(model::EEEOSphericalHarmonicsModel)
    return @view(model.moments[(:eee, :eoo, :oeo, :ooe)])
end

function odd(model::EEEOSphericalHarmonicsModel)
    return @view(model.moments[(:oee, :eoe, :eeo, :ooo)])
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

function num_dofs(model::EOHarmonicsModel)
    return length(plus(model)) + length(minus(model))
end

function num_dofs(model::EEEOSphericalHarmonicsModel)
    return length(plus(model)) + length(minus(model))
end

function n_basis(model::AbstractHarmonicsModel)
    return (p=length(plus(model)), m=length(minus(model)))
end

function _eval_basis_functions!(::AbstractHarmonicsModel{3}, Ω::VectorValue{3}, idx)
    return idx(Ω)
end

function _eval_basis_functions!(::AbstractHarmonicsModel{2}, Ω::VectorValue{2}, idx)
    return idx(Ω)
end

function _eval_basis_functions!(::AbstractHarmonicsModel{3}, Ω::VectorValue{3}, idx1, idx2)
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

function _eval_basis_functions!(::AbstractHarmonicsModel{2}, Ω::VectorValue{2}, idx1, idx2)
    return idx1(Ω), idx2(Ω)
end

function eval_basis_functions!(model::AbstractHarmonicsModel{ND}, Ω::VectorValue{ND}, idx...=model.moments) where ND
    return _eval_basis_functions!(model, Ω, idx...)
    # return _eval_basis_functions!(model, extend_3D(Ω), idx...)
end

function eval_basis_functions!(model::AbstractHarmonicsModel{ND1}, Ω::VectorValue{ND2}, idx...=model.moments) where {ND1, ND2}
    @warn "spherical harmonics basis of dimension $ND1 evaluated with direction of dimension $ND2"
    _eval_basis_functions!(model, extend_3D(Ω), idx...)
end

#dirac basis evaluation
function eval_basis(model::AbstractHarmonicsModel{2}, Ω::VectorValue)
    return (p=plus(model)(Ω), m=minus(model)(Ω))
end

function eval_basis(model::AbstractHarmonicsModel, Ω::VectorValue)
    p_vals, m_vals = eval_basis_functions!(model, Ω, plus(model), minus(model))
    return (p=p_vals, m=m_vals)
end

#integrated basis functions
function eval_basis(model, h::Function)
    (p=assemble_linear(∫S²_hv(h), model, plus(model)), m=assemble_linear(∫S²_hv(h), model, minus(model)))
end

function eval_basis(model::EOHarmonicsModel{ND, :EO}, ::typeof(one)) where ND
    (p=assemble_linear(∫S²_hv(Ω -> 1.0), model, plus(model)), m=nothing)
end

function eval_basis(model::EOHarmonicsModel{ND, :OE}, ::typeof(one)) where ND
    (p=nothing, m=assemble_linear(∫S²_hv(Ω -> 1.0), model, minus(model)))
end

function interpolable(b, model)
    function interpolant(Ω)
        # Ωx = Dimensions.constrain(Ω, dimensionality(model))
        if hasproperty(b, :m) # if not we assume its zero
            Yp, Ym = eval_basis_functions!(model, Ω, plus(model), minus(model))
            return dot(b.p, Yp) + dot(b.m, Ym)
        elseif hasproperty(b, :p) # if not we assume that b = b.p
            Yp = eval_basis_functions!(model, Ω, plus(model))
            return dot(b.p, Yp)
        else
            Yp = eval_basis_functions!(model, Ω, plus(model))
            return dot(b, Yp)
        end
    end
    return interpolant
end


