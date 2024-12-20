
"""
    A concrete struct of abstract type AbstractPNEquations should support the following functions:
"""
abstract type AbstractPNEquations end

"""
    PNEquations
    ``∂ₑ(su) + Ω⋅∇u + τu ...``
"""
struct PNEquations{EEQ} <: AbstractPNEquations
    eq::EEQ
end

function unit_energy(eq::PNEquations)
    return 1.0u"keV"
end

function unit_length(eq::PNEquations)
    return 1.0u"nm"
end

function unit_mass(eq::PNEquations)
    return 1.0u"u"
end

function _s(eq::PNEquations, e)
    return function(ϵ_)
        units_specific_stopping_power = unit_energy(eq) * unit_length(eq)^5 / unit_mass(eq)^2
        return 0.5 * specific_stopping_power(eq.eq, e)(ϵ_*unit_energy(eq)) / units_specific_stopping_power
    end
end

function _τ(eq::PNEquations, e)
    return function(ϵ_::T) where T
        units_specific_stopping_power = unit_energy(eq) * unit_length(eq)^5 / unit_mass(eq)^2
        units_specific_electron_scattering_cross_section = unit_length(eq)^5 / unit_mass(eq)^2
        # do we have to be careful with the derivative here?
        ∂spe_stop_pow = Enzyme.autodiff(Forward, ϵ_ -> specific_stopping_power(eq.eq, e)(ϵ_*unit_energy(eq))/units_specific_stopping_power, Duplicated(ϵ_, one(T)))[1]
        σ_tot = total_specific_electron_scattering_cross_section(eq.eq, e)(ϵ_*unit_energy(eq)) / units_specific_electron_scattering_cross_section
        return σ_tot - 0.5 * ∂spe_stop_pow
    end
end

function _σ(eq::PNEquations, e, i)
    return function(ϵ_)
        units_specific_electron_scattering_cross_section = unit_length(eq)^5 / unit_mass(eq)^2
        return specific_electron_scattering_cross_section(eq.eq, e, i)(ϵ_*unit_energy(eq)) / units_specific_electron_scattering_cross_section
    end
end

number_of_elements(eq::PNEquations) = number_of_elements(eq.eq)
number_of_scatterings(eq::PNEquations) = number_of_scatterings(eq.eq)
number_of_beam_energies(eq::PNEquations) = number_of_beam_energies(eq.eq)
number_of_beam_positions(eq::PNEquations) = number_of_beam_positions(eq.eq)
number_of_beam_directions(eq::PNEquations) = number_of_beam_directions(eq.eq)
number_of_extraction_positions(eq::PNEquations) = number_of_extraction_positions(eq.eq)
number_of_extraction_directions(eq::PNEquations) = number_of_extraction_directions(eq.eq)
number_of_extraction_energies(eq::PNEquations) = number_of_extraction_energies(eq.eq)

max_number_of_space_rhs(eq::PNEquations) = max(number_of_beam_positions(eq), number_of_extraction_positions(eq))
max_number_of_direction_rhs(eq::PNEquations) = max(number_of_beam_directions(eq), number_of_extraction_directions(eq))

extend_3D(x::VectorValue{1, T}) where T = (x.data..., zero(T), zero(T))
extend_3D(x::VectorValue{2, T}) where T = (x.data..., zero(T))
extend_3D(x::VectorValue{3, T}) where T = (x.data..., )

function _mass_concentrations(eq::PNEquations, e)
    units_mass_concentration = unit_mass(eq)/unit_length(eq)^3
    return x -> mass_concentrations(eq.eq, e, extend_3D(x).*unit_length(eq)) / units_mass_concentration
end

function _electron_scattering_kernel(eq::PNEquations, e, i)
    return electron_scattering_kernel(eq.eq, e, i)
end

#always take the first for now..
function _excitation_energy_distribution(eq::PNEquations)
    return ϵ_ -> beam_energy_distribution(eq.eq, 1)(ϵ_ * unit_energy(eq))
end

function _excitation_spatial_distribution(eq::PNEquations)
    return x_ -> beam_spatial_distribution(eq.eq, 1)(extend_3D(x_).*unit_length(eq))
end

function _excitation_direction_distribution(eq::PNEquations)
    return beam_direction_distribution(eq.eq, 2)
end

function _extraction_energy_distribution(eq::PNEquations)
    return ϵ_ -> extraction_energy_distribution(eq.eq, 1)(ϵ_ * unit_energy(eq))
end

function _extraction_spatial_distribution(eq::PNEquations)
    return x_ -> extraction_spatial_distribution(eq.eq, 2)(extend_3D(x_).*unit_length(eq))
end

function _extraction_direction_distribution(eq::PNEquations)
    return extraction_direction_distribution(eq.eq, 1)
end

function _specific_attenuation_coefficient(eq::PNEquations, e, j)
    units_specific_attenuation_coefficient = unit_mass(eq) / unit_length(eq)^3
    return specific_attenuation_coefficient(eq.eq, e, j) / units_specific_attenuation_coefficient
end

