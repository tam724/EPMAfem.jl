@concrete struct FilteredPNEquations <: AbstractPNEquations
    eq
    σ_f
    filter_function
end

filter_exp(eq::AbstractPNEquations, σ_f, α) = FilteredPNEquations(eq, σ_f, SphericalHarmonicsModels.ExpFilter(α))

# impl
is_filter_element(eq::FilteredPNEquations, e) = e == number_of_elements(eq.eq) + 1
function filtered_scattering_kernel(eq::FilteredPNEquations)
    return eq.filter_function
end

number_of_elements(eq::FilteredPNEquations) = number_of_elements(eq.eq) + 1
number_of_scatterings(eq::FilteredPNEquations) = number_of_scatterings(eq.eq)
function stopping_power(eq::FilteredPNEquations, e, ϵ)
    if is_filter_element(eq, e)
        return 0.0
    else
        return stopping_power(eq.eq, e, ϵ)
    end
end
function absorption_coefficient(eq::FilteredPNEquations, e, ϵ)
    if is_filter_element(eq, e)
        return 0.0
    else
        return absorption_coefficient(eq.eq, e, ϵ)
    end
end
function scattering_coefficient(eq::FilteredPNEquations, e, i, ϵ)
    if is_filter_element(eq, e)
        return eq.σ_f
    else
        return scattering_coefficient(eq.eq, e, i, ϵ)
    end
end

function mass_concentrations(eq::FilteredPNEquations, e, x)
    if is_filter_element(eq, e)
        return 1.0
    else
        return mass_concentrations(eq.eq, e, x)
    end
end

function scattering_kernel(eq::FilteredPNEquations, e, i)
    if is_filter_element(eq, e)
        return filtered_scattering_kernel(eq)
    else
        return scattering_kernel(eq.eq, e, i)
    end
end

# the same for monochrom-equations
@concrete struct FilteredMonochromPNEquations <: AbstractMonochromPNEquations
    eq
    σ_f
    filter_function
end

filter_exp(eq::AbstractMonochromPNEquations, σ_f, α) = FilteredMonochromPNEquations(eq, σ_f, SphericalHarmonicsModels.ExpFilter(α))

#impl
is_filter_element(eq::FilteredMonochromPNEquations, e) = e == number_of_elements(eq.eq) + 1
function filtered_scattering_kernel(eq::FilteredMonochromPNEquations)
    return eq.filter_function
end

number_of_elements(eq::FilteredMonochromPNEquations) = number_of_elements(eq.eq) + 1
function absorption_coefficient(eq::FilteredMonochromPNEquations, e)
    if is_filter_element(eq, e)
        return 0.0
    else
        return absorption_coefficient(eq.eq, e)
    end
end
function scattering_coefficient(eq::FilteredMonochromPNEquations, e)
    if is_filter_element(eq, e)
        return eq.σ_f
    else
        return scattering_coefficient(eq.eq, e)
    end
end

function mass_concentrations(eq::FilteredMonochromPNEquations, e, x)
    if is_filter_element(eq, e)
        return 1.0
    else
        return mass_concentrations(eq.eq, e, x)
    end
end

function scattering_kernel(eq::FilteredMonochromPNEquations, e)
    if is_filter_element(eq, e)
        return filtered_scattering_kernel(eq)
    else
        return scattering_kernel(eq.eq, e)
    end
end

