@concrete struct PNEquations
    scattering_norm_factor
end

function PNEquations()
    scattering_norm_factor = 2*π*hquadrature(x -> scattering_kernel_func(x), -1.0, 1.0)[1]
    return PNEquations(scattering_norm_factor)
end

function number_of_elements(::PNEquations)
    return 2
end

function number_of_scatterings(::PNEquations)
    return 1
end

function stopping_power(::PNEquations, e, ϵ)
    return 1.0
end

function absorption_coefficient(::PNEquations, e, ϵ)
    return 1.0
end

function scattering_coefficient(::PNEquations, e, i, ϵ)
    return 1.0
end

function mass_concentrations(::PNEquations, e, x)
    return 1.0
end

scattering_kernel_func(μ) = exp(-5.0*(μ-1.0)^2)

function electron_scattering_kernel(eq::PNEquations, e, i, μ)
    # for now we ignore e and i
    return scattering_kernel_func(μ) / eq.scattering_norm_factor
end
