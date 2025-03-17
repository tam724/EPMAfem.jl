
@concrete struct MonochromPNEquations <: AbstractMonochromPNEquations
    scattering_norm_factor
end

monochrom_scattering_kernel_func(::MonochromPNEquations, μ) = exp(-5.0*(μ-1.0)^2)

function MonochromPNEquations()
    dummy_eq = MonochromPNEquations(1.0)
    scattering_norm_factor = 2*π*hquadrature(x -> monochrom_scattering_kernel_func(dummy_eq, x), -1.0, 1.0)[1]
    return MonochromPNEquations(scattering_norm_factor)
end

function number_of_elements(eq::MonochromPNEquations)
    return 2
end

function scattering_coefficient(eq::MonochromPNEquations, e)
    return 1.0
end

function absorption_coefficient(eq::MonochromPNEquations, e)
    # make it a balance
    return scattering_coefficient(eq, e)
end

function mass_concentrations(eq::MonochromPNEquations, e, x)
    return 1.0
end

function scattering_kernel(eq::MonochromPNEquations, e)
    # for now we ignore e
    return μ -> monochrom_scattering_kernel_func(eq, μ) / eq.scattering_norm_factor
end