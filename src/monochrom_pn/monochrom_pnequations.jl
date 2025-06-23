# so we can use the DirectionDiscretization with multiple scattering
number_of_scatterings(::AbstractMonochromPNEquations) = 1
scattering_coefficient(eq::AbstractMonochromPNEquations, e, i) = begin @assert i == 1; scattering_coefficient(eq, e) end
scattering_kernel(eq::AbstractMonochromPNEquations, e, i) = begin @assert i == 1; scattering_kernel(eq, e) end

@concrete struct DefaultMonochromPNEquations <: AbstractMonochromPNEquations
    scattering_norm_factor
end

monochrom_scattering_kernel_func(::DefaultMonochromPNEquations, μ) = exp(-10.0*(μ-1.0)^2)

function MonochromPNEquations()
    dummy_eq = DefaultMonochromPNEquations(1.0)
    scattering_norm_factor = 2*π*hquadrature(x -> monochrom_scattering_kernel_func(dummy_eq, x), -1.0, 1.0)[1]
    return DefaultMonochromPNEquations(scattering_norm_factor)
end

function number_of_elements(eq::DefaultMonochromPNEquations)
    return 2
end


function scattering_coefficient(eq::DefaultMonochromPNEquations, e)
    return e == 1 ? 0.5 : 0.1
end

function absorption_coefficient(eq::DefaultMonochromPNEquations, e)
    # make it a balance
    return e == 1 ? 1.0 : 1.0
    return e == 1 ? 0.01 : 1.0
    return scattering_coefficient(eq, e)
end

function mass_concentrations(eq::DefaultMonochromPNEquations, e, x)
    if x[1] < 0.0
    # if x[1] - x[2] < 0
        return e == 1 ? 1.0 : 0.0
    else
        return e == 1 ? 0.0 : 1.0
    end
end

function scattering_kernel(eq::DefaultMonochromPNEquations, e)
    # for now we ignore e
    return μ -> monochrom_scattering_kernel_func(eq, μ) / eq.scattering_norm_factor
end
