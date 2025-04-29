
@concrete struct DegeneratePNEquations <: AbstractDegeneratePNEquations
    Ω
end

function direction(eq::DegeneratePNEquations)
    return eq.Ω
end

function number_of_elements(eq::DegeneratePNEquations)
    return 2
end

function absorption_coefficient(eq::DegeneratePNEquations, e)
    # make it a balance
    return 0.5
    # return e == 1 ? 1.0 : 1.0
    # return e == 1 ? 1.0 : 1.0
    # return e == 1 ? 0.01 : 1.0
    # return scattering_coefficient(eq, e)
end

function mass_concentrations(eq::DegeneratePNEquations, e, x)
    if x[1]*x[1] + x[2]*x[2] > 0.1^2
    # if x[1] - x[2] < 0
        return e == 1 ? 0.0 : 0.0
    else
        return e == 1 ? 0.0 : 0.0
    end
end
