@concrete struct PNEquations <: AbstractPNEquations
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
    if e == 1
        return exp(-ϵ*0.5)
    elseif e == 2
        return exp(-ϵ*0.8)
    end 
end

function absorption_coefficient(eq::PNEquations, e, ϵ)
    return scattering_coefficient(eq, e, 1, ϵ)
end

function scattering_coefficient(::PNEquations, e, i, ϵ)
    if e == 1
        return 2*exp(-ϵ*0.7)
    elseif e == 2
        return 2*exp(-ϵ*1.1)
    end 
end

function mass_concentrations(::PNEquations, e, x)
    return 1.0
end

scattering_kernel_func(μ) = exp(-5.0*(μ-1.0)^2)

function electron_scattering_kernel(eq::PNEquations, e, i, μ)
    # for now we ignore e and i
    return scattering_kernel_func(μ) / eq.scattering_norm_factor
end

@concrete struct PNExcitation
    beam_positions
    beam_energies
    beam_directions
end

number_of_beam_energies(eq::PNExcitation) = length(eq.beam_energies)
number_of_beam_positions(eq::PNExcitation) = length(eq.beam_positions)
number_of_beam_directions(eq::PNExcitation) = length(eq.beam_directions)

# SOME math FUNCTIONS
function expm2(x::Number, μ::Number, σ::Number)
    return exp(-0.5*(x - μ)^2 / σ^2)
end

function expm2(x, μ, σ)
    # try to iterate x and μ
    return exp(-sum((0.5*(x_ - μ_)^2/σ_^2 for (x_, μ_, σ_) in zip(x, μ, σ))))
end

function beam_space_distribution(eq::PNExcitation, i, (z, x, y))
    # should not depend on z (well dirac maybe..)
    return isapprox(0.0, z)*expm2((x, y), (eq.beam_positions[i].x, eq.beam_positions[i].y), (0.1, 0.1))
end

function beam_direction_distribution(eq::PNExcitation, i, Ω)
    return pdf(VonMisesFisher([eq.beam_directions[i]...], 10.0), [Ω...])
end

function beam_energy_distribution(eq::PNExcitation, i, ϵ)
    return expm2(ϵ, eq.beam_energies[i], 0.1)
end

@concrete struct PNExtraction
    extraction_energies
    pn_eq
end

number_of_extractions(eq::PNExtraction) = number_of_elements(eq.pn_eq)

function extraction_space_distribution(eq::PNExtraction, i, x)
    mass_concentrations(eq.pn_eq, i, x)
end

function extraction_direction_distribution(eq::PNExtraction, i, Ω)
    return 1.0
end

function extraction_energy_distribution(eq::PNExtraction, i, ϵ)
    if ϵ < eq.extraction_energies[i]
        return 0.0
    else
        return sqrt(ϵ - eq.extraction_energies[i])
    end
end
