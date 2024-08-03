abstract type EPMAEquations end

# SOME math FUNCTIONS
function expm2(x::Number, μ::Number, σ::Number)
    return exp(-0.5*(x - μ)^2 / σ^2)
end

function expm2(x, μ, σ)
    # try to iterate x and μ
    return exp(-sum((0.5*(x_ - μ_)^2/σ_^2 for (x_, μ_, σ_) in zip(x, μ, σ))))
end

function stopping_power(eq::EPMAEquations)
    return (ϵ, x) -> sum(mass_concentrations(eq, e)(x)*specific_stopping_power(eq, e)(ϵ) for e in 1:number_of_elements(eq))
end

function total_electron_scattering_cross_section(eq::EPMAEquations)
    return (ϵ, x) -> sum(mass_concentrations(eq, e)(x)*total_specific_electron_scattering_cross_section(eq, e)(ϵ) for e in 1:number_of_elements(eq))
end

function total_specific_electron_scattering_cross_section(eq::EPMAEquations, e)
    return function(ϵ)
        return sum((specific_electron_scattering_cross_section(eq, e, i)(ϵ) for i in 1:number_of_scatterings(eq)))
    end
end

# use a struct that we can dispatch on (to precompute stuff)
abstract type BeamDirection end
struct VMFBeam <: BeamDirection
    direction::SVector{3, Float64}
    κ::Float64
end
(vmf_beam::VMFBeam)(Ω) = pdf(VonMisesFisher(Vector(vmf_beam.direction), vmf_beam.κ), [Ω...])

abstract type ExtractionDirection end
struct IsotropicExtraction <: ExtractionDirection end
(::IsotropicExtraction)(Ω) = 1.0

# SOME DUMMY VALUES TO TEST THE CODE WITH

struct DummyEPMAEquations <: EPMAEquations
    scattering_norm_factor::Float64
    gϵpos::Vector{Quantity{Float64}}
    gxpos::Vector{Quantity{Float64}}
    gΩ::Vector{BeamDirection}

    μϵpos::Vector{Quantity{Float64}}
    takeoff_angles::Vector{Vector{Float64}}
end

scattering_kernel_func(μ) = exp(-5.0*(μ-1.0)^2)
function dummy_epma_equations(gϵpos, gxpos, gΩpos, κ, μϵpos)
    scattering_norm_factor = 2*π*hquadrature(x -> scattering_kernel_func(x), -1.0, 1.0)[1]
    return DummyEPMAEquations(
        scattering_norm_factor,
        gϵpos,
        gxpos,
        [VMFBeam(normalize(gΩp), κ) for gΩp ∈ gΩpos],
        μϵpos,
        [[1.0, 0.0, 0.0]])
end

number_of_elements(::DummyEPMAEquations) = 2
number_of_scatterings(::DummyEPMAEquations) = 1
number_of_beam_energies(eq::DummyEPMAEquations) = length(eq.gϵpos)
number_of_beam_positions(eq::DummyEPMAEquations) = length(eq.gxpos)
number_of_beam_directions(eq::DummyEPMAEquations) = length(eq.gΩ)

number_of_extraction_positions(eq::DummyEPMAEquations) = number_of_elements(eq)
number_of_extraction_directions(eq::DummyEPMAEquations) = 1
number_of_extraction_energies(eq::DummyEPMAEquations) = number_of_elements(eq)

function electron_scattering_kernel(eq::DummyEPMAEquations, e, i)
    # for now we ignore e and i
    return μ -> scattering_kernel_func(μ) / eq.scattering_norm_factor
end

function specific_stopping_power(::DummyEPMAEquations, e)
    u_sp = u"keV"*(u"nm")^5 / u"u"^2
    return ϵ -> (1.0u_sp, 1.0u_sp)[e]
end

function specific_electron_scattering_cross_section(::DummyEPMAEquations, e, i)
    u_scs = (u"nm")^5 / u"u"^2
    return ϵ -> (0.0u_scs, 0.0u_scs)[e]
end

beam_energy_distribution(eq::DummyEPMAEquations, i) = ϵ -> expm2(ϵ, eq.gϵpos[i], 0.04u"keV")
function beam_spatial_distribution(eq::DummyEPMAEquations, j)
    return function ((z, x, y))
        # should not depend on z (well dirac maybe..)
        isapprox(1.0u"nm", z)*expm2((x, y), (eq.gxpos[j], 0.0u"nm"), (0.05u"nm", 0.05u"nm"))
    end
end
beam_direction_distribution(eq::DummyEPMAEquations, k) = eq.gΩ[k]

extraction_energy_distribution(eq::DummyEPMAEquations, i) = ϵ -> (ϵ-eq.μϵpos[i] > 0.0u"keV") ? sqrt((ϵ-eq.μϵpos[i])/1u"keV") : 0.0
extraction_spatial_distribution(eq::DummyEPMAEquations, j) = x -> mass_concentrations(eq, j)(x) / (u"u"/u"nm"^3) # add absorption here ! this should be unitless
extraction_direction_distribution(::DummyEPMAEquations, k) = IsotropicExtraction()

function mass_concentrations(::DummyEPMAEquations, e::Int64)
    u_mc = u"u"/u"nm"^3
    return function((z, x, y))
        if abs(x) < 0.2u"nm"
            return (0.0u_mc, 1.0u_mc)[e]
        else
            return (1.0u_mc, 0.0u_mc)[e]
        end
    end
end

function energy_interval(::DummyEPMAEquations)
    return (0.0u"keV", 1.0u"keV")
end
