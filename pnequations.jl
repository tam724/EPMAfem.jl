abstract type PNEquations{NE, Ni, Ngϵ, Ngx, NgΩ, Nμϵ} end

function s(eq::PNEquations, ϵ, e)
    return 0.5 * stopping_power(eq, ϵ, e)
end

function τ(eq::PNEquations, ϵ, e)
    return total_scattering_cross_section(eq, ϵ, e) - 0.5 * ∂stopping_power(eq, ϵ, e)
end

function σ(eq::PNEquations, ϵ, e, i)
    return scattering_cross_section(eq, ϵ, e, i)
end

function ∂stopping_power(eq::PNEquations, ϵ, e)
    Enzyme.autodiff(Forward, ϵ -> stopping_power(eq, ϵ, e), Duplicated(ϵ, 1.0))[1]
end

number_of_elements(::PNEquations{NE}) where NE = NE
number_of_scatterings(::PNEquations{NE, Ni}) where {NE, Ni} = Ni
number_of_beam_energies(::PNEquations{NE, Ni, Ngϵ}) where {NE, Ni, Ngϵ} = Ngϵ
number_of_beam_positions(::PNEquations{NE, Ni, Ngϵ, Ngx}) where {NE, Ni, Ngϵ, Ngx} = Ngx
number_of_beam_directions(::PNEquations{NE, Ni, Ngϵ, Ngx, NgΩ}) where {NE, Ni, Ngϵ, Ngx, NgΩ} = NgΩ

number_of_extraction_positions(::PNEquations{NE}) where NE = NE
number_of_extraction_directions(::PNEquations) = 1

extraction_direction(::PNEquations, Ω, k) = 1

# SOME FUNCTIONS

function expm2(x::Number, μ::Number, σ::Number)
    return exp(-0.5*(x - μ)^2 / σ^2)
end

function expm2(x, μ, σ)
    # try to iterate x and μ
    return exp(-sum((0.5*(x_ - μ_)^2/σ_ for (x_, μ_, σ_) in zip(x, μ, σ))))
end

# SOME DUMMY VALUES TO TEST THE CODE WITH

struct DummyPNEquations{Ngx, Ngϵ, NgΩ} <: PNEquations{2, 1, Ngx, Ngϵ, NgΩ, 2}
    scattering_norm_factor::Float64
    gϵpos::Vector{Float64}
    gxpos::Vector{Float64}
    gΩpos::Vector{Vector{Float64}}

    μϵpos::Vector{Float64}
end

function dummy_equations(gϵpos, gxpos, gΩpos, μϵpos)
    scattering_norm_factor = 2*π*hquadrature(x -> _scattering_kernel(x), -1.0, 1.0, rtol=1e-8, atol=1e-8, maxevals=100000)[1]
    return DummyPNEquations{length(gϵpos), length(gxpos), length(gΩpos)}(
        scattering_norm_factor,
        gϵpos,
        gxpos,
        normalize.(gΩpos),
        μϵpos)
end

_scattering_kernel(μ) = exp(-20.0*(μ-0.0)^2)

function scattering_kernel(eq::DummyPNEquations, μ, e, i)
    # for now we ignore e and i
    return _scattering_kernel(μ) / eq.scattering_norm_factor
end

function stopping_power(::DummyPNEquations, ϵ, e)
    if e == 1
        return 1.0 + 0.1*exp(-ϵ)
    elseif e == 2
        return 1.0 + 0.01*exp(-ϵ)
    else
        error("index too high")
    end
    return 0.0
end

scattering_cross_section(::DummyPNEquations, ϵ, e, i) = 2.0
total_scattering_cross_section(eq::DummyPNEquations, ϵ , e) = scattering_cross_section(eq, ϵ, e, 1)

beam_energy(eq::DummyPNEquations, ϵ, i) = expm2(ϵ, eq.gϵpos[i], 0.04)
function beam_position(eq::DummyPNEquations, x, j)
    if isapprox(x[1], 1.0)
        return expm2([(length(x)>1) ? x[2] : 0.0, (length(x)>2) ? x[3] : 0.0], [eq.gxpos[j], 0.0], [0.0025, 0.0025])
    else 
        return 0.0
    end
end
beam_direction(eq::DummyPNEquations, Ω, k) = pdf(VonMisesFisher(normalize(eq.gΩpos[k]), 10.0), [Ω...])

extraction_energy(eq::DummyPNEquations, ϵ, e) = (ϵ-eq.μϵpos[e] > 0.0) ? sqrt(ϵ-eq.μϵpos[e]) : 0.0

extraction_position(eq::DummyPNEquations, x, e) = mass_concentrations(eq, x, e) # add absorption here !

function mass_concentrations(::DummyPNEquations, x, e)
    z_ = x[1]
    if length(x) == 1 # 1D version
        return 1.0
    end
    x_ = x[2]
    if length(x) == 2 # 2D version
        if abs(x_) < 0.2
            return (0.0, 1.2)[e]
        else
            return (0.8, 0.0)[e]
        end
        return 1.0
    end
    if length(x) == 3
        if x[2]^2 + x[3]^2 < 0.5^2
            return (0.0, 1.2)[e]
        else
            return (0.8, 0.0)[e]
        end
    end
    error("")
end
