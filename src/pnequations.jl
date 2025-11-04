function stopping_power(eq::AbstractPNEquations, energy_model)
    n_elem = number_of_elements(eq)
    S = zeros(n_elem, length(energy_model))
    for e in 1:n_elem
        for (i_ϵ, ϵ) in enumerate(energy_model)
            S[e, i_ϵ] = stopping_power(eq, e, ϵ)
        end
    end
    return S
end

function absorption_coefficient(eq::AbstractPNEquations, energy_model)
    n_elem = number_of_elements(eq)
    A = zeros(n_elem, length(energy_model))
    for e in 1:n_elem
        for (i_ϵ, ϵ) in enumerate(energy_model)
            A[e, i_ϵ] += absorption_coefficient(eq, e, ϵ)
        end
    end
    return A
end


function scattering_coefficient(eq::AbstractPNEquations, energy_model)
    n_elem = number_of_elements(eq)
    n_scat = number_of_scatterings(eq)
    S = zeros(n_elem, n_scat, length(energy_model))
    for e in 1:n_elem
        for (i_ϵ, ϵ) in enumerate(energy_model)
            for i_s = 1:n_scat
                S[e, i_s, i_ϵ] = scattering_coefficient(eq, e, i_s, ϵ)
            end
        end
    end
    return S
end

# an implementation of the AbstractPNEquations
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
    # for viz:
    # return 1.0
    if e == 1
        return 1/(sqrt(5ϵ)+ 1e-10)
    elseif e == 2
        return 1/(sqrt(2ϵ)+ 1e-10)
    end 
end

function absorption_coefficient(eq::PNEquations, e, ϵ)
    return sum(scattering_coefficient(eq, e, i, ϵ) for i in 1:number_of_scatterings(eq))
end


function scattering_coefficient(::PNEquations, e, i, ϵ)
    # for viz:
    # return 20.0
    if e == 1
        return 5*exp(-ϵ*0.7)
    elseif e == 2
        return 5*exp(-ϵ*1.1)
    end 
end

function mass_concentrations(::PNEquations, e, x)
    return 1.0
end

# for viz:
# scattering_kernel_func(μ) = exp(-100.0*(μ-1.0)^2)
scattering_kernel_func(μ) = exp(-5.0*(μ-1.0)^2)

function scattering_kernel(eq::PNEquations, e, i)
    # for now we ignore e and i
    return μ -> scattering_kernel_func(μ) / eq.scattering_norm_factor
end

@concrete struct PNExcitation
    beam_positions
    beam_position_σ

    beam_energies
    beam_energy_σ

    beam_directions
    beam_direction_κ
end

function pn_excitation(beam_positions, beam_energies, beam_directions; beam_position_σ=0.1, beam_direction_κ=10.0, beam_energy_σ=0.1)
    return PNExcitation(beam_positions, beam_position_σ, beam_energies, beam_energy_σ, beam_directions, beam_direction_κ)
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
    return isapprox(0.0, z, atol=1e-12)*expm2((x, y), (eq.beam_positions[i].x, eq.beam_positions[i].y), (eq.beam_position_σ, eq.beam_position_σ))
end

function beam_direction_distribution(eq::PNExcitation, i, Ω)
    return pdf(VonMisesFisher([eq.beam_directions[i]...], eq.beam_direction_κ), [Ω...])
end

function beam_energy_distribution(eq::PNExcitation, i, ϵ)
    # return 0.5+0.5*tanh(-eq.beam_energy_σ*(ϵ-eq.beam_energies[i])) # 
    return expm2(ϵ, eq.beam_energies[i], eq.beam_energy_σ)
end

function compute_influx(eq::PNExcitation, mdl, ϵ_func=ϵ->1.0)
    influx = zeros(number_of_beam_energies(eq), number_of_beam_positions(eq), number_of_beam_directions(eq))
    quad = SphericalHarmonicsModels.lebedev_quadrature_max()
    n_ = VectorValue(1.0, 0.0, 0.0) #assuming outwards normal
    dir_influx = [quad(Ω -> dot(n_, Ω) <= 0 ? dot(n_, Ω)*beam_direction_distribution(eq, i, Ω) : 0.0) for i in 1:number_of_beam_directions(eq)]
    space_mdl = space_model(mdl)
    error("TODO: even/plus etc.")
    space_influx = [SpaceModels.assemble_linear(SpaceModels.∫∂R_ngv{Dimensions.Z}(x -> beam_space_distribution(eq, i, Dimensions.extend_3D(x))), space_mdl, SpaceModels.even(space_mdl))|>sum for i in 1:number_of_beam_positions(eq)]
    function trapz_quad(f, xs)
        v = [f(x) for x in xs]
        Δx = step(xs)
        return Δx*(sum(v[2:end-1]) + 0.5*(v[1] + v[end]))
    end
    energy_influx = [trapz_quad(ϵ -> ϵ_func(ϵ)*beam_energy_distribution(eq, i, ϵ), energy_model(mdl)) for i in 1:number_of_beam_energies(eq)]
    for i in 1:number_of_beam_energies(eq), j in 1:number_of_beam_positions(eq), k in 1:number_of_beam_directions(eq)
        influx[i, j, k] = energy_influx[i]*space_influx[j]*dir_influx[k]
    end
    return influx
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
