@concrete struct EPMAEquations <: EPMAfem.AbstractPNEquations
    elements
    dim_basis
    scattering_approx
    energy_model_dimless
    elastic_scattering_cross_section
    bethe_energy_loss
end

function epma_equations(elements, energy_model_units, PN_N; elastic_scattering_cross_section=NeXLCore.Liljequist1989, bethe_energy_loss=NeXLCore.JoyLuo, ϵ_rel=1e-6)
    energy_interval = energy_model_units[end] - energy_model_units[1]
    dim_basis = DimBasis(500u"nm", energy_interval, minimum(e.density for e in elements))

    energy_model_dimless = dimless(energy_model_units, dim_basis)
    
    ## process scattering cross section
    As = [zeros(length(energy_model_units), PN_N+1) for e in 1:length(elements)]
    for (e, elm) in enumerate(elements)
        for (i_n, n) in enumerate(0:PN_N)
            for (i_ϵ, ϵ) in enumerate(energy_model_units)
                ϵ_eV = ϵ / u"eV" |> upreferred
                As[e][i_ϵ, i_n] = dimless(hquadrature(μ -> NeXLCore.δσδΩ(elastic_scattering_cross_section, acos(μ), elm, Float64(ϵ_eV))*Pl.(μ, n), -1, 1, maxevals=100000)[1]*u"cm"^2 / elm.atomic_mass, dim_basis)
            end
        end
    end
    rank, SVDs = compute_svd_approx(As, ϵ_rel)
    EKs = truncate_svd_and_normalize.(SVDs, rank)
    for (e, (E, K)) in enumerate(EKs)
        @show norm(E*K .- As[e]), ϵ_rel*norm(As[e]), norm(As[e])
        @assert norm(E*K .- As[e]) <= ϵ_rel*norm(As[e])
    end
    scattering_approx = [build_scattering_approximation(E, K, energy_model_dimless, rank) for (E, K) in EKs]

    return EPMAEquations(elements, dim_basis, scattering_approx, energy_model_dimless, elastic_scattering_cross_section, bethe_energy_loss)
end

function EPMAfem.number_of_elements(eq::EPMAEquations)
    return length(eq.elements)
end

function EPMAfem.number_of_scatterings(eq::EPMAEquations)
    length(eq.scattering_approx[1])
end

function stopping_power_units(eq::EPMAEquations, e, ϵ)
    ϵ_eV = ϵ / u"eV" |> upreferred # convert energy to eV
    s = -NeXLCore.dEds(eq.bethe_energy_loss, ϵ_eV, eq.elements[e], 1.0)*u"eV"*u"cm"^2/u"g"
    return s
end

function EPMAfem.stopping_power(eq::EPMAEquations, e, ϵ)
    s = stopping_power_units(eq, e, dimful(ϵ, u"eV", eq.dim_basis))
    return dimless(s, eq.dim_basis)
end

function EPMAfem.stopping_power(eq::EPMAEquations, energy_model)
    n_elem = EPMAfem.number_of_elements(eq)
    S = zeros(n_elem, length(energy_model))
    for e in 1:n_elem
        for (i_ϵ, ϵ) in enumerate(energy_model)
            # s = stopping_power_units(eq, e, dimful(ϵ, u"eV", eq.dim_basis))
            S[e, i_ϵ] = EPMAfem.stopping_power(eq, e, ϵ)
        end
    end
    return S
end

function δσδΩ_recon(eq::EPMAEquations, μ, e, ϵ)
    val = 0.0
    for i_s in 1:EPMAfem.number_of_scatterings(eq)
        val += eq.scattering_approx[e][i_s][1](ϵ)*eq.scattering_approx[e][i_s][2](μ)
    end
    return val
end

function σₜ_recon(eq::EPMAEquations, e, ϵ)
    val = 0.0
    for i_s in 1:EPMAfem.number_of_scatterings(eq)
        val += eq.scattering_approx[e][i_s][1](ϵ)
    end
    return val
end

function EPMAfem.absorption_coefficient(eq::EPMAEquations, energy_model)
    n_elem = EPMAfem.number_of_elements(eq)
    A = zeros(n_elem, length(energy_model))
    for e in 1:n_elem
        for (i_ϵ, ϵ) in enumerate(energy_model)
            for i_s = 1:EPMAfem.number_of_scatterings(eq)
                # τ = σ
                A[e, i_ϵ] += EPMAfem.scattering_coefficient(eq, e, i_s, ϵ)
            end
        end
    end
    return A
end

function EPMAfem.scattering_coefficient(eq::EPMAEquations, e, i, ϵ)
    return eq.scattering_approx[e][i][1](ϵ)
end

function EPMAfem.scattering_coefficient(eq::EPMAEquations, energy_model)
    n_elem = EPMAfem.number_of_elements(eq)
    n_scat = EPMAfem.number_of_scatterings(eq)
    S = zeros(n_elem, n_scat, length(energy_model))
    for e in 1:n_elem
        for (i_ϵ, ϵ) in enumerate(energy_model)
            for i_s = 1:n_scat
                # τ = σ
                S[e, i_s, i_ϵ] = EPMAfem.scattering_coefficient(eq, e, i_s, ϵ)
            end
        end
    end
    return S
end

function EPMAfem.mass_concentrations(eq::EPMAEquations, e, x)
    return dimless(eq.elements[1].density, eq.dim_basis)
end

function EPMAfem.electron_scattering_kernel_f(eq::EPMAEquations, e, i)
    return eq.scattering_approx[e][i][2]
end
