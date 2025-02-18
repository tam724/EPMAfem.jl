module NeXLCoreExt

using EPMAfem
using Dimensionless
using ConcreteStructs
using NeXLCore.Unitful
using NeXLCore
using HCubature
using LegendrePolynomials
using LinearAlgebra
using Interpolations

function Dimensionless.dimless(v::Base.TwicePrecision, bas::DimBasis)
    return Base.TwicePrecision(dimless(v.hi, bas), dimless(v.lo, bas))
end

function Dimensionless.dimless(v::StepRangeLen, bas::DimBasis)
    return StepRangeLen(dimless(v.ref, bas), dimless(v.step, bas), v.len)
end


@concrete struct EPMAEquations <: EPMAfem.AbstractPNEquations
    elements
    energy_range
    dim_basis
    scattering_approx
end

function epma_equations(elements, energy_model_units, N, number_of_scatterings)
    energy_interval = energy_model_units[end] - energy_model_units[1]
    dim_basis = DimBasis(500u"nm", energy_interval, minimum(e.density for e in elements))


    energy_model_dimless = dimless(energy_model_units, dim_basis)
    
    ## process scattering cross section
    A = zeros(length(energy_model_units), N+1)

    scattering_approx = []

    for (e, elm) in enumerate(elements)
        @show elm
        for (i_n, n) in enumerate(0:N)
            for (i_ϵ, ϵ) in enumerate(energy_model_units)
                ϵ_eV = ϵ / u"eV" |> upreferred
                A[i_ϵ, i_n] = dimless(hquadrature(μ -> NeXLCore.δσδΩ(NeXLCore.ScreenedRutherford, acos(μ), elm, Float64(ϵ_eV))*Pl.(μ, n), -1, 1)[1]*u"cm"^2 / elm.atomic_mass, dim_basis)
            end
        end

        S, V, D = svd(A)
        E = S*Diagonal(V)*Diagonal(D[1, :])
        @show D[1, :]
        K = inv(Diagonal(D[1, :])) * transpose(D)

        @show maximum(abs.(E[:, 1:number_of_scatterings]*K[1:number_of_scatterings, :] .- A))
        @show V
        
        push!(scattering_approx, [])

        for i_s in 1:number_of_scatterings
            σ_i = scale(interpolate(E[:, i_s], BSpline(Linear())), energy_model_dimless)
            k_i = EPMAfem.SphericalHarmonicsModels.LegendreBasisExp(K[i_s, :])
            push!(scattering_approx[e], (σ_i, k_i))
        end
    end

    return EPMAEquations(elements, nothing, dim_basis, scattering_approx)
end

function EPMAfem.number_of_elements(eq::EPMAEquations)
    return length(eq.elements)
end

function EPMAfem.number_of_scatterings(eq::EPMAEquations)
    length(eq.scattering_approx[1])
end

function stopping_power_units(eq::EPMAEquations, e, ϵ)
    ϵ_eV = ϵ / u"eV" |> upreferred # convert energy to eV
    s = -NeXLCore.dEds(NeXLCore.Bethe, ϵ_eV, eq.elements[e], 1.0)*u"eV"*u"cm"^2/u"g"
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
    for i_s in 1:number_of_scatterings(eq)
        val += eq.scattering_approx[e][i_s][1](ϵ)*eq.scattering_approx[e][i_s][2](μ)
    end
    return val
end

function σₜ_recon(eq::EPMAEquations, e, ϵ)
    val = 0.0
    for i_s in 1:number_of_scatterings(eq)
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


end