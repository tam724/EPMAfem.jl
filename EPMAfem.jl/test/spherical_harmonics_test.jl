module SphericalHarmonicsModelsTest

using Test
using LinearAlgebra
using Random
import EPMAfem.SphericalHarmonicsModels as SH


function test_spherical_cartesian_conversions(; z, x, y, θ_, ϕ_)
    Ω = SH.VectorValue(z, x, y)
    @test SH.unitsphere_cartesian_to_spherical(Ω)[1] ≈ θ_ atol=1e-10
    @test SH.unitsphere_cartesian_to_spherical(Ω)[2] ≈ ϕ_ atol=1e-10
    θ, ϕ = SH.unitsphere_cartesian_to_spherical(Ω)
    @test z ≈ SH.unitsphere_spherical_to_cartesian((θ, ϕ))[1] atol=1e-10
    @test x ≈ SH.unitsphere_spherical_to_cartesian((θ, ϕ))[2] atol=1e-10
    @test y ≈ SH.unitsphere_spherical_to_cartesian((θ, ϕ))[3] atol=1e-10
end

test_spherical_cartesian_conversions(z=1.0, x=0.0, y=0.0, θ_=0.0, ϕ_=0.0)
test_spherical_cartesian_conversions(z=0.0, x=1.0, y=0.0, θ_=π/2, ϕ_=0.0)
test_spherical_cartesian_conversions(z=0.0, x=0.0, y=1.0, θ_=π/2, ϕ_=π/2)

function test_spherical_harmonic_evenodd_classification(; N, ND, Ω)
    model = SH.EOSphericalHarmonicsModel(N, ND)
    even_idx = SH.even(model)
    odd_idx = SH.odd(model)
    Y_even, Y_odd = SH._eval_basis_functions!(model, Ω, even_idx, odd_idx)
    Y_even_ = copy(Y_even)
    Y_odd_ = copy(Y_odd)
    Y_even, Y_odd = SH._eval_basis_functions!(model, -Ω, even_idx, odd_idx)

    @test all(isapprox(Y_even_, Y_even, atol=1e-13))
    @test all(isapprox(Y_odd_, -Y_odd, atol=1e-13))
end

let
    Random.seed!(12345)
    Ωs = [SH.VectorValue(randn(3) |> normalize) for _ in 1:10]
    for ND in (1, 2, 3)
        for N in (1, 11, 21)
            for Ω in Ωs
                test_spherical_harmonic_evenodd_classification(N=N, ND=ND, Ω=Ω)
            end
        end
    end
end

function test_transport_matrix_assembly(N, ND)
    model = SH.EOSphericalHarmonicsModel(N, ND)
    # we use all moments, not just the even or odd ones
    U = model.sh_index
    V = model.sh_index
    
    for ∫ in (SH.∫S²_Ωzuv, SH.∫S²_Ωxuv, SH.∫S²_Ωyuv)
        A_lebedev = SH.assemble_bilinear(∫, model, U, V, SH.lebedev_quadrature)
        A_exact = SH.assemble_bilinear(∫, model, U, V, SH.exact_quadrature)
        @test all(isapprox.(A_lebedev, A_exact, atol=1e-10))
    end
end

function test_boundary_matrix_assembly(N, ND)
    model = SH.EOSphericalHarmonicsModel(N, ND)
    # we use all moments, not just the even or odd ones
    U = SH.even(model)
    V = SH.even(model)
    
    for ∫ in (SH.∫S²_absΩzuv, SH.∫S²_absΩxuv, SH.∫S²_absΩyuv)
        A_cubature = SH.assemble_bilinear(∫, model, U, V, SH.hcubature_quadrature)
        A_exact = SH.assemble_bilinear(∫, model, U, V, SH.exact_quadrature)
        @test all(isapprox.(A_cubature, A_exact, atol=1e-8))
    end
end

let
    for ND in (1, 2, 3)
        for N in (1, 11, 21)
            test_transport_matrix_assembly(N, ND)
        end

        for N in (1, 5)
            test_boundary_matrix_assembly(N, ND)
        end
    end
end

end