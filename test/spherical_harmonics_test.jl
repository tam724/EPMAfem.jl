module SphericalHarmonicsModelsTest

using Test
using LinearAlgebra
using Random
using HCubature
import EPMAfem.SphericalHarmonicsModels as SH
using EPMAfem.Dimensions


function test_spherical_cartesian_conversions(; z, x, y, θ_, ϕ_)
    Ω = SH.VectorValue(z, x, y)
    @test unitsphere_cartesian_to_spherical(Ω)[1] ≈ θ_ atol=1e-10
    @test unitsphere_cartesian_to_spherical(Ω)[2] ≈ ϕ_ atol=1e-10
    θ, ϕ = unitsphere_cartesian_to_spherical(Ω)
    @test z ≈ unitsphere_spherical_to_cartesian((θ, ϕ))[1] atol=1e-10
    @test x ≈ unitsphere_spherical_to_cartesian((θ, ϕ))[2] atol=1e-10
    @test y ≈ unitsphere_spherical_to_cartesian((θ, ϕ))[3] atol=1e-10
end

function test_polar_cartesian_conversions(; z, x, θ_)
    Ω = SH.VectorValue(z, x)
    @test unitcircle_cartesian_to_polar(Ω) ≈ θ_ atol=1e-10
    θ = unitcircle_cartesian_to_polar(Ω)
    @test z ≈ unitcircle_polar_to_cartesian(θ)[1] atol=1e-10
    @test x ≈ unitcircle_polar_to_cartesian(θ)[2] atol=1e-10
end

test_spherical_cartesian_conversions(z=1.0, x=0.0, y=0.0, θ_=0.0, ϕ_=0.0)
test_spherical_cartesian_conversions(z=0.0, x=1.0, y=0.0, θ_=π/2, ϕ_=0.0)
test_spherical_cartesian_conversions(z=0.0, x=0.0, y=1.0, θ_=π/2, ϕ_=π/2)

test_polar_cartesian_conversions(z=1.0, x=0.0, θ_=0.0)
test_polar_cartesian_conversions(z=0.0, x=1.0, θ_=π/2)
test_polar_cartesian_conversions(z=-1/sqrt(2), x=-1/sqrt(2), θ_=-3π/4)

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

function test_circular_harmonic_evenodd_classification(; N, ND, Ω)
    model = SH.EOCircularHarmonicsModel(N, ND)
    even_moms = SH.even(model)
    odd_moms = SH.odd(model)
    Y_even, Y_odd = even_moms(Ω), odd_moms(Ω)
    Y_even_, Y_odd_ = copy(Y_even), copy(Y_odd)

    Y_even, Y_odd = even_moms(-Ω), odd_moms(-Ω)

    @test all(isapprox(Y_even_, Y_even, atol=1-13))
    @test all(isapprox(Y_odd_, -Y_odd, atol=1-13))
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
    Ωs = [SH.VectorValue(randn(2) |> normalize) for _ in 1:10]
    for ND in (1, 2)
        for N in (1, 11, 21)
            for Ω in Ωs
                test_circular_harmonic_evenodd_classification(N=N, ND=ND, Ω=Ω)
            end
        end
    end
end

function test_transport_matrix_assembly(D, N, ND)
    if D == 3
        model = SH.EOSphericalHarmonicsModel(N, ND)
        integrals = (SH.∫S²_Ωzuv, SH.∫S²_Ωxuv, SH.∫S²_Ωyuv)
    else
        @assert D == 2
        model = SH.EOCircularHarmonicsModel(N, ND)
        integrals = (SH.∫S²_Ωzuv, SH.∫S²_Ωxuv)
    end

    # we use all moments, not just the even or odd ones
    U = SH.get_basis_harmonics(model)
    V = SH.get_basis_harmonics(model)

    for ∫ in integrals
        @testset "$(∫), $(D), $(N), $(ND)" begin
            A_exact = SH.assemble_bilinear(∫, model, U, V, SH.ExactQuadrature{D}())
            A_cubature = SH.assemble_bilinear(∫, model, U, V, SH.HCubatureQuadrature{D}(1e-5, 1e-5))
            @test all(isapprox.(A_cubature, A_exact, atol=1e-4))

            A_lebedev = SH.assemble_bilinear(∫, model, U, V, SH.LebedevQuadrature{D}())
            @test all(isapprox.(A_lebedev, A_exact, atol=1e-4))
        end     
    end
end

function test_boundary_matrix_assembly(D, N, ND)
    if D == 3
        model = SH.EOSphericalHarmonicsModel(N, ND)
        integrals = (SH.∫S²_absΩzuv, SH.∫S²_absΩxuv, SH.∫S²_absΩyuv)
    else
        model = SH.EOCircularHarmonicsModel(N, ND)
        integrals = (SH.∫S²_absΩzuv, SH.∫S²_absΩxuv)
    end
    
    U = SH.even(model)
    V = SH.even(model)
    
    for ∫ in integrals
        @testset "$(∫), $(D), $(N), $(ND)" begin
            A_exact = SH.assemble_bilinear(∫, model, U, V, SH.ExactQuadrature{D}())
            A_cubature = SH.assemble_bilinear(∫, model, U, V, SH.HCubatureQuadrature{D}(1e-5, 1e-5))
            @test all(isapprox.(A_cubature, A_exact, atol=1e-4))

            # for lebedev we use a very high order (otherwise the integral is inexact, lebedev is efficient anyways..)
            A_lebedev = SH.assemble_bilinear(∫, model, U, V, SH.LebedevQuadrature{D}())
            @test all(isapprox.(A_lebedev, A_exact, atol=1e-2)) # lebedev is not very good for the discontinuity here..
        end
    end
end

let
    for D in (2, 3)
        for ND in (D==3 ? (1, 2, 3) : (1, 2))
            for N in (1, 5, 11)
                test_transport_matrix_assembly(D, N, ND)
            end

            for N in (1, 5, 11)
                test_boundary_matrix_assembly(D, N, ND)
            end
        end
    end
end

function test_scattering_kernel_integration(D)
    scattering_kernel_func(μ) = exp(-5.0*(μ-1.0)^2)

    if D == 3
        model = SH.EOSphericalHarmonicsModel(11, 3)
        scattering_norm_factor = 2*π*hquadrature(x -> scattering_kernel_func(x), -1.0, 1.0)[1]
    else
        model = SH.EOCircularHarmonicsModel(11, 2)
        scattering_norm_factor = hquadrature(x -> scattering_kernel_func(x), -1.0, 1.0)[1]
    end

    U = SH.get_basis_harmonics(model)
    V = SH.get_basis_harmonics(model)
    
    scattering_kernel(μ) = scattering_kernel_func(μ) / scattering_norm_factor
    
    A_1D_expand = SH.assemble_bilinear(SH.∫S²_kuv(scattering_kernel), model, U, V, SH.HCubatureQuadrature{D}(1e-5, 1e-5, 5000)) #quadrature only used in 1D
    A_full = SH.assemble_bilinear(SH.∫∫S²_kuv((Ω1, Ω2) -> scattering_kernel(dot(Ω1, Ω2))), model, U, V, SH.LebedevQuadrature{D}())
    # also test against full HCubature ? 

    @test all(isapprox(A_1D_expand, A_full, atol=1e-6, rtol=1e-6))
end

test_scattering_kernel_integration(2)
test_scattering_kernel_integration(3)

function test_real_spherical_harmonics_function_definition()
    moms = SH.spherical_harmonics(5, SH.Dimensions._3D())
    for mom in moms
        for _ in 1:50 # test 50 random directions ∈ S^2
            Ω = randn(3) |> normalize
            Ylm_Ω = SH.eval_naive(mom, SH.VectorValue(Ω...))
            Ylm_Ω2 = mom(SH.VectorValue(Ω...))
            @assert Ylm_Ω ≈ Ylm_Ω2
            Ylm_mΩ = SH.eval_naive(mom, -SH.VectorValue(Ω...))
            Ylm_mΩ2 = mom(-SH.VectorValue(Ω...))
            if SH.is_even(mom)
                @assert Ylm_Ω ≈ Ylm_mΩ
                @assert Ylm_Ω2 ≈ Ylm_mΩ2
            else
                @assert SH.is_odd(mom)
                @assert Ylm_Ω ≈ -Ylm_mΩ
                @assert Ylm_Ω2 ≈ -Ylm_mΩ2
            end
        end
    end
end

test_real_spherical_harmonics_function_definition()


end
