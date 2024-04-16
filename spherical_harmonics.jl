module SphericalHarmonicsMatrices

    using SparseArrays
    using SphericalHarmonics
    using HCubature
    using LinearAlgebra
    using LegendrePolynomials


    ## advection matrices
    function Θ(k::Int)::Int
        if k < 0
            return -1
        elseif k >= 0
            return 1
        end
    end

    function plus(k::Int)::Int
        if k == 0
            return 1
        end
        return k + sign(k)
    end

    function minus(k::Int)::Int
        if k == 0
            return -1
        end
        return k - sign(k)
    end

    function a(k::Int, l::Int)::Float64
        return sqrt((l - k + 1) * (l + k + 1) / ((2 * l + 3) * (2 * l + 1)))
    end

    function b(k::Int, l::Int)::Float64
        return sqrt((l - k) * (l + k) / ((2 * l + 1) * (2 * l - 1)))
    end

    function c(k::Int, l::Int)::Float64
        c = sqrt((l + k + 1) * (l + k + 2) / ((2 * l + 3) * (2 * l + 1)))
        if k < 0
            return 0
        elseif k > 0
            return c
        else
            return c * sqrt(2)
        end
    end

    function d(k::Int, l::Int)::Float64
        d = sqrt((l - k) * (l - k - 1) / ((2 * l + 1) * (2 * l - 1)))
        if k < 0
            return 0
        elseif k > 0
            return d
        else
            return d * sqrt(2)
        end
    end

    function e(k::Int, l::Int)::Float64
        e = sqrt((l - k + 1) * (l - k + 2) / ((2 * l + 3) * (2 * l + 1)))
        if k == 1
            return e * sqrt(2)
        elseif k > 1
            return e
        else
            error("k < 1")
        end
    end

    function f(k::Int, l::Int)::Float64
        f = sqrt((l + k) * (l + k - 1) / ((2 * l + 1) * (2 * l - 1)))
        if k == 1
            return f * sqrt(2)
        elseif k > 1
            return f
        else
            error("k < 1")
        end
    end

    function A_minus(l::Int, k::Int, k´::Int, ::Val{1})::Float64
        if k´ == minus(k) && k != -1
            return 1 / 2 * c(abs(k) - 1, l - 1)
        elseif k´ == plus(k)
            return -1 / 2 * e(abs(k) + 1, l - 1)
        else
            return 0
        end
    end

    function A_minus(l::Int, k::Int, k´::Int, ::Val{2})::Float64
        if k´ == -minus(k) && k != 1
            return -Θ(k) / 2 * c(abs(k) - 1, l - 1)
        elseif k´ == -plus(k)
            return -Θ(k) / 2 * e(abs(k) + 1, l - 1)
        else
            return 0
        end
    end

    function A_minus(l::Int, k::Int, k´::Int, ::Val{3})::Float64
        if k´ == k
            return a(k, l - 1)
        else
            return 0
        end
    end

    function A_plus(l::Int, k::Int, k´::Int, ::Val{1})::Float64
        if k´ == minus(k) && k != -1
            return -1 / 2 * d(abs(k) - 1, l + 1)
        elseif k´ == plus(k)
            return 1 / 2 * f(abs(k) + 1, l + 1)
        else
            return 0
        end
    end

    function A_plus(l::Int, k::Int, k´::Int, ::Val{2})::Float64
        if k´ == -minus(k) && k != 1
            return Θ(k) / 2 * d(abs(k) - 1, l + 1)
        elseif k´ == -plus(k)
            return Θ(k) / 2 * f(abs(k) + 1, l + 1)
        else
            return 0
        end
    end

    function A_plus(l::Int, k::Int, k´::Int, ::Val{3})::Float64
        if k´ == k
            return b(k, l + 1)
        else
            return 0
        end
    end

    function get_coefficient(l, k, l_, k_, dim)::Float64
        val = 0.0
        if l == l_- 1
            # take Aplus
            val = A_plus(l, k, k_, dim)
        elseif l == l_ + 1
            # take Aminus
            val = A_minus(l, k, k_, dim)
        else
            val = 0.0
        end
        if val == -0.0
            val = 0.0
        end
        return val
    end

    function get_moments(N, ::Val{1})
        # only allocate moments for z
        moments = [(l, k) for l in 0:N for k in -l:l if (!is_odd_in(l, k, Val{1}()) || !is_odd_in(l, k, Val{2}()))]
        return moments
    end

    function get_moments(N, ::Val{2})
        # only allocate moments for x and z
        moments = [(l, k) for l in 0:N for k in -l:l if !is_odd_in(l, k, Val{2}())]
        return moments
    end

    function get_moments(N, ::Val{3})
        # allocate moments for x, y, and z
        moments = [(l, k) for l in 0:N for k in -l:l]
        return moments
    end

    function is_even_in(l, k, ::Val{1})
        return (k < 0 && isodd(k)) || (k >= 0 && iseven(k))
    end
    
    function is_even_in(l, k, ::Val{2})
        return k >= 0
    end
    
    function is_even_in(l, k, ::Val{3})
        return iseven(l+k)
    end

    function is_odd_in(l, k, d::Val{D}) where D
        return !is_even_in(l, k, d)
    end

    function is_even(l, k)
        return mod(l, 2) == 0
    end

    function is_odd(l, k)
        return !is_even(l, k)
    end

    function get_eo_moments(N, nd::Val{ND}) where ND
        even_moments = [m for m in get_moments(N, nd) if is_even(m...)]
        odd_moments = [m for m in get_moments(N, nd) if !is_even(m...)]
        return vcat(even_moments, odd_moments)
    end

    function assemble_transport_matrix(N, dim, parity, nd::Val{ND}) where ND
        if parity == :pm
            eo_moms1 = [m for m ∈ get_eo_moments(N, nd) if is_even(m...)]
            eo_moms2 = [m for m ∈ get_eo_moments(N, nd) if is_odd(m...)]
        else
            eo_moms1 = [m for m ∈ get_eo_moments(N, nd) if is_odd(m...)]
            eo_moms2 = [m for m ∈ get_eo_moments(N, nd) if is_even(m...)]
        end
        # eo_moms = get_eo_moments(N, nd)
        A = spzeros(length(eo_moms2), length(eo_moms1))
        for (i, (l, k)) = enumerate(eo_moms1)
            for (j, (l_, k_)) = enumerate(eo_moms2)
                c = get_coefficient(l, k, l_, k_, dim)
                if c != 0
                    A[j, i] = get_coefficient(l, k, l_, k_, dim)
                end
            end
        end
        return A
    end

    function assemble_boundary_matrix(N, ::Val{D}, parity, nd::Val{ND}) where {D, ND}
        function integrand((θ, ϕ))
            abs_Ω = abs((sin(θ)*cos(ϕ), sin(θ)*sin(ϕ), cos(θ))[D])
            Y = computeYlm(θ, ϕ, lmax=N, SHType=SphericalHarmonics.RealHarmonics())
            if parity == :pp
                Y_e = [Y[m] for m ∈ get_eo_moments(N, nd) if is_even(m...)]
            else
                Y_e = [0]
            end
            return abs_Ω * ((Y_e' .* Y_e))
        end
        A = hcubature(x -> integrand(x)*sin(x[1]), (0, 0), (π, 2π), rtol=1e-8, atol=1e-8, maxevals=100000)[1]
        A = round.(A, digits=10)
        return sparse(A)
    end

    function assemble_scattering_matrix(N, scattering_kernel, nd::Val{ND}) where ND
        eo_moms = get_eo_moments(N, nd)
        Σl = 2*π*hquadrature(x -> scattering_kernel(x).*Pl.(x, 0:N), -1.0, 1.0, rtol=1e-8, atol=1e-8, maxevals=100000)[1]
        return Diagonal([Σl[l+1] for (l, k) in eo_moms])
    end

    function assemble_direction_rhs(N, qΩ, nd::Val{ND}) where ND
        function integrand((θ, ϕ))
            Ω = (sin(θ)*cos(ϕ), sin(θ)*sin(ϕ), cos(θ))
            Y = computeYlm(θ, ϕ, lmax=N, SHType=SphericalHarmonics.RealHarmonics())
            Y_eo = [Y[m] for m ∈ get_eo_moments(N, nd)]
            return qΩ(Ω)*Y_eo
        end
        return hcubature(x -> integrand(x)*sin(x[1]), (0, 0), (π, 2π))[1]
    end

    function assemble_gram_matrix(N, nd::Val{ND}) where ND
        function integrand((θ, ϕ))
            # Ω = (sin(θ)*cos(ϕ), sin(θ)*sin(ϕ), cos(θ))
            Y = computeYlm(θ, ϕ, lmax=N, SHType=SphericalHarmonics.RealHarmonics())
            Y_eo = [Y[m] for m ∈ get_eo_moments(N, nd)]
            return Y_eo*Y_eo'
        end
        return hcubature(x -> integrand(x)*sin(x[1]), (0, 0), (π, 2π))[1]
    end

    function cut_pos(x)
        return x <= 0.0 ? x : 0.0
    end

    function assemble_boundary_source(N, g_Ω, n, nd::Val{ND}) where {ND}
        function integrand((θ, ϕ))
            Ω = (sin(θ)*cos(ϕ), sin(θ)*sin(ϕ), cos(θ))
            Y = computeYlm(θ, ϕ, lmax=N, SHType=SphericalHarmonics.RealHarmonics())
            Y_eo_even = [is_even(m...) ? Y[m] : 0.0 for m ∈ get_eo_moments(N, nd)]
            return cut_pos(dot(Ω, n))*g_Ω(Ω)*Y_eo_even
        end
        return hcubature(x -> integrand(x)*sin(x[1]), (0, 0), (π, 2π))[1]
    end
    
    export assemble_transport_matrix, assemble_boundary_matrix, assemble_scattering_matrix, assemble_direction_rhs
end
#### FOR TESTING

# A = assemble_transport_matrix(3, Val{3}())
# K = assemble_scattering_matrix(3, scattering)

# N = 3

# function integrand((θ, ϕ), ::Val{D}) where D
#     Y = computeYlm(θ, ϕ, lmax=N, SHType=SphericalHarmonics.RealHarmonics())
#     Y_eo = [Y[m] for m ∈ get_eo_moments(N)]
#     Ω = (sin(θ)*cos(ϕ), sin(θ)*sin(ϕ), cos(θ))
#     return Ω[D]*(Y_eo' .* Y_eo)
# end

# function scattering(μ) 
#     # @show μ
#     @assert -1.0001 <= μ <= 1.0001
#     σ = 0.1
#     return exp(-0.5*μ^2/(σ*σ))/(sqrt(2π)σ)
# end



# using Plots
# plot(-1:0.01:1, scattering)

# function integrand_2((θ, ϕ, θ_, ϕ_))
#     Y = computeYlm(θ, ϕ, lmax=N, SHType=SphericalHarmonics.RealHarmonics())
#     Y_ = computeYlm(θ_, ϕ_, lmax=N, SHType=SphericalHarmonics.RealHarmonics())
#     Y_eo = [Y[m] for m ∈ get_eo_moments(N)]
#     Y_eo_ = [Y_[m] for m ∈ get_eo_moments(N)]
#     Ω = (sin(θ)*cos(ϕ), sin(θ)*sin(ϕ), cos(θ))
#     Ω_ = (sin(θ_)*cos(ϕ_), sin(θ_)*sin(ϕ_), cos(θ_))
#     return scattering(dot(Ω, Ω_))*(Y_eo .* Y_eo_)
# end


# A_x = hcubature(x -> integrand(x, Val{1}())*sin(x[1]), (0, 0), (π, 2π))[1]
# A_x2 = assemble_transport_matrix(N, Val{1}())

# round.(A_x, digits=4)
# round.(Matrix(A_x2), digits=4)

# abs.(A_x .- A_x2) |> maximum

# K = hcubature(x -> integrand_2(x)*sin(x[1])*sin(x[3]), (0, 0, 0, 0), (π, 2π, π, 2π), rtol=1e-4, atol=1e-4, maxevals=1000000)[1]
# K2 = assemble_scattering_matrix(N, scattering)