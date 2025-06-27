using Plots
using HCubature

# implement the analytical solution along one dimension according to my bachelor's thesis: (x==ρ)

σ = 5.670374 * 10^-8 # Stefan Boltzmann constant

function I_analytical(x::Float64, θ::Float64, ϕ::Float64; r=0.01, R=0.03, β_i =10^3, β_o=10^-1, T_i=1000, T_o=300, T_w=300)
    I_i = σ/π * T_i^4
    I_o = σ/π * T_o^4
    I_w = σ/π * T_w^4
    if x <= r && x >= -r  # point inside the cylinder (case a from thesis)
        t_a12 = x*cos(ϕ) + sqrt(R^2 - (x*sin(ϕ))^2)
        t_a2 = x*cos(ϕ) + sqrt(r^2 - (x*sin(ϕ))^2)
        t_a1 = t_a12 - t_a2

        I_a1 = (I_w-I_o)*exp(-(β_o/sin(θ))*t_a1) + I_o
        I = (I_a1-I_i)*exp(-(β_i/sin(θ))*t_a2) + I_i
    else        # point outside of the cylinder
        ϕ_star = asin(r/x)
        if ϕ<ϕ_star     # case b from thesis
            t_b123 = x*cos(ϕ) + sqrt(R^2 - (x*sin(ϕ))^2)
            t_b23 = x*cos(ϕ) + sqrt(r^2 - (x*sin(ϕ))^2)
            t_b3 = x*cos(ϕ) - sqrt(r^2 - (x*sin(ϕ))^2)
            t_b2 = t_b23 - t_b3
            t_b1 = t_b123 - t_b23

            I_b1 = (I_w-I_o)*exp(-(β_o/sin(θ))*t_b1) + I_o
            I_b2 = (I_b1-I_i)*exp(-(β_i/sin(θ))*t_b2) + I_i
            I = (I_b2-I_o)*exp(-(β_o/sin(θ))*t_b3) + I_o
        else            # case c from thesis
            t_c = x*cos(ϕ) + sqrt(R^2 - (x*sin(ϕ))^2)
            I = (I_w-I_o)*exp(-(β_o/sin(θ))*t_c) + I_o
        end    
    end
    return I
end


function zeroth_moment(x::Float64)
    if x<0
        x = -x
    end
        
    function integrand((θ, ϕ); x=x)
        return I_analytical(x,θ,ϕ).*sin(θ)
    end
    return hcubature(integrand, (0,0), (π, 2π))[1]
end

function scaled_zeroth_moment(x::Float64 , numerical::Float64)
    y = zeroth_moment(x)
    scaling_factor = maximum(numerical)/maximum(y)
     
    return scaling_factor*y
end
