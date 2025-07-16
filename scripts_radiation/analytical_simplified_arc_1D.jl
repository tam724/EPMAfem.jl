using Plots
using HCubature

# implement the analytical solution along one dimension according to my bachelor's thesis: (x==ρ)

# parameters
σ = 5.670374 * 10^-8 # Stefan Boltzmann constant
r = 0.01
R = 0.03
β_i =10^3
β_o=10^-1
T_i=1000
T_o=300
T_w=300

function I_analytical(x::Float64, θ::Float64, ϕ::Float64; r=r, R=R, β_i =β_i, β_o=β_o, T_i=T_i, T_o=T_o, T_w=T_w)
    @assert 0 <= x <= R "x must be between 0 and R"
    I_i = σ/π * T_i^4
    I_o = σ/π * T_o^4
    I_w = σ/π * T_w^4
    if x <= r  # point inside the cylinder (case a from thesis)
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
    print(numerical)
    scaling_factor = maximum(numerical)/maximum(y)
    print(scaling_factor)
     
    return scaling_factor*y
end


function slab_I_analytical(x, θ; r=r, R=R, β_i =β_i, β_o=β_o, T_i=T_i, T_o=T_o, T_w=T_w)
    @assert 0 <= x <= R "x must be between 0 and R"
    @assert 0 <= θ <= π "θ must be between 0 and π"
    @assert r < R "r must be smaller than R"

    I_i = σ/π * T_i^4
    I_o = σ/π * T_o^4
    I_w = σ/π * T_w^4
    
    if x <= r
        if θ==π/2
            return I_i
        elseif θ<π/2
            s_12 = (R+x)/cos(θ) 
            s_2 = (r+x)/cos(θ) 
            s_1 = s_12-s_2
        else  
            s_12 = (R-x)/cos(π-θ)
            s_2 = (r-x)/cos(π-θ)
            s_1 = s_12-s_2
        end
        I_1 = (I_w-I_o)*exp(-β_o*s_1)+I_o
        return (I_1-I_i)*exp(-β_i*s_2)+I_i
    else
        if θ==π/2
            return I_o
        elseif θ<π/2
            s_123 = (R+x)/cos(θ) 
            s_23 = (r+x)/cos(θ) 
            s_3 = (x-r)/cos(θ)
            s_2 = s_23-s_3
            s_1 = s_123-s_23

            I_1 = (I_w-I_o)*exp(-β_o*s_1)+I_o
            I_2 = (I_1-I_i)*exp(-β_i*s_2)+I_i
            return (I_2-I_o)*exp(-β_o*s_3)+I_o
        else    
            s = (R-x)/cos(π-θ)
            return (I_w-I_o)*exp(-β_o*s)+I_o
        end
    end
end

function slab_zeroth_moment(x::Float64)
    if x<0
        x = -x
    end
        
    function integrand((θ, ϕ); x=x)
        return slab_I_analytical(x,θ).*sin(θ)
    end
    return hcubature(integrand, (0,0), (π, 2π))[1]
end