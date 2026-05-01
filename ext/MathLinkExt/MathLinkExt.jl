module MathLinkExt

using EPMAfem
using EPMAfem.SphericalHarmonicsModels
using EPMAfem.SphericalHarmonicsModels: CircularHarmonic, degree, order
using MathLink

function w_object(ch::CircularHarmonic)
    l, m = degree(ch), order(ch)
    norm = l == 0 ? 1/(W"Sqrt"(2*W"Pi")) : 1/(W"Sqrt"(W"Pi"))
    if m == 0
        return norm * W"Cos"(l*W"t")
    else # m == 1
        return norm * W"Sin"(l*W"t")
    end
end

function w_integrate_circle(w_object)
    return weval(W"Simplify"(W"Integrate"(w_object, (W"t", 0, 2*W"Pi"))))
end

function w_num(w_object)
    return Float64(weval(W"N"(W"Simplify"(w_object))))
end

"""
    computes ∫_{S^1} z(Ω) * m1(Ω) * m2(Ω) dΩ
"""
function get_transport_coefficient_symbolic(m1::CircularHarmonic, m2::CircularHarmonic, ::EPMAfem.Dimensions.Z)
    return w_integrate_circle(W"Cos"(W"t")*w_object(m1)*w_object(m2))
end

"""
    computes ∫_{S^1} x(Ω) * m1(Ω) * m2(Ω) dΩ
"""
function get_transport_coefficient_symbolic(m1::CircularHarmonic, m2::CircularHarmonic, ::EPMAfem.Dimensions.X)
    return w_integrate_circle(W"Sin"(W"t")*w_object(m1)*w_object(m2))
end

"""
    computes ∫_{S^1} abs(z(Ω)) * m1(Ω) * m2(Ω) dΩ
"""
function get_boundary_coefficient_symbolic(m1::CircularHarmonic, m2::CircularHarmonic, ::EPMAfem.Dimensions.Z)
    return w_integrate_circle(W"Abs"(W"Cos"(W"t"))*w_object(m1)*w_object(m2))
end

"""
    computes ∫_{S^1} abs(x(Ω)) * m1(Ω) * m2(Ω) dΩ
"""
function get_boundary_coefficient_symbolic(m1::CircularHarmonic, m2::CircularHarmonic, ::EPMAfem.Dimensions.X)
    return w_integrate_circle(W"Abs"(W"Sin"(W"t"))*w_object(m1)*w_object(m2))
end

# only integrate on the incoming half circle (where cos(t) < 0 <=> [π/2, 3π/2])
function w_integrate_halfcircle(w_object, ::EPMAfem.Dimensions.Z)
    return weval(W"Simplify"(W"Integrate"(w_object, (W"t", W"Pi"/2, 3W"Pi"/2))))
end
# only integrate on the incoming half circle (where sin(t) < 0 <=> [π, 2π])
function w_integrate_halfcircle(w_object, ::EPMAfem.Dimensions.X)
    return weval(W"Simplify"(W"Integrate"(w_object, (W"t", W"Pi", 2*W"Pi"))))
end

# only integrate on the outgoing half circle (where cos(t) > 0 <=> [-π/2, π/2])
function w_integrate_halfcircle_out(w_object, ::EPMAfem.Dimensions.Z)
    return weval(W"Simplify"(W"Integrate"(w_object, (W"t", -W"Pi"/2, W"Pi"/2))))
end
# only integrate on the outgoing half circle (where sin(t) > 0 <=> [0, π])
function w_integrate_halfcircle_out(w_object, ::EPMAfem.Dimensions.X)
    return weval(W"Simplify"(W"Integrate"(w_object, (W"t", 0, W"Pi"))))
end

"""
    computes ∫_{nΩ<0} 2/abs(z(Ω)) * m1(Ω) * m2(Ω) dΩ
    This is Jonas L matrix
"""
function get_boundary_coefficient_symbolic_jonas(m1::CircularHarmonic, m2::CircularHarmonic, dim::EPMAfem.Dimensions.Z)
    return w_integrate_halfcircle_out((2/(W"Cos"(W"t")))*w_object(m1)*w_object(m2), dim)
end
"""
    computes ∫_{nΩ<0} 2/abs(x(Ω)) * m1(Ω) * m2(Ω) dΩ
    This is Jonas L matrix
"""
function get_boundary_coefficient_symbolic_jonas(m1::CircularHarmonic, m2::CircularHarmonic, dim::EPMAfem.Dimensions.X)
    return w_integrate_halfcircle_out((2/(W"Sin"(W"t")))*w_object(m1)*w_object(m2), dim)
end

function get_energy_coefficient_symbolic(m1::CircularHarmonic, m2::CircularHarmonic, dim::EPMAfem.Dimensions.X)
    # return w_integrate_circle(W"Sin"(W"t")*w_object(m1)*w_object(m2))
    return w_integrate_halfcircle(W"Abs"(W"Sin"(W"t"))*w_object(m1)*w_object(m2), dim)
end

"""
    computes ∫_{nΩ>0}m1(Ω)*m2(Ω)dΩ
"""
function get_outgoing_approx(m1::CircularHarmonic, m2::CircularHarmonic, dim::EPMAfem.Dimensions.X)
    return w_integrate_halfcircle_out(w_object(m1)*w_object(m2), dim)
end
function get_outgoing_approx(m1::CircularHarmonic, m2::CircularHarmonic, dim::EPMAfem.Dimensions.Z)
    return w_integrate_halfcircle_out(w_object(m1)*w_object(m2), dim)
end

include("boundary_dicts_generation.jl")

end
