using ChainRulesCore
using Zygote

function my_strange_addition(x, y)
    z = x + y
    return z
end

function rrule(::typeof(my_strange_addition), x, y)
    y = x + y
    function my_strange_addition_pullback(z_bar)
        x_bar = z_bar
        y_bar = z_bar
        return ZeroTangent(), x_bar, y_bar
    end
    return y, my_strange_addition_pullback
end

struct MyAdder
    x::Float64
end

function (A::MyAdder)(x, y)
    @show "peace"
    z = zeros(1)
    z[1] += A.x*x+A.x*y
    return z[1]
end

function rrule(A::MyAdder, x, y)
    z = A(x, y)
    function MyAdder_pullback(z_bar)
        x_bar = A.x*z_bar
        y_bar = A.x*z_bar
        return x_bar, y_bar
    end
    return z, MyAdder_pullback
end

function g(x)
    return A(2*x, x)
end

g(2)

A = MyAdder(2.0)

y, pb = Zygote.pullback(g, 2)

pb(1.0)
