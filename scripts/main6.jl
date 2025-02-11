using Pkg
Pkg.activate(temp=true)
Pkg.add(["Cassette", "ConcreteStructs"])

using Cassette
using ConcreteStructs

struct Func{F}
    f::F
end

mutable struct Active{T}
    value::T
end

## EMPTY PULLBACKS
function pullback!(::Nothing, ::Func{<:Function}, Δouts, args)
    return nothing
end

function pullback!(::NTuple{N, <:Nothing},  ::Func{<:Function}, Δouts, args) where N
    return nothing
end

## ADDITION
function pullback!(Δx::Tuple{<:Ref{<:T}, <:Ref{<:T}}, ::Func{typeof(+)}, Δy, x::Tuple{Active{T}, Active{T}}) where T <: Real
    Δx[1][] += Δy[]
    Δx[2][] += Δy[]
    return nothing
end

function pullback!(Δx::Tuple{<:Ref{<:T}, <:Nothing}, ::Func{typeof(+)}, Δy, x::Tuple{Active{T}, <:Any}) where T <: Real
    Δx[1][] += Δy[]
    return nothing
end

function pullback!(Δx::Tuple{<: Nothing, <:Ref{<:T}}, ::Func{typeof(+)}, Δy, x::Tuple{<:Any, Active{T}}) where T <: Real
    Δx[2][] += Δy[]
    return nothing
end

## MULTIPLICATION
function pullback!(Δx::Tuple{<:Ref{<:T}, <:Ref{<:T}}, ::Func{typeof(*)}, Δy, x::Tuple{Active{T}, Active{T}}) where T <: Real
    Δx[1][] += x[2].value * Δy[]
    Δx[2][] += x[1].value * Δy[]
    return nothing
end

function pullback!(Δx::Tuple{<:Ref{<:T}, <:Nothing}, ::Func{typeof(*)}, Δy, x::Tuple{Active{T}, T}) where T <: Real
    Δx[1][] += x[2] * Δy[]
    return nothing
end

function pullback!(Δx::Tuple{<:Nothing, <:Ref{<:T}}, ::Func{typeof(*)}, Δy, x::Tuple{T, Active{T}}) where T <: Real
    Δx[2][] += x[1] * Δy[]
    return nothing
end

## SIN
function pullback!(Δx::Ref{<:T}, ::Func{typeof(sin)}, Δy, x::Tuple{Active{T}}) where T <: Real
    Δx[] += cos(x.value) * Δy[]
    return nothing
end



# function g(x)
#     v = Func(+)((x, 1.0))
#     y = Func(+)((v, x))
#     y = 1.0
#     return y
# end

function dump_str(f) 
    io = IOBuffer()
    dump(io, f)
    return String(take!(io))
end

(f::Func{typeof(+)})((x, y)::Tuple{<:Active, <:Active}) = Active(f.f(x.value, y.value))
(f::Func{typeof(+)})((x, y)::Tuple{<:Active, <:Any}) = Active(f.f(x.value, y))
(f::Func{typeof(+)})((x, y)::Tuple{<:Any, <:Active}) = Active(f.f(x, y.value))
(f::Func{typeof(+)})((x, y)::Tuple{<:Any, <:Any}) = f.f(x, y)

(f::Func{typeof(*)})((x, y)::Tuple{<:Active, <:Active}) = Active(f.f(x.value, y.value))
(f::Func{typeof(*)})((x, y)::Tuple{<:Active, <:Any}) = Active(f.f(x.value, y))
(f::Func{typeof(*)})((x, y)::Tuple{<:Any, <:Active}) = Active(f.f(x, y.value))
(f::Func{typeof(*)})((x, y)::Tuple{<:Any, <:Any}) = f.f(x, y)

# (f::Func)(x::Active) = Active(f.f(x.value))
# (f::Func)(x) = f.f(x)

# x_v = Active(1.0)

Cassette.@context Ctx;
function Cassette.prehook(::Ctx, f, arg)
    if isa(arg, Active)
        @show "calling $f with active $arg"
    else
        @show "calling $f"
    end
end
function Cassette.prehook(::Ctx, f, args...)
    for arg in args
        if isa(arg, Active)
            @show "calling $f with active $arg"
        end
    end
end

mutable struct Pullback
    pullback::Any
    cache::Any
end

function Cassette.posthook(ctx::Ctx, y, f::Func, x)
    previous = ctx.metadata.pullback
    cache = ctx.metadata.cache
    Δx = ()
    for x_ in x
        if x_ isa Active
            if !haskey(cache, x_)
                cache[x_] = Ref(zero(eltype(x_.value)))
            end
            Δx = (Δx..., cache[x_])
        else
            Δx = (Δx..., nothing)
        end
    end
    if y isa Active
        if !haskey(cache, y)
            cache[y] = Ref(zero(eltype(y.value)))
        end
        ctx.metadata.pullback = () -> (pullback!(Δx, f, cache[y], x); previous())
    end
end

# Cassette.overdub(ctx::Ctx, f::Func, x...) = f(x...)


gg(x) = x^4 + 2*x^3 + 2*x^2 + 1
ddgg(x) = 4*x^3 + 6*x^2 + 4*x + 1
function g(x)
    v = Func(+)((x, 1.0))
    y = Func(*)((v, x))
    z = Func(+)((y, 1.0))
    w = Func(*)((z, y))
    w = 1.0
    return w
end

g(1.0)

_x = Active(1.0)
_f() = g(_x)
ctx = Ctx(metadata=Pullback(() -> nothing, Dict()))
y = Cassette.overdub(ctx, _f)
ctx.metadata.cache[y] = Ref(1.0)
pb = ctx.metadata.pullback
@code_lowered pb()
ctx.metadata.cache
ctx.metadata.cache[_x]

ddgg(1.0)

@generated function test(f, x)
    Core.println("i am generated")
    original_code = @code_lowered f(x)
    return original_code
end


f(x) = x^2

f(2)
test(f, 2)

function pullback!(Δx, f::Function, Δy, x)
    f_code = @code_lowered f(x)
end

function gradient(f::Function, x...)
    y = f(x...)
    Δy = one(typeof(y))
    Δx = Ref.(zero.(typeof.(x...)))
    pullback!(Δx, f, Δy, x)
end

gradient(f, 1.0)