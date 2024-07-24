


mutable struct ExplicitEulerIterator{F}
    T::Tuple{Float64, Float64}
    N::Int64
    Δt::Float64
    f::F
end

function Base.iterate(it::ExplicitEulerIterator)
    state = (u=1.0, i=1, t=it.T[1])
    return (state.u, state.t), state
end

function Base.iterate(it::ExplicitEulerIterator, state)
    if state.i >= it.N
        return nothing
    end
    new_state = (
        u = state.u + it.Δt * it.f(state.t, state.u),
        i = state.i + 1,
        t = state.t + it.Δt
    )
    return (new_state.u, new_state.t), new_state
end

function F(t, u)
    return -1.0*u
end

expl_eul = ExplicitEulerIterator((0.0, 1.0), 100, 1.0 / (100-1), F)


us = zeros(100)
ts = zeros(100)


using BenchmarkTools

function compute!(us, ts, expl_eul)
    for (i, (u, t)) in enumerate(expl_eul)
        us[i] = u
        ts[i] = t
    end
end

@code_warntype compute!(us, ts, expl_eul)


@time begin
    for (i, (u, t)) in enumerate(expl_eul)
        us[i] = u
        ts[i] = t
    end 
end

´using Plots
plot(ts, us)