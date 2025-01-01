using EPMAfem.ConcreteStructs
using LinearAlgebra

using EPMAfem

@concrete struct iterator
    state
end

function Base.iterate(it)
    idx = EPMAfem.first_index(0:0.1:1, false)
    it.state[1] = idx.ϵs[idx]
    return idx, idx
end

function Base.iterate(it, idx)
    idx = EPMAfem.minus1(idx)
    if isnothing(idx) return nothing end
    it.state[1] = idx.ϵs[idx]
    return idx, idx
end

itr = iterator([NaN])

for i in itr
    @show i
end