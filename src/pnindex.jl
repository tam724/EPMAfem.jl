@concrete struct ϵidx
    ϵs # steprangelen
    i # int
    adjoint #bool
end

index_string(i::ϵidx) = i.adjoint ? "$(i.i)-½" : "$(i.i)"
Base.show(io::IO, i::ϵidx) = print(io, "ϵidx("*"$(index_string(i))"*", ϵ=$(ϵ(i)))")
Base.show(io::IO, m::MIME"text/plain", i::ϵidx) = print(io, "ϵidx("*"$(index_string(i))"*", ϵ=$(ϵ(i)))")
Base.to_index(i::ϵidx) = if i.adjoint throw(ErrorException("cannot to_index with adjoint-index")) else return i.i end

ϵ(idx::ϵidx) = idx.adjoint ? idx.ϵs[idx.i] - step(idx.ϵs)/2 : idx.ϵs[idx.i]

@inline plus1(idx::ϵidx) = (idx.i+1 > length(idx.ϵs)) ? nothing : ϵidx(idx.ϵs, idx.i+1, idx.adjoint)
@inline minus1(idx::ϵidx) = (idx.i-1 < 1) ? nothing : ϵidx(idx.ϵs, idx.i-1, idx.adjoint)

@inline function plus½(idx::ϵidx)
    if idx.adjoint
        return ϵidx(idx.ϵs, idx.i, false)
    else
        if idx.i+1 > length(idx.ϵs)
            return nothing
        end
        return ϵidx(idx.ϵs, idx.i+1, true)
    end
end

@inline function minus½(idx::ϵidx)
    if idx.adjoint
        if idx.i-1 < 1
            return nothing
        end
        return ϵidx(idx.ϵs, idx.i-1, false)
    else
        return ϵidx(idx.ϵs, idx.i, true)
    end
end

first_index(ϵs, adjoint) = adjoint ? ϵidx(ϵs, 1, adjoint) : ϵidx(ϵs, length(ϵs), adjoint)
last_index(ϵs, adjoint) = adjoint ? ϵidx(ϵs, length(ϵs), adjoint) : ϵidx(ϵs, 1, adjoint)
is_first(idx::ϵidx) = idx.adjoint ? idx.i == 1 : idx.i == length(idx.ϵs)
next(idx::ϵidx) = idx.adjoint ? plus1(idx) : minus1(idx)
previous(idx::ϵidx) = idx.adjoint ? minus1(idx) : plus1(idx)

function Base.isless(i::ϵidx, j::ϵidx)
    if i.adjoint == j.adjoint
        return i.i < j.i
    elseif i.adjoint
        return i.i <= j.i
    elseif j.adjoint
        return i.i < j.i
    end
end

function Base.isequal(i::ϵidx, j::ϵidx)
    return i.adjoint == j.adjoint && i.i == j.i && i.ϵs === j.ϵs
end

first_index_nonadjoint(ϵs) = first_index(ϵs, false)
first_index_adjoint(ϵs) = first_index(ϵs, true)
last_index_nonadjoint(ϵs) = last_index(ϵs, false)
last_index_adjoint(ϵs) = last_index(ϵs, true)
