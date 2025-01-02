@concrete struct ϵidx
    ϵs # steprangelen
    i # int
    adjoint #bool
end

index_string(i::ϵidx) = i.adjoint ? "$(i.i)-½" : "$(i.i)"
Base.show(io::IO, i::ϵidx) = print(io, "ϵidx("*"$(index_string(i))"*", ϵ=$(ϵ(i)))")
Base.show(io::IO, m::MIME"text/plain", i::ϵidx) = print(io, "ϵidx("*"$(index_string(i))"*", ϵ=$(ϵ(i)))")
Base.to_index(i::ϵidx) = if i.adjoint throw(ErrorException("cannot to_index with adjoint-index")) else return i.i end

function ϵ(idx::ϵidx)
    if idx.adjoint
        Δϵ2 = step(idx.ϵs)/2
        return idx.ϵs[idx.i] - Δϵ2
    else
        return idx.ϵs[idx.i]
    end
end

@inline function plus1(i::ϵidx)
    if i.i+1 > length(i.ϵs)
        return nothing
    else
        return ϵidx(i.ϵs, i.i+1, i.adjoint)
    end
end

@inline function plus½(i::ϵidx)
    if i.adjoint
        return ϵidx(i.ϵs, i.i, false)
    else
        if i.i+1 > length(i.ϵs)
            return nothing
        end
        return ϵidx(i.ϵs, i.i+1, true)
    end
end

@inline function minus½(i::ϵidx)
    if i.adjoint
        if i.i-1 < 1
            return nothing
        end
        return ϵidx(i.ϵs, i.i-1, false)
    else
        return ϵidx(i.ϵs, i.i, true)
    end
end

@inline function minus1(i::ϵidx)
    if i.i-1 < 1
        return nothing
    else
        return ϵidx(i.ϵs, i.i-1, i.adjoint)
    end
end

function first_index(ϵs, adjoint)
    if adjoint
        return ϵidx(ϵs, 1, adjoint)
    else
        return ϵidx(ϵs, length(ϵs), adjoint)
    end
end

function last_index(ϵs, adjoint)
    if adjoint
        return ϵidx(ϵs, length(ϵs), adjoint)
    else
        return ϵidx(ϵs, 1, adjoint)
    end
end

function is_first(i::ϵidx)
    if i.adjoint
        return i.i == 1
    else
        return i.i == length(i.ϵs)
    end
end

function next(i::ϵidx)
    if i.adjoint
        return plus1(i)
    else
        return minus1(i)
    end
end

first_index_nonadjoint(ϵs) = first_index(ϵs, false)
first_index_adjoint(ϵs) = first_index(ϵs, true)
last_index_nonadjoint(ϵs) = last_index(ϵs, false)
last_index_adjoint(ϵs) = last_index(ϵs, true)
