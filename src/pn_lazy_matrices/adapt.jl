Adapt.adapt_structure(to::Type{<:AbstractArray}, L::LazyMatrix) = lazy(Adapt.adapt_structure(to, L.A))
function Adapt.adapt_structure(to::Type{<:AbstractArray}, L::LazyOpMatrix)
    op = L.op
    args = Adapt.adapt_structure(to, L.args)
    kwargs = Adapt.adapt_structure(to, L.kwargs)
    return lazy(op, args...; kwargs...)
end

function Adapt.adapt_structure(::Type{<:AbstractArray{T_to}}, L::LazyScalar{T}) where {T_to, T}
    return LazyScalar{T_to}(convert(T_to, L.val))
end
