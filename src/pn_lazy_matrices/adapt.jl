function Adapt.adapt_structure(to::Type{<:AbstractArray{T_to}}, L::LazyMatrix{T}) where {T_to, T}
    adapted_A = Adapt.adapt_structure(to, L.A)
    return LazyMatrix{T_to, typeof(adapted_A)}(adapted_A)
end

function Adapt.adapt_structure(to::Type{<:AbstractArray{T_to}}, L::LazyOpMatrix{T}) where {T_to, T}
    adapted_args = Adapt.adapt_structure.(Ref(to), L.args)
    return LazyOpMatrix{T_to}(L.op, adapted_args)
end

function Adapt.adapt_structure(::Type{<:AbstractArray{T_to}}, L::LazyScalar{T}) where {T_to, T}
    return LazyScalar{T_to}(convert(T_to, L.val))
end

