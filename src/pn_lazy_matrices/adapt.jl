Adapt.adapt_structure(to::Type{<:AbstractArray}, L::LazyMatrix) = lazy(Adapt.adapt_structure(to, L.A))
Adapt.adapt_structure(to::Type{<:AbstractArray}, L::LazyOpMatrix) = lazy(L.op, Adapt.adapt_structure.(Ref(to), L.args)...)


function Adapt.adapt_structure(::Type{<:AbstractArray{T_to}}, L::LazyScalar{T}) where {T_to, T}
    return LazyScalar{T_to}(convert(T_to, L.val))
end

