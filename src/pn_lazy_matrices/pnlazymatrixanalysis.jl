using Graphs, SparseArrays
using Plots, GraphMakie
using GraphMakie.NetworkLayout


function add_node!(g, A, A_to_v_dict, v_to_A_dict)
    if !haskey(A_to_v_dict, objectid(A)) # skip check if leaves should not be reused -> leads to a tree structure 
        @assert add_vertex!(g)
        v = nv(g)
        A_to_v_dict[objectid(A)] = v
        v_to_A_dict[v] = A
    end
    return A_to_v_dict[objectid(A)]
end

function add_node!(g, A_or_At::Union{EPMAfem.AbstractLazyMatrix, Transpose{<:Number, <:EPMAfem.AbstractLazyMatrix}}, A_to_v_dict, v_to_A_dict)
    if A_or_At isa Transpose A = parent(A_or_At) else A = A_or_At end
    if !haskey(A_to_v_dict, objectid(A))
        @assert add_vertex!(g)
        v = nv(g)
        A_to_v_dict[objectid(A)] = v
        v_to_A_dict[v] = A
    end
    for B in A.args
        v2 = add_node!(g, B, A_to_v_dict, v_to_A_dict)
        add_edge!(g, A_to_v_dict[objectid(A)], v2)
    end
    return A_to_v_dict[objectid(A)]
end

g = DiGraph()

v_to_A_dict = Dict{Int, Any}()
A_to_v_dict = Dict{UInt64, Int}()

add_node!(g, B, A_to_v_dict, v_to_A_dict)

g

label(A::EPMAfem.LazyOpMatrix) = A.op |> string
label(A::EPMAfem.KronMatrix) = "⊗"
label(A::EPMAfem.BlockMatrix) = "[A B\n Bt C]"
label(A::Diagonal) = "D"
label(A::AbstractSparseArray) = "S"
label(A::AbstractMatrix) = "A"
label(A::Ref) = "α"

nodelabel = [label(v_to_A_dict[i]) for i in 1:nv(g)]

f, ax, p = GraphMakie.graphplot(g, layout=Stress(dim=3), ilabels=nodelabel)
# f, ax, p = GraphMakie.graphplot(g, layout=Buchheim(), node_size=fill(50.0, nv(g)), ilabels=nodelabel)

p.ilabels_fontsize=30
deregister_interaction!(ax, :rectanglezoom)
register_interaction!(ax, :nhover, NodeHoverHighlight(p))
register_interaction!(ax, :ndrag, NodeDrag(p))
