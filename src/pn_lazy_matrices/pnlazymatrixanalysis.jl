using Graphs, SparseArrays
using GLMakie
using Plots, GraphMakie
using GraphMakie.NetworkLayout
using EPMAfem
using EPMAfem.ConcreteStructs
using LinearAlgebra
Lazy = EPMAfem.PNLazyMatrices


function add_leaf!(g, A, A_to_v_dict, v_to_A_dict; flat)
   if flat || !haskey(A_to_v_dict, objectid(A)) # skip check if leaves should not be reused -> leads to a tree structure 
        @assert add_vertex!(g)
        v = nv(g)
        A_to_v_dict[objectid(A)] = v
        v_to_A_dict[v] = A
    end
    return A_to_v_dict[objectid(A)]
end

add_node!(g, A, A_to_v_dict, v_to_A_dict; flat) = add_leaf!(g, A, A_to_v_dict, v_to_A_dict; flat)
function add_node!(g, A::Lazy.LazyResizeMatrix, A_to_v_dict, v_to_A_dict; flat)
    add_leaf!(g, A, A_to_v_dict, v_to_A_dict; flat)
end

function add_node!(g, At::Transpose{<:Number, <:AbstractMatrix}, A_to_v_dict, v_to_A_dict; flat)
    if !haskey(A_to_v_dict, objectid(At))
        @assert add_vertex!(g)
        v = nv(g)
        A_to_v_dict[objectid(At)] = v
        v_to_A_dict[v] = At
    end
    v2 = add_node!(g, parent(At), A_to_v_dict, v_to_A_dict; flat=flat)
    add_edge!(g, A_to_v_dict[objectid(At)], v2)
    return A_to_v_dict[objectid(At)]
end

function add_node!(g, A::AbstractLazyMatrix, A_to_v_dict, v_to_A_dict; flat)
    if !haskey(A_to_v_dict, objectid(A))
        @assert add_vertex!(g)
        v = nv(g)
        A_to_v_dict[objectid(A)] = v
        v_to_A_dict[v] = A
    end
    for B in A.args
        v2 = add_node!(g, B, A_to_v_dict, v_to_A_dict; flat=flat)
        add_edge!(g, A_to_v_dict[objectid(A)], v2)
    end
    return A_to_v_dict[objectid(A)]
end

label(A::Lazy.LazyOpMatrix) = A.op |> string
label(A::Lazy.KronMatrix) = "⊗"
label(A::Diagonal) = "D"
label(A::AbstractSparseArray) = "S"
label(A::AbstractMatrix) = "A"
label(A::Lazy.LazyResizeMatrix) = "resize()"
label(A::Lazy.LazyScalar) = "α"
label(A::Transpose) = "transpose()"

@concrete struct LazyMatrixGraph
    graph
    v_to_A_dict
    A_to_v_dict
end


function build_graph(L::Lazy.AbstractLazyMatrixOrTranspose; flat=true)
    g = DiGraph()
    v_to_A_dict = Dict{Int, Any}()
    A_to_v_dict = Dict{UInt64, Int}()
    
    add_node!(g, L, A_to_v_dict, v_to_A_dict; flat=flat)
    return LazyMatrixGraph(g, v_to_A_dict, A_to_v_dict)
end

function node_hover_highlight_deps(p::GraphPlot, g::SimpleDiGraph)
    action = (state, idx, _, _) -> begin
        for node in BFSIterator(g, idx)
            p.node_color[][node] = state ? colorant"gray" : colorant"lightgray"
        end
        p.node_color[] = p.node_color[]
        end
    return GraphMakie.NodeHoverHandler(action)
end

function plot(LMG::LazyMatrixGraph; layout=Stress(dim=2))
    nodelabel = [label(LMG.v_to_A_dict[i]) for i in 1:nv(LMG.graph)]
    f, ax, p = GraphMakie.graphplot(LMG.graph, layout=layout, ilabels=nodelabel, node_color=fill(colorant"lightgray", nv(LMG.graph)))
    GraphMakie.deregister_interaction!(ax, :rectanglezoom)
    register_interaction!(ax, :nhover, node_hover_highlight_deps(p, LMG.graph))
    register_interaction!(ax, :nodedrag, NodeDrag(p))
    return f, ax, p
end

# # f, ax, p = GraphMakie.graphplot(g, layout=Buchheim(), node_size=fill(50.0, nv(g)), ilabels=nodelabel)

# p.ilabels_fontsize=30
# deregister_interaction!(ax, :rectanglezoom)
# register_interaction!(ax, :nhover, NodeHoverHighlight(p))
# register_interaction!(ax, :ndrag, NodeDrag(p))
