using Ferrite
using SparseArrays
using LinearAlgebra
using Plots


grid = generate_grid(Quadrilateral, (3, 3), Vec(0.0, -1.0), Vec(1.0, 1.0))

FacetIndex(1, 1)

ip_p = Lagrange{RefQuadrilateral, 1}()
ip_m = DiscontinuousLagrange{RefQuadrilateral, 0}()
qrule = QuadratureRule{RefQuadrilateral}(3)
facet_qrule = FacetQuadratureRule{RefQuadrilateral}(10)

cellvalues_p = CellValues(qrule, ip_p)
cellvalues_m = CellValues(qrule, ip_m)

facetvalues = FacetValues(facet_qrule, ip_p)

facet = first(FacetIterator(grid, ∂R))
facet

∂R = union(getfacetset(grid, "top"), getfacetset(grid, "bottom"), getfacetset(grid, "right"))
get_facet_cell(∂R[1])

for cellcache in CellIterator(grid)

dhp = close!(add!(DofHandler(grid), :up, ip_p))
dhm = close!(add!(DofHandler(grid), :um, ip_m))

scatter(getindex.(cellvalues_p.qr.points, 1), getindex.(cellvalues_p.qr.points, 2))


grid.cells[1].nodes
grid.nodes[1].x[1]

celldofs(dhp, 3)
celldofs(dhm, 3)

RefCube


function plot_grid(grid)
    plot()
    for cell in grid.cells
        x = [grid.nodes[n].x[1] for n in cell.nodes]
        y = [grid.nodes[n].x[2] for n in cell.nodes]
        plot!(x, y)
    end
    plot!()
end

plot_grid(grid)

reinit!(cellvalues_p, cellcaches[3].coords) 

cellvalues_p.fun_values |> typeof |> fieldnames
cellvalues_p.fun_values.dNdx

Ferrite.FunctionValues
x = [spatial_coordinate(cellvalues_p, i, cellcaches[3].coords)[1] for i in 1:getnquadpoints(cellvalues_p)]
y = [spatial_coordinate(cellvalues_p, i, cellcaches[3].coords)[2] for i in 1:getnquadpoints(cellvalues_p)]

scatter!(x, y)

@which spatial_coordinate(cellvalues_p, 1, cellcache.coords)

# TODO replace the gridap semidiscretization.