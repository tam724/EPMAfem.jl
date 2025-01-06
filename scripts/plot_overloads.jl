# Plots overloads (for fast plotting of FE functions)
import Plots: plot, plot!, heatmap, surface
function plot(x, u::CellField; kw...)
    points = Point.(x)
    eval = u(points)
    return plot(x, eval; kw...)
end
function plot!(x, u::CellField; kw...)
    points = Point.(x)
    eval = u(points)
    return plot!(x, eval; kw...)
end
function heatmap(x, y, u::CellField; kw...)
    points = Point.(x', y)
    eval = reshape(u(reshape(points, :)), (length(y), length(x)))
    return heatmap(x, y, eval; kw...)
end
function surface(x, y, u::CellField; kw...)
    points = Point.(x', y)
    eval = reshape(u(reshape(points, :)), (length(y), length(x)))
    return surface(x, y, eval; kw...)
end
