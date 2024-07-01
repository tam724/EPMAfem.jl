using GridapGmsh: gmsh

gmsh.initialize()
gmsh.clear()

p1 = gmsh.model.geo.addPoint(-1.5, -1.5, 0)
p2 = gmsh.model.geo.addPoint(-1.5, 1.5, 0)
p3 = gmsh.model.geo.addPoint(1.5, 1.5, 0)
p4 = gmsh.model.geo.addPoint(1.5, -1.5, 0)

l1 = gmsh.model.geo.addLine(p1, p2)
l2 = gmsh.model.geo.addLine(p2, p3)
l3 = gmsh.model.geo.addLine(p3, p4)
l4 = gmsh.model.geo.addLine(p4, p1)

loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
surf = gmsh.model.geo.addPlaneSurface([loop])

gmsh.model.geo.synchronize()

gmsh.option.setNumber("Mesh.MeshSizeMin", 0.01)
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.01)

gmsh.model.mesh.generate(2)

# f = joinpath(d, "square.msh")
gmsh.write("square.msh")
gmsh.clear()