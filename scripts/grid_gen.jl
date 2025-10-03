using GridapGmsh: gmsh

# TODO: check if the tags here correspond to the default Gridap CartesianDiscreteModel tags!
function grid_gen_1D((zl, zr); res=0.01, highres=(dist=0.0, res=0.0, points=[]), filepath="/tmp/tmp_msh.msh")
    gmsh.initialize()
    gmsh.clear()
                                # x, y, z, res, tag
    p1 = gmsh.model.geo.addPoint(zl, 0.0, 0.0, 0.0, 1)
    p2 = gmsh.model.geo.addPoint(zr, 0.0, 0.0, 0.0, 2)

    gmsh.model.geo.addPhysicalGroup(0, [1], 1, "tag_1")
    gmsh.model.geo.addPhysicalGroup(0, [2], 2, "tag_2")

    l1 = gmsh.model.geo.addLine(p1, p2, 1)
    gmsh.model.geo.addPhysicalGroup(1, [3], 3, "tag_3")

    gmsh.model.geo.synchronize()

    gmsh.model.geo.addPhysicalGroup(1, [l1], 4, "interior")
    gmsh.model.geo.addPhysicalGroup(0, [p1, p2], 5, "boundary")

    rps = Int64[]
    for res_point in highres.points
        rp = gmsh.model.geo.addPoint(res_point, 0.0, 0.0, 0.0)
        push!(rps, rp)
    end
    field_id = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(field_id, "NodesList", rps)

    threshold_id = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold_id, "InField", field_id)
    gmsh.model.mesh.field.setNumber(threshold_id, "SizeMin", highres.res)
    gmsh.model.mesh.field.setNumber(threshold_id, "SizeMax", res)
    gmsh.model.mesh.field.setNumber(threshold_id, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(threshold_id, "DistMax", highres.dist)

    gmsh.model.mesh.field.setAsBackgroundMesh(threshold_id)

    gmsh.model.geo.synchronize()

    gmsh.model.mesh.generate(1)

    gmsh.write(filepath)
    gmsh.clear()
end

function grid_gen_2D((zl, zr, xl, xr); min_res=0.1, max_res=0.01, filepath="/tmp/tmp_msh.msh")
    gmsh.initialize()
    gmsh.clear()
                                # x, y, z, res, tag
    p1 = gmsh.model.geo.addPoint(zl, xl, 0.0, 0.0, 1)
    p2 = gmsh.model.geo.addPoint(zl, xr, 0.0, 0.0, 3)
    p3 = gmsh.model.geo.addPoint(zr, xr, 0.0, 0.0, 4)
    p4 = gmsh.model.geo.addPoint(xr, xl, 0.0, 0.0, 2)

    gmsh.model.geo.addPhysicalGroup(0, [1], 1, "tag_1")
    gmsh.model.geo.addPhysicalGroup(0, [2], 2, "tag_2")
    gmsh.model.geo.addPhysicalGroup(0, [3], 3, "tag_3")
    gmsh.model.geo.addPhysicalGroup(0, [4], 4, "tag_4")

    l1 = gmsh.model.geo.addLine(p1, p2, 7)
    l2 = gmsh.model.geo.addLine(p2, p3, 6)
    l3 = gmsh.model.geo.addLine(p3, p4, 8)
    l4 = gmsh.model.geo.addLine(p4, p1, 5)

    gmsh.model.geo.addPhysicalGroup(1, [5], 5, "tag_5")
    gmsh.model.geo.addPhysicalGroup(1, [6], 6, "tag_6")
    gmsh.model.geo.addPhysicalGroup(1, [7], 7, "tag_7")
    gmsh.model.geo.addPhysicalGroup(1, [8], 8, "tag_8")

    gmsh.model.geo.synchronize()

    loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    surf = gmsh.model.geo.addPlaneSurface([loop])
    gmsh.model.geo.addPhysicalGroup(2, [surf], 9, "interior")
    gmsh.model.geo.addPhysicalGroup(1, [loop], 10, "boundary")

    # middle = gmsh.model.geo.addPoint(0.5, 0.5, 0.0, 0.01, 5)

    # Apply the custom mesh size function
    function_field = gmsh.model.mesh.field.add("MathEval")
    @show "$max_res - abs(x - $zr)/abs($(zr-zl))*($max_res - $min_res)"
    gmsh.model.mesh.field.setString(function_field, "F", "$max_res - abs(x - $zr)/abs($(zr-zl))*($max_res - $min_res)")
    
    # Apply the field as the background mesh size
    gmsh.model.mesh.field.setAsBackgroundMesh(function_field)

    gmsh.model.geo.synchronize()

    # gmsh.model.mesh.embed(0, [middle, p1, p2], 2, surf)

    #gmsh.option.setNumber("Mesh.MeshSizeMin", res)
    #gmsh.option.setNumber("Mesh.MeshSizeMax", res)

    gmsh.model.mesh.generate(2)

    # f = joinpath(d, "square.msh")
    gmsh.write(filepath)
    gmsh.clear()
end

# grid_gen_2D((-0.5, 0.5, -0.5, 0.5); filepath="/tmp/tmp_msh.msh")

# using Gridap

# model = DiscreteModelFromFile("/tmp/tmp_msh.msh")
# get_face_labeling(model)
