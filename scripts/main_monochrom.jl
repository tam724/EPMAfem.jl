using Revise
using EPMAfem
using Gridap

eq = EPMAfem.MonochromPNEquations()
space_model = EPMAfem.SpaceModels.GridapSpaceModel(CartesianDiscreteModel((-1.0, 0.0), 50))
direction_model = EPMAfem.SphericalHarmonicsModels.EEEOSphericalHarmonicsModel(13, 1)

model = EPMAfem.DiscreteMonochromPNModel(space_model, direction_model)
problem = EPMAfem.discretize_problem(eq, model, EPMAfem.cpu())

