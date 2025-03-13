# EPMAfem.jl

Implements a mixed variational finite element discretization for the spherical harmonic approximation of linear transport in continuous-slowing-down approximation for application in the *inverse problem of material reconstruction in EPMA*.
Electron probe microanalysis (EPMA) is an imaging technique for solid material samples based on characteristic x-ray emission.

To facilitate efficient algorithms based on the discretization, the resulting linear system as well as the adjoint linear system are implemented.
The solver can be coupled to AD tools (Zygote.jl) to implement gradient-based approaches for optimization of cost functionals.

This is **research code**, hence mostly undocumented. :(

Questions? Feel free to contact: 
 - [Tamme Claus](https://www.acom.rwth-aachen.de/the-lab/team-people/name:tamme_claus) (claus@acom.rwth-aachen.de)

# Reproducibility

- "Twofold Adjoint Method for Inverse Problems with Excitation-Dominant Models" (submitted)
```julia
    # clone the repository and start julia
    > git clone git@github.com:tam724/EPMAfem.jl.git
    > cd EPMAfem.jl
    > julia

    # in pkg mode (julia> ]) activate and instantiate the project
    pkg> activate scripts_twofold_adjoint_method/
    pkg> instantiate

    # run the script
    julia> include("scripts_twofold_adjoint_method/main.jl")
```
(Instantiation will load the version `EPMAfem.jl#v.0.1.0`)