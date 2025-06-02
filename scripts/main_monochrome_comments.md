# Some documentation about monochrome_pnequations:

## Terminology

$$\Omega \cdot{} \nabla \psi + \sum_{e=1}^{n_e} \rho_e(x) (\tau_e + \sigma_e) \psi = \sum_{e=1}^{n_e} \rho_e(x) \sigma_e \int_{S^2} k_e(\Omega \cdot{} \Omega') \psi' d\Omega'$$

- independent variables $x \in \mathcal{R} \subset \mathbb{R}^{1/2/3}$ and $\Omega \in S^2$ (in 1D slab geometry [maybe a coefficient should be fixed here], in 2D circle, in 3D full sphere)
- solution variable $\psi(x, \Omega)$ ($\psi' = \psi(x, \Omega')$)
- mass concentrations $\rho_e(x)$ (only have space dependency)
- absorption/attenuation coefficient $\tau_e$ (sometimes together with $\sigma_e$ called a microscopic (total) cross section) (not balanced)
- scattering coefficient $\sigma_e$ 
- scattering kernel $k_e(\mu)$ ($\cos^{-1}(\mu) = \cos^{-1}(\Omega \cdot{} \Omega')$ is the angle between scattering directions, the scattering kernel $k$ can be thought of a probability of scattering from $\Omega'$ to $\Omega$ and should be normalized $\int_{S^2} k(\Omega \cdot{} \Omega') d \Omega'= 1$, then the $\sigma_e$ on the left and right is balanced)

## weak form/discretization
- the equation is transformed into weak form according to [Egger/Schlottbom "Mixed Variational ..."]
- discretization using spherical harmonics(selfcoded in "spherical harmonics model$) and finite elements (Gridap.jl interfaced in "space_model") leads to the "MonochromPNProblem" (which is basically a collection of matrices, the matrix names are similar to what I wrote in [Claus/Torillhon "Twofold Adjoint.." (C Dual Consistency..)], but e.g. $N_e^\pm$ is $\texttt{ρp/ρm}$, $\nabla_d$ is $\nabla\texttt{pm}$)


# source term/boundary condition
- discretization of a source term is straightforward
- boundary conditions are (Dirichlet for incoming directions)
$$\psi(x, \Omega) = f(x, \Omega) \quad \forall x \in \partial \mathcal{R}, \Omega \cdot{} n < 0$$
(where $n$ is the outwards boundary normal)
- in the code this is split into $f(x, \Omega) = f_x(x) f_\Omega(\Omega)$ to treat space and direction independently (more general boundary function can be represented e.g. by a sum of these)
- boundary conditions are enforced in a weak sense (they are part of the rhs of the system of equations, not build into the space (like e.g. in standard FEM for Poisson equation))
- for the derivation of the boundary contributions, the important equation in [Egger/Schlottbom "Mixed.."] is eq. (3.4). The result is stated in (3.6)/(3.7) (we only "test" with even spherical harmonics)

# Interpreting the solution vector
- EPMAfem.pmview (reshapes the solution vector into even/odd part -> two matrices)
- EPMAfem.SphericalHarmonicsModels.eval_basis(direction_mdl, [either a direction, of a function \Omega -> R]) evaluates/integrates the spherical harmonics basis functions (to partly evaluate the solution)

$$\psi(x, \Omega) \approx \sum_{i} \sum_{j} \Psi^+ \mathcal{X}^+_i(x) \Upsilon^+_j(\Omega) + \sum_{i} \sum_{j} \Psi^- \mathcal{X}^-_i(x) \Upsilon^-_j(\Omega)$$
then e.g. the angle average is
$$\int_{S^2} \psi(x, \Omega) d \Omega \approx \sum_{i} \Psi^+ \mathcal{X}^+_i(x)  \underbrace{\int_{S^2} \Upsilon^+_j(\Omega) d\Omega}_{=(1, 0, 0... )^T} + \sum_{i} \sum_{j} \Psi^- \mathcal{X}^-_i(x) \underbrace{\int_{S^2}\Upsilon^-_j(\Omega) d \Omega}_{=(0, 0, ...)^T}$$


the result can be evaluated at a point $x$ (uses Gridap.jl machinery to find the corresponding gridcell end evaluate the basis functions) (this is implemented by EPMAfem.SpaceModels.interpolable(...))

