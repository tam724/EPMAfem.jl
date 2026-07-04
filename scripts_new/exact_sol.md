$$\frac{d u}{d t} + A_x \frac{d u}{dx} + A_y \frac{d u}{dy} = 0$$
initial condition
$$u_0(x, t=0) = \exp(-\sigma(x^2 + y^2))$$


apply the fourier transformation in space

$$\mathcal{F}(u) = \hat{u}$$
$$\mathcal{F}(\frac{d u}{dx}) = i k_x \hat{u}$$


this leads to 
$$\frac{d\hat{u}}{dt} + i(A k_x + B k_y)\hat{u} = 0$$

fourier transformation of the initial condition

$$ \hat{u}_0(t=0) = \frac{\pi}{\sigma} \exp(-\frac{k_x^2 + k_y^2}{4\sigma})$$

therefore we have the solution

$$\hat{u}(t) = \exp(-i(Ak_x + Bk_y)t) \hat{u}_0 $$



$$
\begin{alignat*}{2}
    \partial_t u &+ \nabla \cdot{} v &&= 0 \\
    \partial_t v &+ \nabla u &&= 0
\end{alignat*}
$$
