# Derivation of DLR3

We consider the system:
$$\partial_t \left[\begin{pmatrix} M_t & 0 \\ 0 & N_t \end{pmatrix} \begin{pmatrix} x_t \\ y_t \end{pmatrix}\right] + \begin{pmatrix} E_t & B \\ B^T & G_t\end{pmatrix}\begin{pmatrix} x_t \\ y_t \end{pmatrix} + \begin{pmatrix} u_t \\ v_t \end{pmatrix} = 0$$

 - $N_t + G_t$ is diagonal. (therefore computing $y$ is essentially for free (retain structure) but do low rank approximation of $x$)
 - implicit midpoint method for time stepping

for each of the (K, L, S-) steps we have to solve the following "projected" problem (for $\hat x$)

$$\partial_t \left[\begin{pmatrix} \hat M_t & 0 \\ 0 & N_t \end{pmatrix} \begin{pmatrix} \hat x_t \\ y_t \end{pmatrix}\right] + \begin{pmatrix} \hat E_t & \hat B \\ \hat B^T & G_t\end{pmatrix}\begin{pmatrix} \hat x_t \\ y_t \end{pmatrix} + \begin{pmatrix} \hat u_t \\ v_t \end{pmatrix} = 0$$

after discretization

$$
  \frac{1}{\Delta t} \left[
      \begin{pmatrix} \hat{M}_{i+1} & 0 \\ 0 & N_{i+1} \end{pmatrix} \begin{pmatrix} \hat x_{i+1} \\ y_{i+1} \end{pmatrix}
    - \begin{pmatrix} \hat{M}_{i}   & 0 \\ 0 & N_{i}   \end{pmatrix} \begin{pmatrix} \hat x_{i}   \\ y_{i}   \end{pmatrix}
    \right]
  + \frac{1}{2} \left[
      \begin{pmatrix} \hat E_{i+1} & \hat B \\ \hat B^T & G_{i+1} \end{pmatrix} \begin{pmatrix} \hat x_{i+1} \\ y_{i+1} \end{pmatrix} 
    + \begin{pmatrix} \hat E_{i}   & \hat B \\ \hat B^T & G_{i}   \end{pmatrix} \begin{pmatrix} \hat x_{i}   \\ y_{i}   \end{pmatrix}
    + \begin{pmatrix} \hat u_{i+1} \\ v_{i+1} \end{pmatrix} 
    + \begin{pmatrix} \hat u_{i}   \\ v_{i}   \end{pmatrix}
    \right] 
  = 0
$$

reformulate

$$
    \begin{pmatrix} \frac{ \hat M_{i+1}}{\Delta t} + \frac{\hat E_{i+1}}{2} & \frac{\hat B}{2} \\ \frac{\hat B^T}{2} & \frac{ N_{i+1}}{\Delta t} + \frac{G_{i+1}}{2} \end{pmatrix} \begin{pmatrix} \hat x_{i+1} \\ y_{i+1} \end{pmatrix}
    = 
    -\left[
          \begin{pmatrix} \frac{-\hat M_{i}  }{\Delta t} + \frac{\hat E_{i  }}{2} & \frac{\hat B}{2} \\ \frac{\hat B^T}{2} & \frac{-N_{i}  }{\Delta t} + \frac{G_{i  }}{2} \end{pmatrix} \begin{pmatrix} \hat x_{i  } \\ y_{i  } \end{pmatrix}
        + \begin{pmatrix} \frac{\hat u_{i+1} + \hat u_{i}}{2} \\ \frac{v_{i+1} v_{i}}{2} \end{pmatrix}
    \right]
$$

rename
 - $\hat{A}_{i+1} = \frac{\hat M_{i+1}}{\Delta t} + \frac{\hat E_{i+1}}{2}$ (for $\hat A_{i} = \frac{-\hat M_{i}}{\Delta t} + \frac{\hat E_{i}}{2}$, note the minus!)
 - $C_{i+1} = \frac{ N_{i+1}}{\Delta t} + \frac{G_{i+1}}{2}$ (for $C_{i} = \frac{-N_{i}  }{\Delta t} + \frac{G_{i  }}{2} $, note the minus!)
 - $\hat B_i = \frac{\hat B_i}{2}$ (sorry, not wanna keep the $2$)
 - $\hat u = \frac{\hat u_{i+1} + \hat u_{i}}{2}$ 
 - $v = \frac{v_{i+1} + v_{i}}{2}$

$$
    \begin{pmatrix} \hat A_{i+1} & \hat B \\ \hat B^T & C_{i+1} \end{pmatrix} \begin{pmatrix} \hat x_{i+1} \\ y_{i+1} \end{pmatrix}
    = 
    -\left[
          \begin{pmatrix} \hat A_{i} & \hat B \\ \hat B^T & C_i \end{pmatrix} \begin{pmatrix} \hat x_{i  } \\ y_{i  } \end{pmatrix}
        + \begin{pmatrix} \hat u \\ v \end{pmatrix}
    \right]
    = -\begin{pmatrix}
        \hat{A}_i \hat{x}_i + \hat{B} y_i + \hat{u} \\
        \hat{B}^T \hat{x}_i + C_i y_i + v
    \end{pmatrix}
$$

first we compute $\hat{x}_{i+1}$ (using the schur complement)
$$
    (\hat{A}_{i+1} - \hat{B} (C_{i+1})^{-1} \hat{B}^T) \hat{x}_{i+1} = -\left[ \hat{A}_i \hat{x}_i + \hat{B} y_i + \hat{u}  - \hat{B}(C_{i+1})^{-1} (\hat{B}^T \hat{x}_i + C_i y_i + v) \right]
$$
rewrite 
$$
    (\hat{A}_{i+1} - \hat{B} (C_{i+1})^{-1} \hat{B}^T) \hat{x}_{i+1} = -\left[ \hat{u} + \hat{A}_i \hat{x}_i + \hat{B} (y_i - (C_{i+1})^{-1} (\hat{B}^T \hat{x}_i + C_i y_i + v)) \right]
$$
then (together with the S-step)
$$
    C_{i+1} y_{i+1} = -\left[\hat{B}^T \hat{x}_i + C_i y_i + v - \hat{B}^T \hat{x}_{i+1} \right]
$$
