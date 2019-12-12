"""
.. math:
    \\frac{\partial u}{\partial t} = \nabla^2u + f in \Omega \times (0,T], \\
    u = u_{\mathrm{D}} on \delta\Omega \times(0,T], \\
    u = u_0 at t=0

1. find u^0 \in V such that a_0(u^0, v) = L_0(v) holds for all v \in \hat{V}
2. find u^{n+1}\in V such that a(u^{n+1}, v)=L_{n+1}(v) for all v \in \hat{V}

Or...
F_{n_1}(u^{n+1},v)=0 \forall v in \hat{V} for n = 0, 1, 2,...

Our program needs to implement the time-stepping manually, but can rely on FEniCS to
easily compute a_0, L_0, a, and L (or F_{n+1}), and solve the linear systems for the unknowns


variable notation:
    -   u --> unknown u^{n+1} at the new time step
    -   u_n --> variable known at previous time step
"""
