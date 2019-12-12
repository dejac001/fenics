"""
Test problem 1: a known analytical solution

Since we know that our firt-order time-stepping scheme is exact for linear functions,
we create a test problem which has a linear variation in time.
We combine this with a quadratic variation in space.
We thus take

.. math:
    u = 1 + x^2 + \alpha y^2 + \beta t

"""
import fenics as fe
import numpy as np
import matplotlib.pyplot as plt

T = 2.0                 # final time
num_steps = 10          # number of time steps
dt = T / num_steps      # time step size
alpha = 3               # parameter alpha
beta = 1.2              # parameter beta

# Create mesh and define function space
nx = ny = 8
mesh = fe.UnitSquareMesh(nx, ny)
V = fe.FunctionSpace(mesh, 'P', 1)

# Define boundary condition
#   note, boundary is time-dependent here!. Can update with time later by setting u_D.t = new_time
u_D = fe.Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t',
                    degree=2, alpha=alpha, beta=beta, t=0)


def boundary(x, on_boundary):
    return on_boundary


bc = fe.DirichletBC(V, u_D, boundary)

# Define initial value
#   Usually project, but, to actually recover the exact solution (3.15) to machine precision,
#   it is important to compute the discrete initial condition by interpolating u_0.
#   This ensures that the degrees of freedom are exact (to machine precision) at t=0.
#   Projection results in approximate values at the nodes
u_n = fe.interpolate(u_D, V)
# u_n = fe.project(u_D, V)

# Define variational problem
u = fe.TrialFunction(V)
v = fe.TestFunction(V)
f = fe.Constant(beta - 2 - 2*alpha)

# We may either define a or L according to the formulas above, or we may define F and ask FEniCS to figure
# out which terms should go into the bilinear form *a* and which should go into the linear form L.
#   The latter is convenient
#   ??? But what if the problem is nonlinear??
F = u*v*fe.dx + dt*fe.dot(fe.grad(u), fe.grad(v))*fe.dx - (u_n + dt*f)*v*fe.dx
a, L = fe.lhs(F), fe.rhs(F)

# Time-stepping
u = fe.Function(V)
t = 0
for n in range(num_steps):

    # Update current time
    t += dt
    u_D.t = t   # remember to update expression objects with the current time!

    # Compute solution
    fe.solve(a == L, u, bc)

    # Plot solution
    fe.plot(u)

    # Compute error at vertices
    u_e = fe.interpolate(u_D, V)
    vertex_values_u_e = u_e.compute_vertex_values(mesh)
    vertex_values_u = u.compute_vertex_values(mesh)
    error_max = np.max(np.abs(vertex_values_u_e - vertex_values_u))
    print('t = %.2f: error = %.3g' % (t, error_max))

    # Update previous solution
    #   note that we have to use *assign* here,
    #   otherwise if we do `u_n = u` we will set u_n to be the *same* variable as u.
    u_n.assign(u)

    # Hold plot
    # plt.show()

# final plot
plt.show()