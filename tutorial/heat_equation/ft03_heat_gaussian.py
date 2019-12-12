"""
Solve the diffusion of a Gaussian hill. We take the initial value to be

.. math:
    u_0(x,y) = e^{-ax^2-ay^2}

with a=5 on [-2,-2]x[2,2]

For this problem we will use homogeneous Dirichlet boundary conditions (u_D = 0)
"""

import fenics as fe
import matplotlib.pyplot as plt

T = 2.0             # final time
num_steps = 50      # number of time steps
dt = T / num_steps  # time step size

# Create mesh and define function space
# we no longer have a unit square **
nx = ny = 30
mesh = fe.RectangleMesh(
    fe.Point(-2, -2), fe.Point(2, 2), nx, ny
)
V = fe.FunctionSpace(mesh, 'P', 1)


# Define boundary condition
def boundary(x, on_boundary):
    return on_boundary


bc = fe.DirichletBC(V, fe.Constant(0), boundary)

# Define initial value
u_0 = fe.Expression(
    'exp(-a*pow(x[0], 2) - a*pow(x[1], 2))',
    degree=2, a=5
)
u_n = fe.interpolate(u_0, V)

# Define variational problem
u = fe.TrialFunction(V)
v = fe.TestFunction(V)
f = fe.Constant(0)

F = u*v*fe.dx + dt*fe.dot(fe.grad(u), fe.grad(v))*fe.dx - (u_n+dt*f)*v*fe.dx
a, L = fe.lhs(F), fe.rhs(F)

# Create VTK file for saving solution
vtkfile = fe.File('heat_gaussian/solution.pvd')

# Time-stepping
u = fe.Function(V)
t = 0
for n in range(num_steps):

    # Update current time
    t += dt

    # Compute solution
    fe.solve(
        a == L, u, bc
    )

    # Save to file and plot solution
    vtkfile << (u, t)
    fe.plot(u)

    # Update previous solution
    u_n.assign(u)

# HOld plot
plt.show()