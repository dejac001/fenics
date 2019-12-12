"""
FEniCS tutorial demo program: Incompressible Navier-Stokes equations
for flow around a cylinder using the Incremental Pressure Correction
Scheme (IPCS).

  u' + u . nabla(u)) - div(sigma(u, p)) = f
                                 div(u) = 0
"""

import fenics as fe
import mshr as m
import numpy as np
import matplotlib.pyplot as plt

T = 5.0                 # final time
num_steps = 5000        # number of time steps
dt = T / num_steps      # time step size
mu = 0.001              # dynamic viscosity
rho = 1                 # density

# Create mesh
channel = m.Rectangle(fe.Point(0, 0), fe.Point(2.2, 0.41))
cylinder = m.Circle(fe.Point(0.2, 0.2), 0.05)
domain = channel - cylinder
mesh = m.generate_mesh(domain, 64)

# Define function spaces
V = fe.VectorFunctionSpace(mesh, 'P', 2)
Q = fe.FunctionSpace(mesh, 'P', 1)

# Define boundaries
inflow   = 'near(x[0], 0)'
outflow  = 'near(x[0], 2.2)'
walls    = 'near(x[1], 0) || near(x[1], 0.41)'
cylinder = 'on_boundary && x[0]>0.1 && x[0]<0.3 && x[1]>0.1 && x[1]<0.3'

# Define inflow profile
inflow_profile = ('4.0*1.5*x[1]*(0.41 - x[1]) / pow(0.41, 2)', '0')

# Define boundary conditions
bcu_inflow = fe.DirichletBC(V, fe.Expression(inflow_profile, degree=2), inflow)
bcu_walls = fe.DirichletBC(V, fe.Constant((0, 0)), walls)
bcu_cylinder = fe.DirichletBC(V, fe.Constant((0, 0)), cylinder)
bcp_outflow = fe.DirichletBC(Q, fe.Constant(0), outflow)
bcu = [bcu_inflow, bcu_walls, bcu_cylinder]
bcp = [bcp_outflow]

# Define trial and test functions
u = fe.TrialFunction(V)
v = fe.TestFunction(V)
p = fe.TrialFunction(Q)
q = fe.TestFunction(Q)

# Define functions for solutions at previous and current time steps
u_n = fe.Function(V)
u_  = fe.Function(V)
p_n = fe.Function(Q)
p_  = fe.Function(Q)

# Define expressions used in variational forms
U  = 0.5*(u_n + u)
n  = fe.FacetNormal(mesh)
f  = fe.Constant((0, 0))
k  = fe.Constant(dt)
mu = fe.Constant(mu)
rho = fe.Constant(rho)

# Define symmetric gradient
def epsilon(u):
    return fe.sym(fe.nabla_grad(u))

# Define stress tensor
def sigma(u, p):
    return 2 * mu * epsilon(u) - p * fe.Identity(len(u))

# Define variational problem for step 1
F1 = rho * fe.dot((u - u_n) / k, v) * fe.dx \
     + rho * fe.dot(fe.dot(u_n, fe.nabla_grad(u_n)), v) * fe.dx \
     + fe.inner(sigma(U, p_n), epsilon(v)) * fe.dx \
     + fe.dot(p_n * n, v) * fe.ds - fe.dot(mu * fe.nabla_grad(U) * n, v) * fe.ds \
     - fe.dot(f, v) * fe.dx
a1 = fe.lhs(F1)
L1 = fe.rhs(F1)

# Define variational problem for step 2
a2 = fe.dot(fe.nabla_grad(p), fe.nabla_grad(q)) * fe.dx
L2 = fe.dot(fe.nabla_grad(p_n), fe.nabla_grad(q)) * fe.dx - (1 / k) * fe.div(u_) * q * fe.dx

# Define variational problem for step 3
a3 = fe.dot(u, v) * fe.dx
L3 = fe.dot(u_, v) * fe.dx - k * fe.dot(fe.nabla_grad(p_ - p_n), v) * fe.dx

# Assemble matrices
A1 = fe.assemble(a1)
A2 = fe.assemble(a2)
A3 = fe.assemble(a3)

# Apply boundary conditions to matrices
[bc.apply(A1) for bc in bcu]
[bc.apply(A2) for bc in bcp]

# Create XDMF files for visualization output
xdmffile_u = fe.XDMFFile('navier_stokes_cylinder/velocity.xdmf')
xdmffile_p = fe.XDMFFile('navier_stokes_cylinder/pressure.xdmf')

# Create time series (for use in reaction_system.py)
timeseries_u = fe.TimeSeries('navier_stokes_cylinder/velocity_series')
timeseries_p = fe.TimeSeries('navier_stokes_cylinder/pressure_series')

# Save mesh to file (for use in reaction_system.py)
fe.File('navier_stokes_cylinder/cylinder.xml.gz') << mesh

# Create progress bar
progress = fe.Progress('Time-stepping')
fe.set_log_level(0)

# Time-stepping
t = 0
for n in range(num_steps):

    # Update current time
    t += dt

    # Step 1: Tentative velocity step
    b1 = fe.assemble(L1)
    [bc.apply(b1) for bc in bcu]
    fe.solve(A1, u_.vector(), b1, 'bicgstab', 'hypre_amg')

    # Step 2: Pressure correction step
    b2 = fe.assemble(L2)
    [bc.apply(b2) for bc in bcp]
    fe.solve(A2, p_.vector(), b2, 'bicgstab', 'hypre_amg')

    # Step 3: Velocity correction step
    b3 = fe.assemble(L3)
    fe.solve(A3, u_.vector(), b3, 'cg', 'sor')

    # Plot solution
    fe.plot(u_, title='Velocity')
    fe.plot(p_, title='Pressure')

    # Save solution to file (XDMF/HDF5)
    xdmffile_u.write(u_, t)
    xdmffile_p.write(p_, t)

    # Save nodal values to file
    timeseries_u.store(u_.vector(), t)
    timeseries_p.store(p_.vector(), t)

    # Update previous solution
    u_n.assign(u_)
    p_n.assign(p_)

    # Update progress bar
    progress._assign(t / T)
    print('u max:', np.max(np.array(u_.vector())))

# Hold plot
plt.show()