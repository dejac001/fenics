import fenics as fe
import matplotlib.pyplot as plt

T = 5.0                 # final time
num_steps = 500         # number of time steps
dt = T / num_steps      # time step size
eps = 0.01              # diffusion coefficient
K = 10.0                # reaction rate

# Read mesh from file--from previous problem
mesh = fe.Mesh(
    'navier_stokes_cylinder/cylinder.xml.gz'
)

# Define function space for velocity
W = fe.VectorFunctionSpace(mesh, 'P', 2)

# Define function space for system of concentrations
P1 = fe.FiniteElement('P', fe.triangle, 1)
element = fe.MixedElement([P1, P1, P1])
V = fe.FunctionSpace(mesh, element)

# Define test functions
v_1, v_2, v_3 = fe.TestFunctions(V)

# Define functions for velocity and concentrations
w = fe.Function(W)
u = fe.Function(V)
u_n = fe.Function(V)

# Split system functions to access components
u_1, u_2, u_3 = fe.split(u)
u_n1, u_n2, u_n3 = fe.split(u_n)

# Define source terms
f_1 = fe.Expression(
    'pow(x[0] - 0.1, 2), + pow(x[1]-0.1, 2) < 0.05*0.05 ? 0.1: 0', degree=1
)
f_2 = fe.Expression(
    'pow(x[0] - 0.1, 2), + pow(x[1]-0.3, 2) < 0.05*0.05 ? 0.1: 0', degree=1
)
f_3 = fe.Constant(0)

# Define expressions used in variational forms
k = fe.Constant(dt)
K = fe.Constant(K)
eps = fe.Constant(eps)

# Define variational problem
F = ((u_1 - u_n1) / k)*v_1*fe.dx + fe.dot(w, fe.grad(u_1))*v_1*fe.dx \
    + eps*fe.dot(fe.grad(u_1), fe.grad(v_1))*fe.dx + K*u_1*u_2*v_1*fe.dx \
    + ((u_2 - u_n2) / k)*v_2*fe.dx + fe.dot(w, fe.grad(u_2))*v_2*fe.dx \
    + eps*fe.dot(fe.grad(u_2), fe.grad(v_2))*fe.dx + K*u_1*u_2*v_2*fe.dx \
    + ((u_3 - u_n3) / k)*v_3*fe.dx + fe.dot(w, fe.grad(u_3))*v_3*fe.dx \
    + eps*fe.dot(fe.grad(u_3), fe.grad(v_3))*fe.dx - K*u_1*u_2*v_3*fe.dx + K*u_3*v_3*fe.dx \
    - f_1*v_1*fe.dx - f_2*v_2*fe.dx - f_3*v_3*fe.dx

# Create time series for reading velocity data
timeseries_w = fe.TimeSeries('navier_stokes_cylinder/velocity_series')

# Create VTK files for visualization output
vtkfile_u_1 = fe.File('reaction_system/u_1.pvd')
vtkfile_u_2 = fe.File('reaction_system/u_2.pvd')
vtkfile_u_3 = fe.File('reaction_system/u_3.pvd')

# Create progress bar
progress = fe.Progress('Time-stepping')
fe.set_log_active(30)

# Time-stepping
t = 0
for n in range(num_steps):

    # Update current time
    t += dt

    # Read velocity from file
    timeseries_w.retrieve(w.vector(), t)

    # Solve variational problem for time step
    fe.solve(F == 0, u)

    # Save solution to file (VTK)
    _u_1, _u_2, _u_3 = u.split()
    vtkfile_u_1 << (_u_1, t)
    vtkfile_u_2 << (_u_2, t)
    vtkfile_u_3 << (_u_3, t)

    # Update previous solution
    u_n.assign(u)

    # Update progress bar
    progress._assign(t/T)

_u_1, _u_2, _u_3 = u.split()
fe.plot(_u_1, interactive=True)
plt.show()
