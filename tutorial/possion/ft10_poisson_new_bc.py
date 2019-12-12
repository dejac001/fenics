import fenics as fe
import numpy as np
import matplotlib.pyplot as plt

# Create mesh and define function space----------
#   defines a uniform finite element mesh over the unit square
#   cells in 2D are triangles with straight sides.
#   8 x 8 divides square into 8 x 8 rectangles, each divided into 2 triangles (=128 triangles)
#   The total number of vertices in mesh is 9 x 9 = 81
mesh = fe.UnitSquareMesh(8, 8)
V = fe.FunctionSpace(mesh, 'P', 1)
# ^^ note: In mathematics, distinguish between trial and test spaces V and \hat{V}
#          The only difference in the present problem is the boundary conditions.
#          In FEniCS, we do not specify the boundary conditions as part of the function
#          space, so it is sufficient to work with one common space *V* for both the trial
#          and test functions in the program

# Define boundary condition u = u_D on \delta\Omega
u_D = fe.Expression(
    # 1 + x^2 + 2y^2
    '1 + x[0]*x[0] + 2*x[1]*x[1]',  # must be written with a C++ syntax
    degree=2    # *degree* specifies how the expression should be treated in computations
                #   on each local element, FEniCS will interpolate the expression into
                #   a finite element space of the specified degree
                #   to obtain optimal (order of) accuracy in computations, it is usually
                #   a good choice to use the same degree as for the space V that is used
                #   for the trial and test functions. Here, since we have an exact solution
                #   we use a higher degree of accuracy though
)


def boundary_D(x, on_boundary):
    """function or object defining which points belong to the boundary.

    *Essential boundary conditions* because they need to be imposed
    explicitly as part of the trial space

    FEniCS already knows whether the piont belongs to the *actual* boundary

       (the mathematical boundary of the domain) and kindly shares this
       information with you in the variable on_boundary

      You may choose to use this information (as done below) or ignore completely

      Alternatively, could do

      return abs(x[0]) < 1e-14 or abs(x[1]) < 1e-14 or abs(x[0]-1) < 1e-14 or abs(x[1]-1) < 1e-14

      or, could use *near* command in FEniCS

      return fe.near(x[0], 0, 1e-14) or fe.near(x[1], 0, 1e-14) or fe.near(x[0], 1, 1e-14) or fe.near(x[1], 1, 1e-14)

    """
    tol=1e-14
    return on_boundary and (
        fe.near(x[0], 0, tol) or fe.near(x[0], 1, tol)
    )


bc = fe.DirichletBC(V, u_D, boundary_D)
g = fe.Expression('-4*x[1]', degree=1)

# Define variational problem
u = fe.TrialFunction(V)
v = fe.TestFunction(V)
f = fe.Constant(-6.0)
# ^^ note: could also do
#           f = Expression('-6', degree=0)
#       but it is more efficient to use *Constant*

a = fe.dot(fe.grad(u), fe.grad(v)) * fe.dx
L = f*v*fe.dx - g*v*fe.ds  # ds implies a boundary integral, while dx implies an integral over domain \Omega
# Note that the integration *ds is carried out over the entire boundary, including the Dirichlet boundary
# However, since the test function v vanishes on the Dirichlet boundary (as a result of specifying a DirichletBC),
#   the integral will only include the contribution from the Neumann boundary

# compute solution
#       Redefine u to be a Function object representing the solution
#       - two types of objects that u refers to are equal from a mathematical point of view
#           so natural to use the same variable name for both objects
u = fe.Function(V)
fe.solve(a == L, u, bc)

# plot solution and mesh
fe.plot(mesh)
fe.plot(u)

# save solution to file in VTK format
vtkfile = fe.File('solution.pvd')
vtkfile << u

# Computer error in L2 norm
error_L2 = fe.errornorm(u_D, u, 'L2')

# compute maximum error at vertices
vertex_values_u_D = u_D.compute_vertex_values(mesh)
vertex_values_u = u.compute_vertex_values(mesh)
error_max = np.max(np.abs(vertex_values_u_D - vertex_values_u))

# print errors
print('error_L2 =', error_L2)
print('error_max =', error_max)

# Hold plot
plt.show()
