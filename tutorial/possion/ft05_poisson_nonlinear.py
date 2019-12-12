"""
Consider the problem

.. math:
    -\nabla\cdot(q(u)\nabla u)=f

in :math:`\Omega` with :math:`u=u_{\mathrm{D}}` on the
boundary :math:`\delta\Omega`

The coefficient :math:`q(u)` makes the equation nonlinear

We take :math:`q(u) = 1 + u^2` and define a
two-dimensional manufactured solution that
is linear in :math:`x` and :math:`y`

"""
import fenics as fe
import sympy as sym


def q(_u):
    """:return nonlinear coefficient"""
    return 1 + _u*_u


def boundary(x, on_boundary):
    return on_boundary


mesh = fe.UnitSquareMesh(8, 8)
V = fe.FunctionSpace(mesh, 'P', 1)

# Use SymPy to compute f from the manufactured solution u
x, y = sym.symbols('x[0], x[1]')
u = 1 + x + 2*y
f = -sym.diff(q(u)*sym.diff(u, x), x) - sym.diff(q(u)*sym.diff(u,y), y)
f = sym.simplify(f)
# turn the expressions for u and f into C or C++ syntax for FEniCX Expression objects
#   1. ask for the C code of the expressions
u_code = sym.printing.ccode(u)
f_code = sym.printing.ccode(f)
print('u = ', u_code)
print('f = ', f_code)
#   2. after having defined the mesh, the function space, and the boundary,
#       we define the boundary value u_D as
u_D = fe.Expression(u_code, degree=1)
f = fe.Expression(f_code, degree=1)

bc = fe.DirichletBC(V, u_D, boundary)
u = fe.Function(V)
v = fe.TestFunction(V)
f = fe.Expression(f_code, degree=1)
F = q(u)*fe.dot(fe.grad(u), fe.grad(v))*fe.dx - f*v*fe.dx

fe.solve(F == 0, u, bc)
