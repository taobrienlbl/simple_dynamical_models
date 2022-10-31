![build](https://github.com/taobrienlbl/simple_dynamical_models/actions/workflows/test-simplemodels.yml/badge.svg)


# Simple dynamical models
This code contains a set of dynamical models suitable for teaching.  The intent is for these to have a low set of requirements such that they can be run in Google CoLab or other environments with little-or-no installation required.

## `simplemodels.shallowwater1d`
Implements and a numerical solution to the single-layer shallow water equations on a cartesian geometry, using 2nd order finite differences and a 4th-order Runge-Kutta solver for stepping forward in time.  Boundary conditions are implemented in a flexible way, allowing for periodicity in either dimension, flux-free boundary conditions, or numerically-specified boundary conditions via use of ghost cells.

The equation set solved follows Holton, Ch 3.1 equations SW.1--4:

$$\begin{eqnarray}
\frac{D\mathbf{u}}{dt} + \mathbf{f} \times \mathbf{u} & = & -g \nabla \eta \\
\frac{Dh}{dt} + h \nabla \cdot \mathbf{u} & = & 0 \\
h(x,y,t) & = & \eta(x,y,t) - \eta_b(x,y) \\
\frac{D}{Dt} & \equiv & \frac{\partial}{\partial t} + u \frac{\partial}{\partial x} + v \frac{\partial}{\partial y} \\
\end{eqnarray}$$