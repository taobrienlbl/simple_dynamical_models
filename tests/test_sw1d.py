import numpy as np
from simplemodels.shallowwater1d import *

def test_periodic_boundaries():
    """ Test a: periodic boundaries """

    nx = ny = 10
    a = np.zeros([ny+2,nx+2])
    a[c] = 1

    # apply periodic boundaries in both the x- and y- directions
    periodic_boundary(a, axis = -1)
    periodic_boundary(a, axis = -2)

    # expected result: array elements are 1 at the boundaries
    assert np.all(a == 1)

def test_instantiation():
    """ Test 0: Instantiation """

    test0 = SW1DSolver()

    # expected result: no errors raised

def test_run_one_step():
    """ Test 1: Run 1 step w/o breaking """
    test1 = SW1DSolver(state0 = 0, advection_testing = True)
    test1.RHS(test1._state_, 0, test1._solver_tmp_[0,...])

    # expected result: no errors raised

def test_run_one_step_advection():
    """ Test 2: Run 1 step advection-only w/o breaking """
    test2 = SW1DSolver(advection_testing=True, state0 = 1)
    test2.RHS(test2._state_, 0, test2._solver_tmp_[0,...])

    # expected result: no errors raised

def test_run_advection_one_day():
    """ Test 3: Run 1 day advection-only on uniform field """
    state0 = 1 # initialize with all 1's
    test3 = SW1DSolver(advection_testing=True, state0 = state0)

    test3.step_forward(ndays = 1)

    # expected result: initial field is identical to final field
    assert np.all(test3._state_[:,1:-1,1:-1] == state0)

def test_advect_uniform_field_one_step():
    """ Test 4: Run 1 step advection-only on a non-uniform field """
    # set the initial condition to be a bell
    nx = ny = 101
    dx = dy = 50e3
    s_b = 5
    a_b = 10
    cx = (nx-1)/2
    cy = (ny-1)/2
    x = np.arange(nx)
    y = np.arange(ny)
    x, y = np.meshgrid(x,y)
    test4 = SW1DSolver(
        nx = nx, ny = ny, dx = dx, dy = dy,
        advection_testing=True, state0 = 0)

    # set the wind speed
    U0 = 10 # m/s

    state0 = np.array(test4._state_[:,1:-1, 1:-1], copy = True)
    state0[test4.state_indices['h'],...] = \
        a_b * np.exp(-((x - cx)**2 + (y - cy)**2)/(4*(s_b**2)))
    state0[test4.state_indices['u'],...] = U0 

    test4.step_forward(ndays = test4.dt/86400, state0 = state0)

    # expected result: final field differs from initial field
    assert np.logical_not(np.all(test4._state_[:,1:-1,1:-1] == state0))

def test_advect_bell_one_cycle():
    """ Test 5: Advect a full cycle with a horizontal periodic boundary condition """
    # set the initial condition to be a bell
    nx = ny = 101
    dx = dy = 50e3
    s_b = 10 
    a_b = 10
    cx = (nx-1)/2
    cy = (ny-1)/2
    x = np.arange(nx)
    y = np.arange(ny)
    x, y = np.meshgrid(x,y)
    test5 = SW1DSolver(
        nx = nx, ny = ny, dx = dx, dy = dy,
        advection_testing=True, state0 = 0)

    # set the wind speed
    U0 = 10 # m/s

    state0 = np.array(test5._state_[:,1:-1, 1:-1], copy = True)
    state0[test5.state_indices['h'],...] = \
        a_b * np.exp(-((x - cx)**2 + (y - cy)**2)/(4*(s_b**2)))
    # advect diagonally at a total speed of U0
    state0[test5.state_indices['u'],...] = U0 *np.sqrt(2)/2
    state0[test5.state_indices['v'],...] = U0 *np.sqrt(2)/2

    # calculate the time required for one full transit
    x_domain = np.sqrt((nx * dx)**2 + (ny * dy)**2)
    t_transit = x_domain / U0

    # adjust dt so we can do exactly one full transit
    n_transit = int(t_transit/test5.dt)
    test5.dt = t_transit / n_transit

    test5.step_forward(ndays = t_transit/86400, state0 = state0)

    # calculate the mean squared error
    MSE = np.sqrt(np.mean((state0[2,...]-test5._state_[2,...][c])**2))

    # expected result: mean squared error is small
    assert MSE < 0.05

def test_zonal_geostrophic_jet():
    """ Test 6: symmetric, geostrophically-balanced zonal jets """
    # set the initial condition to be a bell
    nx = ny = 101
    dx = dy = 50e3
    s_b = 5 
    a_b = 10
    h0 = 5e3 # m
    u0 = 30 # m/s
    cx = (nx-1)/2
    cy = (ny-1)/2
    x = np.arange(nx)
    y = np.arange(ny)
    x, y = np.meshgrid(x,y)
    eta_b = 0 # m
    f = 1e-5 # 1/s
    g = 9.806 # m/s/s

    # make a zonal jet
    state0 = np.zeros([3,ny,nx])
    state0[0,...] = \
        u0 * np.exp(-(y - (cy + cy/2))**2/(2*s_b**2)) \
        - u0 * np.exp(-(y - (cy - cy/2))**2/(2*s_b**2))

    #state0[1,...] = \
    #      u0 * np.exp(-(x - (cx + cx/2))**2/(2*s_b**2)) \
    #    - u0 * np.exp(-(x - (cx - cx/2))**2/(2*s_b**2))

    # calculate the geostrophically-balanced height field
    state0[2,...] = h0
    for i in range(-1,ny-2):
        state0[2,i+2,:] = state0[2,i,:] - (2*dy*f/g)*state0[0,i+1,:]
    for j in range(-1,nx-2):
        state0[2,:,j+2] = state0[2,:,j] + (2*dx*f/g)*state0[1,:,j+1]

    test6 = SW1DSolver(
        nx = nx, ny = ny, dx = dx, dy = dy,
        f = f, eta_b = eta_b, g = g,
        advection_testing=False, state0 = state0)

    test6.step_forward(ndays = 1*test6.dt/86400, state0 = state0)


    # expected result: no change
    assert np.all(np.isclose(test6._state_[:,1:-1,1:-1],state0))

def test_meridional_geostrophic_jet():
    """ Test 7: symmetric, geostrophically-balanced meridional jets """
    # set the initial condition to be a bell
    nx = ny = 101
    dx = dy = 50e3
    s_b = 5 
    a_b = 10
    h0 = 5e3 # m
    u0 = 30 # m/s
    cx = (nx-1)/2
    cy = (ny-1)/2
    x = np.arange(nx)
    y = np.arange(ny)
    x, y = np.meshgrid(x,y)
    eta_b = 0 # m
    f = 1e-5 # 1/s
    g = 9.806 # m/s/s

    # make a meridional jet
    state0 = np.zeros([3,ny,nx])
    state0[1,...] = \
        u0 * np.exp(-(x - (cx + cx/2))**2/(2*s_b**2)) \
        - u0 * np.exp(-(x - (cx - cx/2))**2/(2*s_b**2))

    # calculate the geostrophically-balanced height field
    state0[2,...] = h0
    for i in range(-1,ny-2):
        state0[2,i+2,:] = state0[2,i,:] - (2*dy*f/g)*state0[0,i+1,:]
    for j in range(-1,nx-2):
        state0[2,:,j+2] = state0[2,:,j] + (2*dx*f/g)*state0[1,:,j+1]

    test7 = SW1DSolver(
        nx = nx, ny = ny, dx = dx, dy = dy,
        f = f, eta_b = eta_b, g = g,
        advection_testing=False, state0 = state0)

    test7.step_forward(ndays = 1*test7.dt/86400, state0 = state0)

    # expected result: no change
    assert np.all(np.isclose(test7._state_[:,1:-1,1:-1],state0))

def test_zonal_geostrophic_jet_fixed_y_boundary():
    """ Test 8: symmetric, geostrophically-balanced zonal jets with fixed boundary condition """
    # set the initial condition to be a bell
    nx = ny = 101
    dx = dy = 50e3
    s_b = 5 
    a_b = 10
    h0 = 5e3 # m
    u0 = 30 # m/s
    cx = (nx-1)/2
    cy = (ny-1)/2
    x = np.arange(nx)
    y = np.arange(ny)
    x, y = np.meshgrid(x,y)
    eta_b = 0 # m
    f = 1e-5 # 1/s
    g = 9.806 # m/s/s

    # make a zonal jet
    state0 = np.zeros([3,ny,nx])
    state0[0,...] = \
        u0 * np.exp(-(y - (cy + cy/2))**2/(2*s_b**2)) \
        - u0 * np.exp(-(y - (cy - cy/2))**2/(2*s_b**2))

    # calculate the geostrophically-balanced height field
    state0[2,...] = h0
    for i in range(-1,ny-2):
        state0[2,i+2,:] = state0[2,i,:] - (2*dy*f/g)*state0[0,i+1,:]
    for j in range(-1,nx-2):
        state0[2,:,j+2] = state0[2,:,j] + (2*dx*f/g)*state0[1,:,j+1]

    test8 = SW1DSolver(
        nx = nx, ny = ny, dx = dx, dy = dy,
        f = f, eta_b = eta_b, g = g,
        advection_testing=False, state0 = state0,
        y_is_periodic=False, x_is_periodic=True)

    test8.step_forward(ndays = 1*test8.dt/86400, state0 = state0)

    # expected result: no change
    assert np.all(np.isclose(test8._state_[:,1:-1,1:-1],state0))
