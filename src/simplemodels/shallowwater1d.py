import numpy as np
import xarray as xr

# define helper tuples for shifting data in the left, right, up, and down directions
# this is used for centered finite differences
c = 2*(slice(1,-1,None),)
l = (slice(1,-1,None), slice(None, -2, None))
r = (slice(1,-1,None), slice(2, None, None))
d = (slice(None, -2, None), slice(1, -1, None))
u = (slice(2, None, None), slice(1, -1, None))

def zero_neumann_boundary(array, axis = -1):
    """ Implements a boundary condition for no derivative in the given quantity. """

    ndims = len(array.shape)
    slice_accessor_lhs = ndims*[slice(None,None,None)]
    slice_accessor_rhs = ndims*[slice(None,None,None)]

   # extend the boundaries on the low-index side
    slice_accessor_lhs[axis] = 0
    slice_accessor_rhs[axis] = 1
    array[tuple(slice_accessor_lhs)] = array[tuple(slice_accessor_rhs)]

    # extend the boundaries on the high-index side
    slice_accessor_lhs[axis] = -1 
    slice_accessor_rhs[axis] = -2
    array[tuple(slice_accessor_lhs)] = array[tuple(slice_accessor_rhs)]


def zero_dirichlet_boundary(array, axis = -1):
    """ Implements a boundary condition that sets the boundary to 0. 
        
        input:
        ------

            array : the input array

            axis : the axis along which to apply the boundary condition
    """

    ndims = len(array.shape)
    slice_accessor_lhs = ndims*[slice(None,None,None)]
    slice_accessor_lhs[axis] = 0
    slice_accessor_rhs = ndims*[slice(None,None,None)]
    slice_accessor_rhs[axis] = -1

    # set the boundary values
    array[tuple(slice_accessor_lhs)] = 0
    array[tuple(slice_accessor_rhs)] = 0

    return


def periodic_boundary(array, axis = -1):
    """ Implements a periodic boundary condition via ghost cells for the specified dimension """
    ndims = len(array.shape)
    slice_accessor_lhs = ndims*[slice(None,None,None)]
    slice_accessor_rhs = ndims*[slice(None,None,None)]
    
    # mirror the end of the array to the beginning
    slice_accessor_lhs[axis] = 0
    slice_accessor_rhs[axis] = -2 
    array[tuple(slice_accessor_lhs)] = array[tuple(slice_accessor_rhs)]
    
    # mirror the beginning of the array to the end
    slice_accessor_lhs[axis] = -1 
    slice_accessor_rhs[axis] = 1 
    array[tuple(slice_accessor_lhs)] = array[tuple(slice_accessor_rhs)]
    
    return

def advection_test_UV_tendency(state, eta, tendency_array, U_index = 1, V_index = 2):
    """ Returns zeros for the U/V tendencies; used for testing of advection """
    tendency_array[U_index,...] = 0.0
    tendency_array[V_index,...] = 0.0
    
    return
    
def cfd_h_sw1d_tendency(state, tendency_array, dx, dy, H_index = 0, U_index = 1, V_index = 2):
    """ sets the tendency for the height variable in the shallow-water system, calculated using 2nd-order finite differences. """
    # get pointers to the u,v, h and h tendency arrays
    U = state[U_index,:,:]
    V = state[V_index,:,:]
    H = state[H_index,:,:]
    dHdt = tendency_array[H_index,:,:]
    
    # it is assumed that boundary conditions are already loaded in the single layer of
    # ghost cells surrounding the state array
    
    # dhdt = -u * grad(h) - h * grad(u)
    dHdt[c] = \
        -U[c] * (H[r] - H[l])/(2*dx) \
        -V[c] * (H[u] - H[d])/(2*dy) \
        -H[c] * (U[r] - U[l])/(2*dx) \
        -H[c] * (V[u] - V[d])/(2*dy)
    
    # note that the above modifies a view of tendency_array, so no data need to be returned; the modification is in-place
    return

def cfd_uv_sw1d_tendency(state, eta, tendency_array, f, g, dx, dy, H_index = 0, U_index = 1, V_index = 2):
    """ sets the tendency for the u and v variables in the shallow-water system, calculated using 2nd-order finite differences. """
    # get pointers to the u,v, h and h tendency arrays
    U = state[U_index,:,:]
    V = state[V_index,:,:]
    H = state[H_index,:,:]
    dUdt = tendency_array[U_index,:,:]
    dVdt = tendency_array[V_index,:,:]
    
    # it is assumed that boundary conditions are already loaded in the single layer of
    # ghost cells surrounding the state array
    
    dx2 = 2*dx
    dy2 = 2*dy
    
    # dudt = -u * grad(u) + f * v - g * d_eta/dx
    dUdt[c] = \
        -U[c]*(U[r] - U[l])/dx2 \
        -V[c]*(U[u] - U[d])/dy2 \
        +f*V[c]                 \
        -g*(eta[r] - eta[l])/dx2
    
    # dvdt = -u * grad(v) - f * U - g * d_eta/dy
    dVdt[c] = \
        -U[c]*(V[r] - V[l])/dx2 \
        -V[c]*(V[u] - V[d])/dy2 \
        -f*U[c]                 \
        -g*(eta[u] - eta[d])/dy2
    
    # note that the above modifies a view of tendency_array, so no data need to be returned; the modification is in-place
    return
    
        
class SW1DSolver:
    
    def __init__(
        self,
        nx = 32,
        ny = 32,
        dx = 50e3,
        dy = 50e3,
        c_cfl = 30,
        cfl = 0.1,
        g = 9.806,
        f = 0, 
        eta_b = 0,
        x_is_periodic = True,
        y_is_periodic = True,
        array_type = np.float64,
        advection_testing = False,
        state0 = None,
        current_t = 0,
    ):
        """ Initialize the solver class; allocate variables, etc. """
        # store inputs
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.c_cfl = c_cfl
        self.cfl = cfl
        self.g = g
        self.f = f
        self.eta_b = eta_b,
        self.x_is_periodic = x_is_periodic
        self.y_is_periodic = y_is_periodic
        self.array_type = array_type
        self.advection_testing = advection_testing
        self.current_t = current_t
        
        # initialize the state variables
        self.state_indices = dict(u = 0, v = 1, h = 2)
        self.num_state = len(self.state_indices)
        
        # allocate the state variable
        # (index 0 is the state variable, index 1 is y, index 2 is x)
        # Note that the dimension size is increased by 2 to account for ghost cells
        self._state_ = np.empty([self.num_state, ny+2, nx+2], dtype = array_type)
        if state0 is not None:
            # set the initial condition from what was passed in
            self._state_[:, 1:-1, 1:-1] = state0
            
            if not self.advection_testing:
                # set the CFL velocity based on the average fluid depth
                h_avg = np.mean(state0[self.state_indices['h'],:,:])
                # calculate the phase speed of typical gravity waves in this fluid
                self.c_cfl = np.sqrt(np.abs(self.g * h_avg))
            
        # determine the timestep
        self.dt = int(self.cfl * max([self.dx,self.dy]) / (self.c_cfl))
            
        # a temporary array for assisting in the RK4 solver
        self._solver_tmp_ = np.zeros([5,self.num_state, ny+2, nx+2], dtype = array_type)
        
        # a temporary array for the total height of the surface
        self._eta_tmp_ = np.zeros([ny+2, nx+2])
        
        # set the boundary condition callback function
        if x_is_periodic:
            self.x_boundary_function = lambda x: periodic_boundary(x, axis = -1)
        else:
            def fixed_x_boundary(state):
                """ Sets h and u to 0 at the x-boundary, and no-slip in v"""
                zero_neumann_boundary(
                    state[self.state_indices['h']],
                    axis = -1
                    )
                zero_neumann_boundary(
                    state[self.state_indices['u']],
                    axis = -1
                    )
                zero_neumann_boundary(
                    state[self.state_indices['v']],
                    axis = -1)
            self.x_boundary_function = fixed_x_boundary

        if y_is_periodic:
            self.y_boundary_function = lambda x: periodic_boundary(x, axis = -2)
        else:
            def fixed_y_boundary(state):
                """ Sets h and v to 0 at the y-boundary, and no-slip in u"""
                zero_neumann_boundary(
                    state[self.state_indices['h']],
                    axis = -2
                    )
                zero_neumann_boundary(
                    state[self.state_indices['v']],
                    axis = -2
                    )
                zero_neumann_boundary(
                    state[self.state_indices['u']],
                    axis = -2)
            self.y_boundary_function = fixed_y_boundary

            
        if advection_testing:
            # set the RHS for the u and v quantities such that they aren't updated
            self.uv_rhs_function = lambda state, eta, dstate_dt : \
                advection_test_UV_tendency(
                    state,
                    eta,
                    dstate_dt,
                    self.state_indices['u'],
                    self.state_indices['v'],
                )
        else:
            # set the RHS for u and v
            self.uv_rhs_function = lambda state, eta, dstate_dt : \
                cfd_uv_sw1d_tendency(
                    state,
                    eta,
                    dstate_dt,
                    self.f,
                    self.g,
                    self.dx,
                    self.dy,
                    H_index = self.state_indices['h'],
                    U_index = self.state_indices['u'],
                    V_index = self.state_indices['v'],
                )
            
        self.h_rhs_function = lambda state, dstate_dt : \
            cfd_h_sw1d_tendency(
                state,
                dstate_dt,
                self.dx,
                self.dy,
                H_index = self.state_indices['h'],
                U_index = self.state_indices['u'],
                V_index = self.state_indices['v'],
            )
        
        return
    
    def RHS(self, state, eta_b, dstate_dt):
        """ Updates dstate_dt with the right-hand side values of the 1D SW system. """
        
        # enforce the boundary conditions
        self.x_boundary_function(state)
        self.y_boundary_function(state)
        
        # update the total height field
        self._eta_tmp_[:] = state[self.state_indices['h'],:,:] + eta_b
        
        # update the U and V tendency variables;
        # this directly updates dstate_dt
        self.uv_rhs_function(state, self._eta_tmp_, dstate_dt)
        
        # update the h tendency
        # this directly updates dstate_dt
        self.h_rhs_function(state, dstate_dt)
        
     
    def step_forward(
        self,
        ndays = 5,
        state0 = None,
    ):
        """Steps the model forward a specified number of days; 
        uses the model's current state as the initial condition if one isn't given"""
        
        if state0 is not None:
            self._state_[:,1:-1,1:-1] = np.copy(np.array(state0))
            
            
        delta_t = ndays * 86400 # s
        tmax = self.current_t + delta_t
        dt = self.dt
        
        # pre-define some helper variables/views for the RK4 integration
        k1 = self._solver_tmp_[1,...]
        k2 = self._solver_tmp_[2,...]
        k3 = self._solver_tmp_[3,...]
        k4 = self._solver_tmp_[4,...]
        
        while self.current_t < tmax:
            # stage 1
            self.RHS(self._state_, self.eta_b, k1)
            
            # stage 2
            self._solver_tmp_[0,...] = self._state_ + k1 * dt/2
            self.RHS(self._solver_tmp_[0,...], self.eta_b, k2)
            
            # stage 3
            self._solver_tmp_[0,...] = self._state_ + k2 * dt/2
            self.RHS(self._solver_tmp_[0,...], self.eta_b, k3)
            
            # stage 4
            #print(self._state_.max(),k1.max(),k2.max(),k3.max(),k4.max())
            self._solver_tmp_[0,...] = self._state_ + k3 * dt
            self.RHS(self._solver_tmp_[0,...], self.eta_b, k4)
            
            # combine the tendencies into the final value
            self._state_[:] = \
                self._state_ + (1/6)*dt*(k1 + 2*k2 + 2*k3 + k4)
            
            # calculate the new time
            self.current_t += dt
            
    def get_state_xarray(self):
        """ Returns an xarray dataset containing the state variable. """
        
        x = np.arange(self.nx)*self.dx/1000
        x -= x.mean()
        y = np.arange(self.ny)*self.dy/1000
        y -= y.mean()
        t = np.array([self.current_t])
        
        out_ds = xr.Dataset()
        out_ds['time'] = xr.DataArray(t, dims = ['time'], coords = dict(time = t))
        out_ds['x'] = xr.DataArray(x, dims = ['x'], coords = dict(x = x))
        out_ds['y'] = xr.DataArray(y, dims = ['y'], coords = dict(y = y))
        
        out_ds['U'] = xr.DataArray(
            self._state_[self.state_indices['u'],:,:][c][np.newaxis,:,:],
            dims = ['time', 'y', 'x']
        )
        out_ds['V'] = xr.DataArray(
            self._state_[self.state_indices['v'],:,:][c][np.newaxis,:,:],
            dims = ['time', 'y', 'x']
        )
        out_ds['H'] = xr.DataArray(
            self._state_[self.state_indices['h'],:,:][c][np.newaxis,:,:],
            dims = ['time', 'y', 'x']
        )
        
        out_ds['time'].attrs['long_name'] = 'time'
        out_ds['time'].attrs['units'] = 'seconds since 1850-01-01 00:00:00'
        out_ds['x'].attrs['long_name'] = 'Zonal distance'
        out_ds['x'].attrs['units'] = 'km'
        out_ds['y'].attrs['long_name'] = 'Meridional distance'
        out_ds['y'].attrs['units'] = 'km'
        out_ds['U'].attrs['long_name'] = "Zonal wind"
        out_ds['U'].attrs['units'] = "m/s"
        out_ds['V'].attrs['long_name'] = "Meridional wind"
        out_ds['V'].attrs['units'] = "m/s"
        
        return out_ds
        
        
            
    