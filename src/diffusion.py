"""
Diffusion and Advection-Diffusion Equation Solvers
Numerical methods for solving diffusion, advection-diffusion, and reaction-diffusion equations
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from typing import Callable, Optional, Tuple, Dict, Union
import warnings
warnings.filterwarnings('ignore')


class DiffusionSolver:
    """
    Solver for diffusion and advection-diffusion equations
    """
    
    def __init__(self, method: str = 'finite_difference'):
        """
        Initialize diffusion solver
        
        Args:
            method: Numerical method ('finite_difference', 'finite_element', 'spectral')
        """
        self.method = method
    
    def solve_diffusion_1d(self,
                          length: float = 1.0,
                          nx: int = 100,
                          nt: int = 1000,
                          diffusion_coeff: float = 0.01,
                          initial_condition: Union[str, Callable] = 'gaussian',
                          boundary_conditions: str = 'dirichlet_zero',
                          source_term: Optional[Callable] = None,
                          end_time: float = 1.0) -> Dict:
        """
        Solve 1D diffusion equation: ∂u/∂t = D∂²u/∂x² + S(x,t)
        
        Args:
            length: Domain length
            nx: Number of spatial grid points
            nt: Number of time steps
            diffusion_coeff: Diffusion coefficient D
            initial_condition: Initial condition
            boundary_conditions: Boundary condition type
            source_term: Source term function S(x,t)
            end_time: Final time
            
        Returns:
            Dictionary containing solution data
        """
        # Create spatial grid
        x = np.linspace(0, length, nx)
        dx = x[1] - x[0]
        
        # Create time grid
        dt = end_time / nt
        times = np.linspace(0, end_time, nt + 1)
        
        # Check stability
        r = diffusion_coeff * dt / (dx**2)
        if r > 0.5:
            warnings.warn(f"Explicit scheme may be unstable: r = {r:.3f} > 0.5")
        
        # Set initial condition
        u0 = self._get_initial_condition_1d(initial_condition, x)
        
        # Allocate solution array
        u = np.zeros((nt + 1, nx))
        u[0] = u0
        
        # Time stepping (explicit scheme)
        for n in range(nt):
            # Interior points
            u[n + 1, 1:-1] = (u[n, 1:-1] + 
                              r * (u[n, 2:] - 2*u[n, 1:-1] + u[n, :-2]))
            
            # Add source term if provided
            if source_term is not None:
                for i in range(1, nx - 1):
                    u[n + 1, i] += dt * source_term(x[i], times[n])
            
            # Apply boundary conditions
            self._apply_boundary_conditions_1d(u, n + 1, boundary_conditions)
        
        return {
            'solution': u,
            'grid': {'x': x},
            'times': times,
            'parameters': {
                'diffusion_coeff': diffusion_coeff,
                'dx': dx,
                'dt': dt,
                'r_parameter': r
            }
        }
    
    def solve_advection_diffusion(self,
                                 domain_size: Tuple[float, float] = (1.0, 1.0),
                                 grid_points: Tuple[int, int] = (50, 50),
                                 time_steps: int = 1000,
                                 diffusion_coeff: float = 0.01,
                                 velocity_field: Union[str, Tuple[float, float]] = (0.1, 0.0),
                                 initial_condition: Union[str, Callable] = 'gaussian_peak',
                                 boundary_conditions: str = 'periodic',
                                 end_time: float = 2.0) -> Dict:
        """
        Solve 2D advection-diffusion equation: ∂u/∂t + v·∇u = D∇²u
        
        Args:
            domain_size: (Lx, Ly) domain dimensions
            grid_points: (nx, ny) grid resolution
            time_steps: Number of time steps
            diffusion_coeff: Diffusion coefficient D
            velocity_field: Velocity field (vx, vy) or string preset
            initial_condition: Initial concentration
            boundary_conditions: Boundary condition type
            end_time: Final time
            
        Returns:
            Dictionary containing solution data
        """
        Lx, Ly = domain_size
        nx, ny = grid_points
        nt = time_steps
        D = diffusion_coeff
        
        # Create spatial grid
        x = np.linspace(0, Lx, nx)
        y = np.linspace(0, Ly, ny)
        X, Y = np.meshgrid(x, y)
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        
        # Create time grid
        dt = end_time / nt
        times = np.linspace(0, end_time, nt + 1)
        
        # Get velocity field
        vx, vy = self._get_velocity_field(velocity_field, X, Y)
        
        # Check stability conditions
        r_diff = D * dt * (1/dx**2 + 1/dy**2)
        cfl_x = np.max(np.abs(vx)) * dt / dx
        cfl_y = np.max(np.abs(vy)) * dt / dy
        
        if r_diff > 0.25:
            warnings.warn(f"Diffusion stability: r = {r_diff:.3f} > 0.25")
        if max(cfl_x, cfl_y) > 1.0:
            warnings.warn(f"Advection CFL: max({cfl_x:.3f}, {cfl_y:.3f}) > 1.0")
        
        # Set initial condition
        u0 = self._get_initial_condition_2d(initial_condition, X, Y)
        
        # Allocate solution array
        u = np.zeros((nt + 1, ny, nx))
        u[0] = u0
        
        # Time stepping using operator splitting
        for n in range(nt):
            # Step 1: Advection (upwind scheme)
            u_temp = self._advection_step(u[n], vx, vy, dx, dy, dt)
            
            # Step 2: Diffusion (explicit scheme)
            u[n + 1] = self._diffusion_step(u_temp, D, dx, dy, dt)
            
            # Apply boundary conditions
            self._apply_boundary_conditions_2d(u, n + 1, boundary_conditions)
        
        return {
            'solution': u,
            'grid': {'x': x, 'y': y, 'X': X, 'Y': Y},
            'times': times,
            'velocity_field': {'vx': vx, 'vy': vy},
            'parameters': {
                'diffusion_coeff': D,
                'dx': dx,
                'dy': dy,
                'dt': dt,
                'r_diffusion': r_diff,
                'cfl_x': cfl_x,
                'cfl_y': cfl_y
            }
        }
    
    def solve_reaction_diffusion(self,
                                domain_size: Tuple[float, float] = (1.0, 1.0),
                                grid_points: Tuple[int, int] = (100, 100),
                                time_steps: int = 5000,
                                diffusion_coeffs: Tuple[float, float] = (2e-5, 1e-5),
                                reaction_params: Dict = None,
                                initial_condition: str = 'random_spots',
                                boundary_conditions: str = 'neumann_zero',
                                end_time: float = 50.0) -> Dict:
        """
        Solve Gray-Scott reaction-diffusion system
        
        ∂u/∂t = Du∇²u - uv² + F(1-u)
        ∂v/∂t = Dv∇²v + uv² - (F+k)v
        
        Args:
            domain_size: (Lx, Ly) domain dimensions
            grid_points: (nx, ny) grid resolution
            time_steps: Number of time steps
            diffusion_coeffs: (Du, Dv) diffusion coefficients
            reaction_params: Reaction parameters {'F': feed_rate, 'k': kill_rate}
            initial_condition: Initial condition type
            boundary_conditions: Boundary condition type
            end_time: Final time
            
        Returns:
            Dictionary containing solution data
        """
        if reaction_params is None:
            # Default parameters for spot patterns
            reaction_params = {'F': 0.037, 'k': 0.06}
        
        Lx, Ly = domain_size
        nx, ny = grid_points
        nt = time_steps
        Du, Dv = diffusion_coeffs
        F = reaction_params['F']
        k = reaction_params['k']
        
        # Create spatial grid
        x = np.linspace(0, Lx, nx)
        y = np.linspace(0, Ly, ny)
        X, Y = np.meshgrid(x, y)
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        
        # Create time grid
        dt = end_time / nt
        times = np.linspace(0, end_time, nt + 1)
        
        # Set initial conditions
        u0, v0 = self._get_reaction_diffusion_initial(initial_condition, X, Y)
        
        # Allocate solution arrays
        u = np.zeros((nt + 1, ny, nx))
        v = np.zeros((nt + 1, ny, nx))
        u[0] = u0
        v[0] = v0
        
        # Precompute diffusion operators
        ru = Du * dt / (dx**2)
        rv = Dv * dt / (dy**2)
        
        # Time stepping
        for n in range(nt):
            u_curr = u[n]
            v_curr = v[n]
            
            # Reaction terms
            reaction_u = -u_curr * v_curr**2 + F * (1 - u_curr)
            reaction_v = u_curr * v_curr**2 - (F + k) * v_curr
            
            # Diffusion terms (5-point stencil)
            laplacian_u = np.zeros_like(u_curr)
            laplacian_v = np.zeros_like(v_curr)
            
            # Interior points
            laplacian_u[1:-1, 1:-1] = ((u_curr[1:-1, 2:] - 2*u_curr[1:-1, 1:-1] + u_curr[1:-1, :-2]) / dx**2 +
                                      (u_curr[2:, 1:-1] - 2*u_curr[1:-1, 1:-1] + u_curr[:-2, 1:-1]) / dy**2)
            
            laplacian_v[1:-1, 1:-1] = ((v_curr[1:-1, 2:] - 2*v_curr[1:-1, 1:-1] + v_curr[1:-1, :-2]) / dx**2 +
                                      (v_curr[2:, 1:-1] - 2*v_curr[1:-1, 1:-1] + v_curr[:-2, 1:-1]) / dy**2)
            
            # Update concentrations
            u[n + 1] = u_curr + dt * (Du * laplacian_u + reaction_u)
            v[n + 1] = v_curr + dt * (Dv * laplacian_v + reaction_v)
            
            # Apply boundary conditions
            self._apply_boundary_conditions_2d_species(u, v, n + 1, boundary_conditions)
        
        return {
            'solution': {'u': u, 'v': v},
            'grid': {'x': x, 'y': y, 'X': X, 'Y': Y},
            'times': times,
            'parameters': {
                'diffusion_coeffs': diffusion_coeffs,
                'reaction_params': reaction_params,
                'dx': dx,
                'dy': dy,
                'dt': dt
            }
        }
    
    def solve_nonlinear_diffusion(self,
                                 domain_size: float = 1.0,
                                 nx: int = 100,
                                 nt: int = 1000,
                                 diffusivity_func: Callable = None,
                                 initial_condition: Union[str, Callable] = 'gaussian',
                                 boundary_conditions: str = 'dirichlet_zero',
                                 end_time: float = 1.0) -> Dict:
        """
        Solve nonlinear diffusion equation: ∂u/∂t = ∂/∂x(D(u)∂u/∂x)
        
        Args:
            domain_size: Domain length
            nx: Number of spatial grid points
            nt: Number of time steps
            diffusivity_func: Diffusivity function D(u)
            initial_condition: Initial condition
            boundary_conditions: Boundary condition type
            end_time: Final time
            
        Returns:
            Dictionary containing solution data
        """
        if diffusivity_func is None:
            # Default: porous medium equation D(u) = u^m
            diffusivity_func = lambda u: np.maximum(u**2, 1e-10)
        
        # Create spatial grid
        x = np.linspace(0, domain_size, nx)
        dx = x[1] - x[0]
        
        # Create time grid
        dt = end_time / nt
        times = np.linspace(0, end_time, nt + 1)
        
        # Set initial condition
        u0 = self._get_initial_condition_1d(initial_condition, x)
        
        # Allocate solution array
        u = np.zeros((nt + 1, nx))
        u[0] = u0
        
        # Time stepping using implicit scheme
        for n in range(nt):
            u[n + 1] = self._implicit_nonlinear_step(u[n], diffusivity_func, dx, dt, boundary_conditions)
        
        return {
            'solution': u,
            'grid': {'x': x},
            'times': times,
            'parameters': {
                'dx': dx,
                'dt': dt,
                'diffusivity_function': 'nonlinear'
            }
        }
    
    def _advection_step(self, u: np.ndarray, vx: np.ndarray, vy: np.ndarray,
                       dx: float, dy: float, dt: float) -> np.ndarray:
        """Perform one advection step using upwind scheme"""
        u_new = u.copy()
        ny, nx = u.shape
        
        # Upwind finite differences
        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                # x-direction advection
                if vx[i, j] > 0:
                    dudx = (u[i, j] - u[i, j-1]) / dx
                else:
                    dudx = (u[i, j+1] - u[i, j]) / dx
                
                # y-direction advection
                if vy[i, j] > 0:
                    dudy = (u[i, j] - u[i-1, j]) / dy
                else:
                    dudy = (u[i+1, j] - u[i, j]) / dy
                
                u_new[i, j] = u[i, j] - dt * (vx[i, j] * dudx + vy[i, j] * dudy)
        
        return u_new
    
    def _diffusion_step(self, u: np.ndarray, D: float, dx: float, dy: float, dt: float) -> np.ndarray:
        """Perform one diffusion step using explicit scheme"""
        u_new = u.copy()
        ny, nx = u.shape
        
        rx = D * dt / (dx**2)
        ry = D * dt / (dy**2)
        
        # Interior points
        u_new[1:-1, 1:-1] = (u[1:-1, 1:-1] + 
                             rx * (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) +
                             ry * (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]))
        
        return u_new
    
    def _implicit_nonlinear_step(self, u: np.ndarray, D_func: Callable, dx: float, dt: float,
                                boundary_conditions: str) -> np.ndarray:
        """Implicit step for nonlinear diffusion using Newton-Raphson"""
        nx = len(u)
        u_new = u.copy()
        
        # Simple explicit approximation for nonlinear case
        for i in range(1, nx - 1):
            D_right = D_func(0.5 * (u[i] + u[i+1]))
            D_left = D_func(0.5 * (u[i-1] + u[i]))
            
            flux_right = D_right * (u[i+1] - u[i]) / dx
            flux_left = D_left * (u[i] - u[i-1]) / dx
            
            u_new[i] = u[i] + dt * (flux_right - flux_left) / dx
        
        # Apply boundary conditions
        if boundary_conditions == 'dirichlet_zero':
            u_new[0] = 0
            u_new[-1] = 0
        elif boundary_conditions == 'neumann_zero':
            u_new[0] = u_new[1]
            u_new[-1] = u_new[-2]
        
        return u_new
    
    def _get_velocity_field(self, velocity: Union[str, Tuple[float, float]], X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate velocity field"""
        if isinstance(velocity, tuple):
            vx = np.full_like(X, velocity[0])
            vy = np.full_like(Y, velocity[1])
        elif velocity == 'uniform':
            vx = np.full_like(X, 0.1)
            vy = np.full_like(Y, 0.0)
        elif velocity == 'rotation':
            # Circular rotation around center
            cx, cy = 0.5, 0.5
            vx = -(Y - cy)
            vy = (X - cx)
        elif velocity == 'shear':
            vx = Y  # Shear flow
            vy = np.zeros_like(Y)
        elif velocity == 'vortex':
            # Double vortex
            vx = np.sin(np.pi * Y) * np.cos(np.pi * X)
            vy = -np.sin(np.pi * X) * np.cos(np.pi * Y)
        else:
            raise ValueError(f"Unknown velocity field: {velocity}")
        
        return vx, vy
    
    def _get_initial_condition_1d(self, condition: Union[str, Callable], x: np.ndarray) -> np.ndarray:
        """Generate 1D initial condition"""
        if callable(condition):
            return condition(x)
        
        if condition == 'gaussian':
            return np.exp(-50 * (x - 0.5)**2)
        elif condition == 'step':
            return np.where((x >= 0.4) & (x <= 0.6), 1.0, 0.0)
        elif condition == 'sine':
            return 0.5 + 0.5 * np.sin(np.pi * x)
        elif condition == 'zero':
            return np.zeros_like(x)
        else:
            raise ValueError(f"Unknown initial condition: {condition}")
    
    def _get_initial_condition_2d(self, condition: Union[str, Callable], X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Generate 2D initial condition"""
        if callable(condition):
            return condition(X, Y)
        
        if condition == 'gaussian_peak':
            return np.exp(-50 * ((X - 0.3)**2 + (Y - 0.3)**2))
        elif condition == 'multiple_peaks':
            return (np.exp(-100 * ((X - 0.2)**2 + (Y - 0.2)**2)) +
                   np.exp(-100 * ((X - 0.8)**2 + (Y - 0.8)**2)) +
                   np.exp(-100 * ((X - 0.2)**2 + (Y - 0.8)**2)))
        elif condition == 'stripe':
            return np.where((Y >= 0.4) & (Y <= 0.6), 1.0, 0.0)
        elif condition == 'circle':
            r = np.sqrt((X - 0.5)**2 + (Y - 0.5)**2)
            return np.where(r <= 0.2, 1.0, 0.0)
        elif condition == 'zero':
            return np.zeros_like(X)
        else:
            raise ValueError(f"Unknown initial condition: {condition}")
    
    def _get_reaction_diffusion_initial(self, condition: str, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate initial conditions for reaction-diffusion system"""
        if condition == 'uniform':
            u = np.ones_like(X)
            v = np.zeros_like(Y)
        elif condition == 'random_spots':
            u = np.ones_like(X)
            v = np.zeros_like(Y)
            
            # Add random spots of v
            np.random.seed(42)
            n_spots = 20
            for _ in range(n_spots):
                cx = np.random.uniform(0.2, 0.8)
                cy = np.random.uniform(0.2, 0.8)
                r = np.random.uniform(0.02, 0.05)
                
                mask = ((X - cx)**2 + (Y - cy)**2) < r**2
                u[mask] = 0.5
                v[mask] = 0.25
        elif condition == 'single_spot':
            u = np.ones_like(X)
            v = np.zeros_like(Y)
            
            # Single central spot
            mask = ((X - 0.5)**2 + (Y - 0.5)**2) < 0.05**2
            u[mask] = 0.5
            v[mask] = 0.25
        else:
            raise ValueError(f"Unknown reaction-diffusion initial condition: {condition}")
        
        return u, v
    
    def _apply_boundary_conditions_1d(self, u: np.ndarray, n: int, bc_type: str):
        """Apply boundary conditions for 1D diffusion"""
        if bc_type == 'dirichlet_zero':
            u[n, 0] = 0
            u[n, -1] = 0
        elif bc_type == 'neumann_zero':
            u[n, 0] = u[n, 1]
            u[n, -1] = u[n, -2]
        elif bc_type == 'periodic':
            u[n, 0] = u[n, -2]
            u[n, -1] = u[n, 1]
    
    def _apply_boundary_conditions_2d(self, u: np.ndarray, n: int, bc_type: str):
        """Apply boundary conditions for 2D diffusion"""
        if bc_type == 'dirichlet_zero':
            u[n, 0, :] = 0
            u[n, -1, :] = 0
            u[n, :, 0] = 0
            u[n, :, -1] = 0
        elif bc_type == 'neumann_zero':
            u[n, 0, :] = u[n, 1, :]
            u[n, -1, :] = u[n, -2, :]
            u[n, :, 0] = u[n, :, 1]
            u[n, :, -1] = u[n, :, -2]
        elif bc_type == 'periodic':
            u[n, 0, :] = u[n, -2, :]
            u[n, -1, :] = u[n, 1, :]
            u[n, :, 0] = u[n, :, -2]
            u[n, :, -1] = u[n, :, 1]
    
    def _apply_boundary_conditions_2d_species(self, u: np.ndarray, v: np.ndarray, n: int, bc_type: str):
        """Apply boundary conditions for both species in reaction-diffusion"""
        if bc_type == 'neumann_zero':
            # No-flux boundaries
            u[n, 0, :] = u[n, 1, :]
            u[n, -1, :] = u[n, -2, :]
            u[n, :, 0] = u[n, :, 1]
            u[n, :, -1] = u[n, :, -2]
            
            v[n, 0, :] = v[n, 1, :]
            v[n, -1, :] = v[n, -2, :]
            v[n, :, 0] = v[n, :, 1]
            v[n, :, -1] = v[n, :, -2]
        elif bc_type == 'periodic':
            u[n, 0, :] = u[n, -2, :]
            u[n, -1, :] = u[n, 1, :]
            u[n, :, 0] = u[n, :, -2]
            u[n, :, -1] = u[n, :, 1]
            
            v[n, 0, :] = v[n, -2, :]
            v[n, -1, :] = v[n, 1, :]
            v[n, :, 0] = v[n, :, -2]
            v[n, :, -1] = v[n, :, 1]