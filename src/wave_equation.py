"""
Wave Equation Solvers
Numerical methods for solving the wave equation in 1D, 2D, and circular domains
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.special import jn, jn_zeros
from typing import Callable, Optional, Tuple, Dict, Union
import warnings
warnings.filterwarnings('ignore')


class WaveEquationSolver:
    """
    Wave equation solver with multiple numerical schemes
    """
    
    def __init__(self, method: str = 'finite_difference'):
        """
        Initialize wave equation solver
        
        Args:
            method: Numerical method ('finite_difference', 'finite_element', 'spectral')
        """
        self.method = method
    
    def solve_1d(self,
                 length: float = 1.0,
                 nx: int = 100,
                 nt: int = 1000,
                 c: float = 1.0,
                 initial_displacement: Union[str, Callable] = 'gaussian',
                 initial_velocity: Union[str, Callable] = 'zero',
                 boundary_conditions: str = 'dirichlet_zero',
                 end_time: float = 2.0) -> Dict:
        """
        Solve 1D wave equation: ∂²u/∂t² = c²∂²u/∂x²
        
        Args:
            length: Domain length
            nx: Number of spatial grid points
            nt: Number of time steps
            c: Wave speed
            initial_displacement: Initial displacement u(x,0)
            initial_velocity: Initial velocity ∂u/∂t(x,0)
            boundary_conditions: Boundary condition type
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
        
        # Check CFL condition
        cfl = c * dt / dx
        if cfl > 1.0:
            warnings.warn(f"CFL condition violated: CFL = {cfl:.3f} > 1.0")
        
        # Set initial conditions
        u0 = self._get_initial_displacement_1d(initial_displacement, x)
        v0 = self._get_initial_velocity_1d(initial_velocity, x)
        
        # Allocate solution array
        u = np.zeros((nt + 1, nx))
        u[0] = u0
        
        # First time step using initial velocity
        r = (c * dt / dx)**2
        u[1, 1:-1] = (u0[1:-1] + dt * v0[1:-1] + 
                     0.5 * r * (u0[2:] - 2*u0[1:-1] + u0[:-2]))
        
        # Apply boundary conditions for first step
        self._apply_boundary_conditions_1d(u, 1, boundary_conditions)
        
        # Time stepping using leapfrog scheme
        for n in range(1, nt):
            u[n + 1, 1:-1] = (2*u[n, 1:-1] - u[n-1, 1:-1] + 
                             r * (u[n, 2:] - 2*u[n, 1:-1] + u[n, :-2]))
            
            # Apply boundary conditions
            self._apply_boundary_conditions_1d(u, n + 1, boundary_conditions)
        
        return {
            'solution': u,
            'grid': {'x': x},
            'times': times,
            'parameters': {
                'wave_speed': c,
                'dx': dx,
                'dt': dt,
                'cfl_number': cfl
            }
        }
    
    def solve_2d(self,
                 domain_size: Tuple[float, float] = (1.0, 1.0),
                 grid_points: Tuple[int, int] = (50, 50),
                 time_steps: int = 1000,
                 wave_speed: float = 1.0,
                 initial_displacement: Union[str, Callable] = 'gaussian_peak',
                 initial_velocity: Union[str, Callable] = 'zero',
                 boundary_conditions: str = 'dirichlet_zero',
                 end_time: float = 2.0) -> Dict:
        """
        Solve 2D wave equation: ∂²u/∂t² = c²(∂²u/∂x² + ∂²u/∂y²)
        
        Args:
            domain_size: (Lx, Ly) domain dimensions
            grid_points: (nx, ny) grid resolution
            time_steps: Number of time steps
            wave_speed: Wave propagation speed
            initial_displacement: Initial displacement field
            initial_velocity: Initial velocity field
            boundary_conditions: Boundary condition type
            end_time: Final time
            
        Returns:
            Dictionary containing solution data
        """
        Lx, Ly = domain_size
        nx, ny = grid_points
        nt = time_steps
        c = wave_speed
        
        # Create spatial grid
        x = np.linspace(0, Lx, nx)
        y = np.linspace(0, Ly, ny)
        X, Y = np.meshgrid(x, y)
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        
        # Create time grid
        dt = end_time / nt
        times = np.linspace(0, end_time, nt + 1)
        
        # Check CFL condition
        cfl = c * dt * np.sqrt(1/dx**2 + 1/dy**2)
        if cfl > 1.0:
            warnings.warn(f"2D CFL condition violated: CFL = {cfl:.3f} > 1.0")
        
        # Set initial conditions
        u0 = self._get_initial_displacement_2d(initial_displacement, X, Y)
        v0 = self._get_initial_velocity_2d(initial_velocity, X, Y)
        
        # Allocate solution array
        u = np.zeros((nt + 1, ny, nx))
        u[0] = u0
        
        # First time step
        rx = (c * dt / dx)**2
        ry = (c * dt / dy)**2
        
        u[1, 1:-1, 1:-1] = (u0[1:-1, 1:-1] + dt * v0[1:-1, 1:-1] + 
                           0.5 * rx * (u0[1:-1, 2:] - 2*u0[1:-1, 1:-1] + u0[1:-1, :-2]) +
                           0.5 * ry * (u0[2:, 1:-1] - 2*u0[1:-1, 1:-1] + u0[:-2, 1:-1]))
        
        # Apply boundary conditions for first step
        self._apply_boundary_conditions_2d(u, 1, boundary_conditions)
        
        # Time stepping using leapfrog scheme
        for n in range(1, nt):
            u[n + 1, 1:-1, 1:-1] = (2*u[n, 1:-1, 1:-1] - u[n-1, 1:-1, 1:-1] +
                                   rx * (u[n, 1:-1, 2:] - 2*u[n, 1:-1, 1:-1] + u[n, 1:-1, :-2]) +
                                   ry * (u[n, 2:, 1:-1] - 2*u[n, 1:-1, 1:-1] + u[n, :-2, 1:-1]))
            
            # Apply boundary conditions
            self._apply_boundary_conditions_2d(u, n + 1, boundary_conditions)
        
        return {
            'solution': u,
            'grid': {'x': x, 'y': y, 'X': X, 'Y': Y},
            'times': times,
            'parameters': {
                'wave_speed': c,
                'dx': dx,
                'dy': dy,
                'dt': dt,
                'cfl_number': cfl
            }
        }
    
    def solve_circular_membrane(self,
                               radius: float = 1.0,
                               nr: int = 50,
                               nt: int = 1000,
                               wave_speed: float = 1.0,
                               initial_displacement: str = 'bessel_mode',
                               damping_coefficient: float = 0.0,
                               end_time: float = 2.0) -> Dict:
        """
        Solve wave equation on circular membrane (drumhead)
        
        Args:
            radius: Drumhead radius
            nr: Number of radial grid points
            nt: Number of time steps
            wave_speed: Wave speed in membrane
            initial_displacement: Initial displacement pattern
            damping_coefficient: Damping coefficient
            end_time: Final time
            
        Returns:
            Dictionary containing solution data
        """
        # Create radial grid
        r = np.linspace(0, radius, nr)
        dr = r[1] - r[0]
        
        # Create time grid
        dt = end_time / nt
        times = np.linspace(0, end_time, nt + 1)
        
        # Check stability
        cfl = wave_speed * dt / dr
        if cfl > 1.0:
            warnings.warn(f"CFL condition violated: CFL = {cfl:.3f} > 1.0")
        
        # Set initial displacement based on Bessel function modes
        u0 = self._get_bessel_mode(initial_displacement, r, radius)
        v0 = np.zeros_like(u0)  # Zero initial velocity
        
        # Allocate solution array
        u = np.zeros((nt + 1, nr))
        u[0] = u0
        
        # First time step
        alpha = (wave_speed * dt / dr)**2
        beta = damping_coefficient * dt
        
        # Special treatment for r=0 (use L'Hopital's rule)
        u[1, 0] = (1 - 2*alpha - beta) * u0[0] + 2*alpha * u0[1] + dt * v0[0]
        
        # Regular points
        for i in range(1, nr - 1):
            laplacian = (u0[i+1] - 2*u0[i] + u0[i-1]) / dr**2 + (u0[i+1] - u0[i-1]) / (2*r[i]*dr)
            u[1, i] = u0[i] + dt * v0[i] + 0.5 * dt**2 * wave_speed**2 * laplacian - 0.5 * beta * dt * v0[i]
        
        # Boundary condition at r = radius (fixed)
        u[1, -1] = 0
        
        # Time stepping
        for n in range(1, nt):
            # Center point (r = 0)
            u[n + 1, 0] = (2*(1 - 2*alpha) * u[n, 0] + 4*alpha * u[n, 1] - 
                          (1 - beta) * u[n-1, 0]) / (1 + beta)
            
            # Interior points
            for i in range(1, nr - 1):
                laplacian = ((u[n, i+1] - 2*u[n, i] + u[n, i-1]) / dr**2 + 
                           (u[n, i+1] - u[n, i-1]) / (2*r[i]*dr))
                
                u[n + 1, i] = (2*u[n, i] - (1 - beta)*u[n-1, i] + 
                              alpha * dr**2 * wave_speed**2 * laplacian) / (1 + beta)
            
            # Boundary condition
            u[n + 1, -1] = 0
        
        # Convert to 2D for visualization
        theta = np.linspace(0, 2*np.pi, 100)
        R, Theta = np.meshgrid(r, theta)
        X = R * np.cos(Theta)
        Y = R * np.sin(Theta)
        
        # Expand solution to 2D (assuming axisymmetric)
        u_2d = np.zeros((nt + 1, len(theta), nr))
        for n in range(nt + 1):
            for i in range(len(theta)):
                u_2d[n, i, :] = u[n, :]
        
        return {
            'solution': u,
            'solution_2d': u_2d,
            'grid': {'r': r, 'theta': theta, 'R': R, 'Theta': Theta, 'X': X, 'Y': Y},
            'times': times,
            'parameters': {
                'wave_speed': wave_speed,
                'dr': dr,
                'dt': dt,
                'cfl_number': cfl,
                'damping': damping_coefficient
            }
        }
    
    def solve_standing_wave_analysis(self,
                                    length: float = 1.0,
                                    nx: int = 100,
                                    wave_speed: float = 1.0,
                                    frequencies: Optional[np.ndarray] = None) -> Dict:
        """
        Analyze standing wave modes
        
        Args:
            length: Domain length
            nx: Number of spatial grid points
            wave_speed: Wave speed
            frequencies: Frequencies to analyze (if None, compute first few modes)
            
        Returns:
            Dictionary containing mode analysis
        """
        x = np.linspace(0, length, nx)
        
        if frequencies is None:
            # Compute first 5 modes for fixed-end string
            n_modes = 5
            frequencies = np.array([n * wave_speed / (2 * length) for n in range(1, n_modes + 1)])
        
        modes = {}
        for i, freq in enumerate(frequencies):
            n = i + 1
            # Standing wave solution: u(x,t) = A * sin(nπx/L) * cos(ωt)
            mode_shape = np.sin(n * np.pi * x / length)
            omega = 2 * np.pi * freq
            
            modes[f'mode_{n}'] = {
                'frequency': freq,
                'angular_frequency': omega,
                'mode_shape': mode_shape,
                'wavelength': 2 * length / n,
                'wave_number': n * np.pi / length
            }
        
        return {
            'modes': modes,
            'grid': {'x': x},
            'fundamental_frequency': frequencies[0],
            'harmonics': frequencies[1:] if len(frequencies) > 1 else []
        }
    
    def _get_initial_displacement_1d(self, condition: Union[str, Callable], x: np.ndarray) -> np.ndarray:
        """Generate 1D initial displacement"""
        if callable(condition):
            return condition(x)
        
        if condition == 'gaussian':
            return np.exp(-50 * (x - 0.5)**2)
        elif condition == 'sine':
            return np.sin(np.pi * x)
        elif condition == 'pluck':
            # Triangular pluck at center
            center = len(x) // 2
            u = np.zeros_like(x)
            u[:center] = x[:center] / x[center]
            u[center:] = (x[-1] - x[center:]) / (x[-1] - x[center])
            return u * 0.1
        elif condition == 'step':
            return np.where((x >= 0.4) & (x <= 0.6), 0.1, 0.0)
        elif condition == 'zero':
            return np.zeros_like(x)
        else:
            raise ValueError(f"Unknown initial displacement: {condition}")
    
    def _get_initial_velocity_1d(self, condition: Union[str, Callable], x: np.ndarray) -> np.ndarray:
        """Generate 1D initial velocity"""
        if callable(condition):
            return condition(x)
        
        if condition == 'zero':
            return np.zeros_like(x)
        elif condition == 'gaussian':
            return 0.1 * np.exp(-50 * (x - 0.5)**2)
        elif condition == 'sine':
            return 0.1 * np.sin(np.pi * x)
        else:
            raise ValueError(f"Unknown initial velocity: {condition}")
    
    def _get_initial_displacement_2d(self, condition: Union[str, Callable], X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Generate 2D initial displacement"""
        if callable(condition):
            return condition(X, Y)
        
        if condition == 'gaussian_peak':
            return 0.1 * np.exp(-50 * ((X - 0.5)**2 + (Y - 0.5)**2))
        elif condition == 'sine_wave':
            return 0.1 * np.sin(np.pi * X) * np.sin(np.pi * Y)
        elif condition == 'cross':
            return 0.1 * (np.exp(-100 * (X - 0.5)**2) * np.exp(-10 * (Y - 0.5)**2) +
                         np.exp(-10 * (X - 0.5)**2) * np.exp(-100 * (Y - 0.5)**2))
        elif condition == 'circular_wave':
            r = np.sqrt((X - 0.5)**2 + (Y - 0.5)**2)
            return 0.1 * np.exp(-100 * r**2)
        elif condition == 'zero':
            return np.zeros_like(X)
        else:
            raise ValueError(f"Unknown initial displacement: {condition}")
    
    def _get_initial_velocity_2d(self, condition: Union[str, Callable], X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Generate 2D initial velocity"""
        if callable(condition):
            return condition(X, Y)
        
        if condition == 'zero':
            return np.zeros_like(X)
        elif condition == 'gaussian':
            return 0.1 * np.exp(-50 * ((X - 0.5)**2 + (Y - 0.5)**2))
        else:
            raise ValueError(f"Unknown initial velocity: {condition}")
    
    def _get_bessel_mode(self, mode: str, r: np.ndarray, radius: float) -> np.ndarray:
        """Generate Bessel function mode for circular membrane"""
        if mode == 'bessel_mode':
            # First mode: J_0(2.405 * r/R)
            return jn(0, 2.405 * r / radius)
        elif mode == 'bessel_11':
            # J_1(3.832 * r/R) * cos(θ) mode (but we use axisymmetric)
            return jn(1, 3.832 * r / radius)
        elif mode == 'bessel_21':
            # J_2(5.136 * r/R) mode
            return jn(2, 5.136 * r / radius)
        elif mode == 'gaussian':
            return np.exp(-10 * (r - radius/3)**2)
        else:
            # Default to fundamental mode
            return jn(0, 2.405 * r / radius)
    
    def _apply_boundary_conditions_1d(self, u: np.ndarray, n: int, bc_type: str):
        """Apply boundary conditions for 1D wave equation"""
        if bc_type == 'dirichlet_zero':
            # Fixed ends
            u[n, 0] = 0
            u[n, -1] = 0
        elif bc_type == 'neumann_zero':
            # Free ends
            u[n, 0] = u[n, 1]
            u[n, -1] = u[n, -2]
        elif bc_type == 'periodic':
            # Periodic boundaries
            u[n, 0] = u[n, -2]
            u[n, -1] = u[n, 1]
    
    def _apply_boundary_conditions_2d(self, u: np.ndarray, n: int, bc_type: str):
        """Apply boundary conditions for 2D wave equation"""
        if bc_type == 'dirichlet_zero':
            # Fixed boundaries
            u[n, 0, :] = 0
            u[n, -1, :] = 0
            u[n, :, 0] = 0
            u[n, :, -1] = 0
        elif bc_type == 'neumann_zero':
            # Free boundaries
            u[n, 0, :] = u[n, 1, :]
            u[n, -1, :] = u[n, -2, :]
            u[n, :, 0] = u[n, :, 1]
            u[n, :, -1] = u[n, :, -2]
        elif bc_type == 'periodic':
            # Periodic boundaries
            u[n, 0, :] = u[n, -2, :]
            u[n, -1, :] = u[n, 1, :]
            u[n, :, 0] = u[n, :, -2]
            u[n, :, -1] = u[n, :, 1]