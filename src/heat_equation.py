"""
Heat Equation Solvers
Numerical methods for solving the heat equation in 1D, 2D, and 3D
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from typing import Callable, Optional, Tuple, Dict, Union, List
import warnings
warnings.filterwarnings('ignore')


class HeatEquationSolver:
    """
    Heat equation solver with multiple numerical schemes
    """
    
    def __init__(self, method: str = 'finite_difference'):
        """
        Initialize heat equation solver
        
        Args:
            method: Numerical method ('finite_difference', 'finite_element', 'spectral')
        """
        self.method = method
        
        # Material properties database
        self.materials = {
            'steel': {'thermal_diffusivity': 4.2e-6, 'conductivity': 50.0},
            'aluminum': {'thermal_diffusivity': 9.7e-5, 'conductivity': 237.0},
            'copper': {'thermal_diffusivity': 1.1e-4, 'conductivity': 401.0},
            'concrete': {'thermal_diffusivity': 5.0e-7, 'conductivity': 1.7},
            'water': {'thermal_diffusivity': 1.4e-7, 'conductivity': 0.6}
        }
    
    def solve_1d(self, 
                 length: float = 1.0,
                 nx: int = 100,
                 nt: int = 1000,
                 alpha: float = 0.01,
                 initial_condition: Union[str, Callable] = 'gaussian',
                 boundary_conditions: str = 'dirichlet_zero',
                 end_time: float = 1.0,
                 scheme: str = 'explicit') -> Dict:
        """
        Solve 1D heat equation: ∂u/∂t = α∂²u/∂x²
        
        Args:
            length: Domain length
            nx: Number of spatial grid points
            nt: Number of time steps
            alpha: Thermal diffusivity
            initial_condition: Initial condition function or preset
            boundary_conditions: Boundary condition type
            end_time: Final time
            scheme: Time integration scheme ('explicit', 'implicit', 'crank_nicolson')
            
        Returns:
            Dictionary containing solution data
        """
        # Create spatial grid
        x = np.linspace(0, length, nx)
        dx = x[1] - x[0]
        
        # Create time grid
        dt = end_time / nt
        times = np.linspace(0, end_time, nt + 1)
        
        # Check stability for explicit scheme
        r = alpha * dt / (dx**2)
        if scheme == 'explicit' and r > 0.5:
            warnings.warn(f"Explicit scheme may be unstable: r = {r:.3f} > 0.5")
        
        # Set initial condition
        u0 = self._get_initial_condition_1d(initial_condition, x)
        
        # Allocate solution array
        u = np.zeros((nt + 1, nx))
        u[0] = u0
        
        # Time stepping
        if scheme == 'explicit':
            u = self._solve_1d_explicit(u, alpha, dx, dt, boundary_conditions)
        elif scheme == 'implicit':
            u = self._solve_1d_implicit(u, alpha, dx, dt, boundary_conditions)
        elif scheme == 'crank_nicolson':
            u = self._solve_1d_crank_nicolson(u, alpha, dx, dt, boundary_conditions)
        else:
            raise ValueError(f"Unknown scheme: {scheme}")
        
        return {
            'solution': u,
            'grid': {'x': x},
            'times': times,
            'parameters': {
                'alpha': alpha,
                'dx': dx,
                'dt': dt,
                'r_parameter': r,
                'scheme': scheme
            }
        }
    
    def solve_2d(self,
                 domain_size: Tuple[float, float] = (1.0, 1.0),
                 grid_points: Tuple[int, int] = (50, 50),
                 time_steps: int = 1000,
                 thermal_diffusivity: float = 0.01,
                 initial_condition: Union[str, Callable] = 'gaussian_peak',
                 boundary_conditions: str = 'dirichlet_zero',
                 end_time: float = 1.0,
                 scheme: str = 'explicit') -> Dict:
        """
        Solve 2D heat equation: ∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²)
        
        Args:
            domain_size: (Lx, Ly) domain dimensions
            grid_points: (nx, ny) grid resolution
            time_steps: Number of time steps
            thermal_diffusivity: Thermal diffusivity coefficient
            initial_condition: Initial condition
            boundary_conditions: Boundary condition type
            end_time: Final time
            scheme: Time integration scheme
            
        Returns:
            Dictionary containing solution data
        """
        Lx, Ly = domain_size
        nx, ny = grid_points
        nt = time_steps
        alpha = thermal_diffusivity
        
        # Create spatial grid
        x = np.linspace(0, Lx, nx)
        y = np.linspace(0, Ly, ny)
        X, Y = np.meshgrid(x, y)
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        
        # Create time grid
        dt = end_time / nt
        times = np.linspace(0, end_time, nt + 1)
        
        # Check stability
        r = alpha * dt * (1/dx**2 + 1/dy**2)
        if scheme == 'explicit' and r > 0.5:
            warnings.warn(f"Explicit scheme may be unstable: r = {r:.3f} > 0.5")
        
        # Set initial condition
        u0 = self._get_initial_condition_2d(initial_condition, X, Y)
        
        # Allocate solution array
        u = np.zeros((nt + 1, ny, nx))
        u[0] = u0
        
        # Time stepping
        if scheme == 'explicit':
            u = self._solve_2d_explicit(u, alpha, dx, dy, dt, boundary_conditions)
        elif scheme == 'implicit':
            u = self._solve_2d_implicit(u, alpha, dx, dy, dt, boundary_conditions)
        else:
            raise ValueError(f"Unknown scheme: {scheme}")
        
        return {
            'solution': u,
            'grid': {'x': x, 'y': y, 'X': X, 'Y': Y},
            'times': times,
            'parameters': {
                'alpha': alpha,
                'dx': dx,
                'dy': dy,
                'dt': dt,
                'r_parameter': r,
                'scheme': scheme
            }
        }
    
    def solve_composite(self,
                       materials: List[str],
                       interface_position: float = 0.5,
                       heat_source: Optional[Callable] = None,
                       domain_size: float = 1.0,
                       nx: int = 100,
                       nt: int = 1000,
                       end_time: float = 1.0) -> Dict:
        """
        Solve heat transfer in composite materials
        
        Args:
            materials: List of material names
            interface_position: Position of interface (0-1)
            heat_source: Heat source function f(x, t)
            domain_size: Domain length
            nx: Number of grid points
            nt: Number of time steps
            end_time: Final time
            
        Returns:
            Dictionary containing solution data
        """
        if len(materials) != 2:
            raise ValueError("Currently supports only two materials")
        
        # Get material properties
        mat1_props = self.materials[materials[0]]
        mat2_props = self.materials[materials[1]]
        
        # Create spatial grid
        x = np.linspace(0, domain_size, nx)
        dx = x[1] - x[0]
        
        # Interface index
        interface_idx = int(interface_position * nx)
        
        # Create material property arrays
        alpha = np.ones(nx)
        alpha[:interface_idx] = mat1_props['thermal_diffusivity']
        alpha[interface_idx:] = mat2_props['thermal_diffusivity']
        
        k = np.ones(nx)
        k[:interface_idx] = mat1_props['conductivity']
        k[interface_idx:] = mat2_props['conductivity']
        
        # Time grid
        dt = end_time / nt
        times = np.linspace(0, end_time, nt + 1)
        
        # Initial condition (room temperature)
        u = np.zeros((nt + 1, nx))
        u[0] = 20.0  # 20°C initial temperature
        
        # Time stepping with variable properties
        for n in range(nt):
            u_new = u[n].copy()
            
            # Interior points with variable thermal diffusivity
            for i in range(1, nx - 1):
                # Use harmonic mean for thermal conductivity at interfaces
                k_left = 2 * k[i-1] * k[i] / (k[i-1] + k[i]) if i > 0 else k[i]
                k_right = 2 * k[i] * k[i+1] / (k[i] + k[i+1]) if i < nx-1 else k[i]
                
                # Heat equation with variable properties
                laplacian = (k_right * (u[n, i+1] - u[n, i]) - k_left * (u[n, i] - u[n, i-1])) / dx**2
                
                # Add heat source if provided
                source = 0.0
                if heat_source is not None:
                    source = heat_source(x[i], times[n])
                
                u_new[i] = u[n, i] + dt * (alpha[i] * laplacian + source)
            
            # Boundary conditions (insulated)
            u_new[0] = u_new[1]
            u_new[-1] = u_new[-2]
            
            u[n + 1] = u_new
        
        return {
            'solution': u,
            'grid': {'x': x},
            'times': times,
            'materials': materials,
            'interface_position': interface_position,
            'material_properties': {
                materials[0]: mat1_props,
                materials[1]: mat2_props
            }
        }
    
    def _solve_1d_explicit(self, u: np.ndarray, alpha: float, dx: float, dt: float,
                          boundary_conditions: str) -> np.ndarray:
        """Explicit finite difference scheme for 1D heat equation"""
        nt, nx = u.shape
        r = alpha * dt / (dx**2)
        
        for n in range(nt - 1):
            # Interior points
            u[n + 1, 1:-1] = u[n, 1:-1] + r * (u[n, 2:] - 2*u[n, 1:-1] + u[n, :-2])
            
            # Apply boundary conditions
            self._apply_boundary_conditions_1d(u, n + 1, boundary_conditions)
        
        return u
    
    def _solve_1d_implicit(self, u: np.ndarray, alpha: float, dx: float, dt: float,
                          boundary_conditions: str) -> np.ndarray:
        """Implicit finite difference scheme for 1D heat equation"""
        nt, nx = u.shape
        r = alpha * dt / (dx**2)
        
        # Build tridiagonal matrix
        main_diag = 1 + 2*r
        off_diag = -r
        
        A = sp.diags([off_diag, main_diag, off_diag], [-1, 0, 1], 
                     shape=(nx, nx), format='csc')
        
        # Modify for boundary conditions
        if boundary_conditions == 'dirichlet_zero':
            A[0, 0] = 1
            A[0, 1] = 0
            A[-1, -1] = 1
            A[-1, -2] = 0
        elif boundary_conditions == 'neumann_zero':
            A[0, 0] = 1 + r
            A[0, 1] = -r
            A[-1, -1] = 1 + r
            A[-1, -2] = -r
        
        for n in range(nt - 1):
            b = u[n].copy()
            if boundary_conditions == 'dirichlet_zero':
                b[0] = 0
                b[-1] = 0
            
            u[n + 1] = spsolve(A, b)
        
        return u
    
    def _solve_1d_crank_nicolson(self, u: np.ndarray, alpha: float, dx: float, dt: float,
                                boundary_conditions: str) -> np.ndarray:
        """Crank-Nicolson scheme for 1D heat equation"""
        nt, nx = u.shape
        r = alpha * dt / (dx**2)
        
        # Build matrices
        A = sp.diags([r/2, -(1 + r), r/2], [-1, 0, 1], shape=(nx, nx), format='csc')
        B = sp.diags([-r/2, 1 - r, -r/2], [-1, 0, 1], shape=(nx, nx), format='csc')
        
        # Modify for boundary conditions
        if boundary_conditions == 'dirichlet_zero':
            A[0, :] = 0; A[0, 0] = 1
            A[-1, :] = 0; A[-1, -1] = 1
            B[0, :] = 0; B[0, 0] = 1
            B[-1, :] = 0; B[-1, -1] = 1
        
        for n in range(nt - 1):
            rhs = B @ u[n]
            if boundary_conditions == 'dirichlet_zero':
                rhs[0] = 0
                rhs[-1] = 0
            
            u[n + 1] = spsolve(-A, rhs)
        
        return u
    
    def _solve_2d_explicit(self, u: np.ndarray, alpha: float, dx: float, dy: float,
                          dt: float, boundary_conditions: str) -> np.ndarray:
        """Explicit finite difference scheme for 2D heat equation"""
        nt, ny, nx = u.shape
        rx = alpha * dt / (dx**2)
        ry = alpha * dt / (dy**2)
        
        for n in range(nt - 1):
            # Interior points
            u[n + 1, 1:-1, 1:-1] = (u[n, 1:-1, 1:-1] + 
                                    rx * (u[n, 1:-1, 2:] - 2*u[n, 1:-1, 1:-1] + u[n, 1:-1, :-2]) +
                                    ry * (u[n, 2:, 1:-1] - 2*u[n, 1:-1, 1:-1] + u[n, :-2, 1:-1]))
            
            # Apply boundary conditions
            self._apply_boundary_conditions_2d(u, n + 1, boundary_conditions)
        
        return u
    
    def _solve_2d_implicit(self, u: np.ndarray, alpha: float, dx: float, dy: float,
                          dt: float, boundary_conditions: str) -> np.ndarray:
        """Implicit finite difference scheme for 2D heat equation using ADI"""
        nt, ny, nx = u.shape
        rx = alpha * dt / (2 * dx**2)
        ry = alpha * dt / (2 * dy**2)
        
        # Alternating Direction Implicit (ADI) method
        u_half = np.zeros((ny, nx))
        
        for n in range(nt - 1):
            # Step 1: Implicit in x-direction
            for j in range(ny):
                # Build tridiagonal system for each row
                A = sp.diags([-rx, 1 + 2*rx, -rx], [-1, 0, 1], 
                            shape=(nx, nx), format='csc')
                
                # Right-hand side
                b = u[n, j].copy()
                if j > 0 and j < ny - 1:
                    b[1:-1] += ry * (u[n, j+1, 1:-1] - 2*u[n, j, 1:-1] + u[n, j-1, 1:-1])
                
                # Apply boundary conditions
                if boundary_conditions == 'dirichlet_zero':
                    A[0, 0] = 1; A[0, 1] = 0; b[0] = 0
                    A[-1, -1] = 1; A[-1, -2] = 0; b[-1] = 0
                
                u_half[j] = spsolve(A, b)
            
            # Step 2: Implicit in y-direction
            for i in range(nx):
                # Build tridiagonal system for each column
                A = sp.diags([-ry, 1 + 2*ry, -ry], [-1, 0, 1], 
                            shape=(ny, ny), format='csc')
                
                # Right-hand side
                b = u_half[:, i].copy()
                if i > 0 and i < nx - 1:
                    b[1:-1] += rx * (u_half[1:-1, i+1] - 2*u_half[1:-1, i] + u_half[1:-1, i-1])
                
                # Apply boundary conditions
                if boundary_conditions == 'dirichlet_zero':
                    A[0, 0] = 1; A[0, 1] = 0; b[0] = 0
                    A[-1, -1] = 1; A[-1, -2] = 0; b[-1] = 0
                
                u[n + 1, :, i] = spsolve(A, b)
        
        return u
    
    def _get_initial_condition_1d(self, condition: Union[str, Callable], x: np.ndarray) -> np.ndarray:
        """Generate 1D initial condition"""
        if callable(condition):
            return condition(x)
        
        if condition == 'gaussian':
            return np.exp(-50 * (x - 0.5)**2)
        elif condition == 'step':
            return np.where((x >= 0.4) & (x <= 0.6), 1.0, 0.0)
        elif condition == 'sine':
            return np.sin(np.pi * x)
        elif condition == 'zero':
            return np.zeros_like(x)
        else:
            raise ValueError(f"Unknown initial condition: {condition}")
    
    def _get_initial_condition_2d(self, condition: Union[str, Callable], X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Generate 2D initial condition"""
        if callable(condition):
            return condition(X, Y)
        
        if condition == 'gaussian_peak':
            return np.exp(-50 * ((X - 0.5)**2 + (Y - 0.5)**2))
        elif condition == 'four_peaks':
            return (np.exp(-100 * ((X - 0.25)**2 + (Y - 0.25)**2)) +
                   np.exp(-100 * ((X - 0.75)**2 + (Y - 0.25)**2)) +
                   np.exp(-100 * ((X - 0.25)**2 + (Y - 0.75)**2)) +
                   np.exp(-100 * ((X - 0.75)**2 + (Y - 0.75)**2)))
        elif condition == 'central_square':
            return np.where((np.abs(X - 0.5) < 0.2) & (np.abs(Y - 0.5) < 0.2), 1.0, 0.0)
        elif condition == 'sine_wave':
            return np.sin(np.pi * X) * np.sin(np.pi * Y)
        elif condition == 'zero':
            return np.zeros_like(X)
        else:
            raise ValueError(f"Unknown initial condition: {condition}")
    
    def _apply_boundary_conditions_1d(self, u: np.ndarray, n: int, bc_type: str):
        """Apply boundary conditions for 1D problem"""
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
        """Apply boundary conditions for 2D problem"""
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