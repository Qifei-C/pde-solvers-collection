"""
Main PDE Solver Interface
Comprehensive collection of numerical methods for solving Partial Differential Equations
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from typing import Callable, Optional, Tuple, Dict, Union, List
import warnings
warnings.filterwarnings('ignore')

from .heat_equation import HeatEquationSolver
from .wave_equation import WaveEquationSolver
from .diffusion import DiffusionSolver


class PDESolver:
    """
    Main interface for PDE solving with multiple numerical methods
    """
    
    def __init__(self, method: str = 'finite_difference'):
        """
        Initialize PDE solver
        
        Args:
            method: Numerical method ('finite_difference', 'finite_element', 'spectral')
        """
        self.method = method
        self.heat_solver = HeatEquationSolver(method)
        self.wave_solver = WaveEquationSolver(method)
        self.diffusion_solver = DiffusionSolver(method)
        
        # Solution storage
        self.last_solution = None
        self.last_grid = None
        self.last_times = None
        
    def solve_heat_equation_1d(self, 
                              length: float = 1.0,
                              nx: int = 100,
                              nt: int = 1000,
                              alpha: float = 0.01,
                              initial_condition: Union[str, Callable] = 'gaussian',
                              boundary_conditions: str = 'dirichlet_zero',
                              end_time: float = 1.0) -> Dict:
        """
        Solve 1D heat equation
        
        Args:
            length: Domain length
            nx: Number of spatial grid points
            nt: Number of time steps
            alpha: Thermal diffusivity
            initial_condition: Initial condition function or preset
            boundary_conditions: Boundary condition type
            end_time: Final time
            
        Returns:
            Dictionary containing solution data
        """
        return self.heat_solver.solve_1d(
            length, nx, nt, alpha, initial_condition, boundary_conditions, end_time
        )
    
    def solve_heat_equation_2d(self,
                              domain_size: Tuple[float, float] = (1.0, 1.0),
                              grid_points: Tuple[int, int] = (50, 50),
                              time_steps: int = 1000,
                              thermal_diffusivity: float = 0.01,
                              initial_condition: Union[str, Callable] = 'gaussian_peak',
                              boundary_conditions: str = 'dirichlet_zero',
                              end_time: float = 1.0) -> Dict:
        """
        Solve 2D heat equation
        
        Args:
            domain_size: (Lx, Ly) domain dimensions
            grid_points: (nx, ny) grid resolution
            time_steps: Number of time steps
            thermal_diffusivity: Thermal diffusivity coefficient
            initial_condition: Initial condition
            boundary_conditions: Boundary condition type
            end_time: Final time
            
        Returns:
            Dictionary containing solution data
        """
        result = self.heat_solver.solve_2d(
            domain_size, grid_points, time_steps, thermal_diffusivity,
            initial_condition, boundary_conditions, end_time
        )
        
        # Store for visualization
        self.last_solution = result['solution']
        self.last_grid = result['grid']
        self.last_times = result['times']
        
        return result
    
    def solve_wave_equation_1d(self,
                              length: float = 1.0,
                              nx: int = 100,
                              nt: int = 1000,
                              c: float = 1.0,
                              initial_displacement: Union[str, Callable] = 'gaussian',
                              initial_velocity: Union[str, Callable] = 'zero',
                              boundary_conditions: str = 'dirichlet_zero',
                              end_time: float = 2.0) -> Dict:
        """
        Solve 1D wave equation
        
        Args:
            length: Domain length
            nx: Number of spatial grid points
            nt: Number of time steps
            c: Wave speed
            initial_displacement: Initial displacement
            initial_velocity: Initial velocity
            boundary_conditions: Boundary condition type
            end_time: Final time
            
        Returns:
            Dictionary containing solution data
        """
        return self.wave_solver.solve_1d(
            length, nx, nt, c, initial_displacement, initial_velocity,
            boundary_conditions, end_time
        )
    
    def solve_wave_equation_2d(self,
                              domain_size: Tuple[float, float] = (1.0, 1.0),
                              grid_points: Tuple[int, int] = (50, 50),
                              time_steps: int = 1000,
                              wave_speed: float = 1.0,
                              initial_displacement: Union[str, Callable] = 'gaussian_peak',
                              initial_velocity: Union[str, Callable] = 'zero',
                              boundary_conditions: str = 'dirichlet_zero',
                              end_time: float = 2.0) -> Dict:
        """
        Solve 2D wave equation
        
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
        result = self.wave_solver.solve_2d(
            domain_size, grid_points, time_steps, wave_speed,
            initial_displacement, initial_velocity, boundary_conditions, end_time
        )
        
        # Store for visualization
        self.last_solution = result['solution']
        self.last_grid = result['grid']
        self.last_times = result['times']
        
        return result
    
    def solve_diffusion_advection(self,
                                 domain_size: Tuple[float, float] = (1.0, 1.0),
                                 grid_points: Tuple[int, int] = (50, 50),
                                 time_steps: int = 1000,
                                 diffusion_coeff: float = 0.01,
                                 velocity_field: Union[str, Tuple[float, float]] = (0.1, 0.0),
                                 initial_condition: Union[str, Callable] = 'gaussian_peak',
                                 boundary_conditions: str = 'periodic',
                                 end_time: float = 2.0) -> Dict:
        """
        Solve diffusion-advection equation
        
        Args:
            domain_size: (Lx, Ly) domain dimensions
            grid_points: (nx, ny) grid resolution
            time_steps: Number of time steps
            diffusion_coeff: Diffusion coefficient
            velocity_field: Advection velocity (vx, vy)
            initial_condition: Initial concentration
            boundary_conditions: Boundary condition type
            end_time: Final time
            
        Returns:
            Dictionary containing solution data
        """
        result = self.diffusion_solver.solve_advection_diffusion(
            domain_size, grid_points, time_steps, diffusion_coeff,
            velocity_field, initial_condition, boundary_conditions, end_time
        )
        
        # Store for visualization
        self.last_solution = result['solution']
        self.last_grid = result['grid']
        self.last_times = result['times']
        
        return result
    
    def solve_composite_heat_transfer(self,
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
        return self.heat_solver.solve_composite(
            materials, interface_position, heat_source,
            domain_size, nx, nt, end_time
        )
    
    def solve_drumhead_vibration(self,
                                radius: float = 1.0,
                                nr: int = 50,
                                nt: int = 1000,
                                wave_speed: float = 1.0,
                                initial_displacement: str = 'bessel_mode',
                                damping_coefficient: float = 0.0,
                                end_time: float = 2.0) -> Dict:
        """
        Solve circular drumhead vibration
        
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
        return self.wave_solver.solve_circular_membrane(
            radius, nr, nt, wave_speed, initial_displacement,
            damping_coefficient, end_time
        )
    
    def plot_solution_1d(self, solution_data: Dict, 
                        time_indices: Optional[List[int]] = None,
                        title: str = "PDE Solution"):
        """
        Plot 1D solution at different times
        
        Args:
            solution_data: Solution dictionary
            time_indices: Time indices to plot
            title: Plot title
        """
        x = solution_data['grid']['x']
        u = solution_data['solution']
        times = solution_data['times']
        
        if time_indices is None:
            time_indices = [0, len(times)//4, len(times)//2, 3*len(times)//4, -1]
        
        plt.figure(figsize=(10, 6))
        for idx in time_indices:
            plt.plot(x, u[idx], label=f't = {times[idx]:.3f}')
        
        plt.xlabel('x')
        plt.ylabel('u(x,t)')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_solution_2d(self, solution_data: Dict, time_index: int = -1,
                        title: str = "2D PDE Solution"):
        """
        Plot 2D solution at specific time
        
        Args:
            solution_data: Solution dictionary
            time_index: Time index to plot
            title: Plot title
        """
        X = solution_data['grid']['X']
        Y = solution_data['grid']['Y']
        u = solution_data['solution'][time_index]
        t = solution_data['times'][time_index]
        
        fig = plt.figure(figsize=(12, 5))
        
        # Surface plot
        ax1 = fig.add_subplot(121, projection='3d')
        surf = ax1.plot_surface(X, Y, u, cmap='viridis', alpha=0.8)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('u(x,y)')
        ax1.set_title(f'{title} (t = {t:.3f})')
        
        # Contour plot
        ax2 = fig.add_subplot(122)
        contour = ax2.contourf(X, Y, u, levels=20, cmap='viridis')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_title(f'Contour Plot (t = {t:.3f})')
        plt.colorbar(contour, ax=ax2)
        
        plt.tight_layout()
        plt.show()
    
    def animate_solution(self, solution_data: Optional[Dict] = None,
                        save_path: Optional[str] = None,
                        fps: int = 10, interval: int = 50) -> FuncAnimation:
        """
        Create animation of solution evolution
        
        Args:
            solution_data: Solution data (uses last solution if None)
            save_path: Path to save animation
            fps: Frames per second
            interval: Animation interval in ms
            
        Returns:
            Animation object
        """
        if solution_data is None:
            if self.last_solution is None:
                raise ValueError("No solution data available for animation")
            solution = self.last_solution
            grid = self.last_grid
            times = self.last_times
        else:
            solution = solution_data['solution']
            grid = solution_data['grid']
            times = solution_data['times']
        
        if len(solution.shape) == 2:
            # 1D animation
            return self._animate_1d(solution, grid['x'], times, save_path, fps, interval)
        else:
            # 2D animation
            return self._animate_2d(solution, grid, times, save_path, fps, interval)
    
    def _animate_1d(self, solution: np.ndarray, x: np.ndarray, times: np.ndarray,
                   save_path: Optional[str], fps: int, interval: int) -> FuncAnimation:
        """Animate 1D solution"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Set up the plot
        line, = ax.plot([], [], 'b-', linewidth=2)
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(solution.min(), solution.max())
        ax.set_xlabel('x')
        ax.set_ylabel('u(x,t)')
        ax.grid(True)
        
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        
        def animate(frame):
            line.set_data(x, solution[frame])
            time_text.set_text(f'Time: {times[frame]:.3f}')
            return line, time_text
        
        # Create animation
        anim = FuncAnimation(fig, animate, frames=len(times),
                           interval=interval, blit=True, repeat=True)
        
        if save_path:
            anim.save(save_path, fps=fps, writer='pillow')
        
        plt.show()
        return anim
    
    def _animate_2d(self, solution: np.ndarray, grid: Dict, times: np.ndarray,
                   save_path: Optional[str], fps: int, interval: int) -> FuncAnimation:
        """Animate 2D solution"""
        X, Y = grid['X'], grid['Y']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Initial contour plot
        vmin, vmax = solution.min(), solution.max()
        im = ax.contourf(X, Y, solution[0], levels=20, cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        cbar = plt.colorbar(im, ax=ax)
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, 
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        def animate(frame):
            ax.clear()
            im = ax.contourf(X, Y, solution[frame], levels=20, cmap='viridis', vmin=vmin, vmax=vmax)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.text(0.02, 0.95, f'Time: {times[frame]:.3f}', transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            return im.collections
        
        # Create animation
        anim = FuncAnimation(fig, animate, frames=len(times),
                           interval=interval, repeat=True)
        
        if save_path:
            anim.save(save_path, fps=fps, writer='pillow')
        
        plt.show()
        return anim
    
    def analyze_stability(self, dx: float, dt: float, diffusion_coeff: float = None,
                         wave_speed: float = None) -> Dict[str, float]:
        """
        Analyze numerical stability conditions
        
        Args:
            dx: Spatial step size
            dt: Time step size
            diffusion_coeff: Diffusion coefficient (for heat/diffusion equations)
            wave_speed: Wave speed (for wave equations)
            
        Returns:
            Dictionary with stability analysis
        """
        analysis = {}
        
        if diffusion_coeff is not None:
            # Heat equation stability (explicit scheme)
            r = diffusion_coeff * dt / (dx**2)
            analysis['heat_r_parameter'] = r
            analysis['heat_stable'] = r <= 0.5
            analysis['heat_max_dt'] = 0.5 * dx**2 / diffusion_coeff
        
        if wave_speed is not None:
            # Wave equation CFL condition
            cfl = wave_speed * dt / dx
            analysis['wave_cfl_number'] = cfl
            analysis['wave_stable'] = cfl <= 1.0
            analysis['wave_max_dt'] = dx / wave_speed
        
        return analysis
    
    def get_method_info(self) -> Dict[str, str]:
        """
        Get information about current numerical method
        
        Returns:
            Dictionary with method information
        """
        info = {
            'method': self.method,
            'description': self._get_method_description(),
            'advantages': self._get_method_advantages(),
            'limitations': self._get_method_limitations()
        }
        return info
    
    def _get_method_description(self) -> str:
        """Get method description"""
        descriptions = {
            'finite_difference': 'Approximates derivatives using finite differences on a regular grid',
            'finite_element': 'Uses piecewise polynomial basis functions over mesh elements',
            'spectral': 'Expands solution in terms of global basis functions (Fourier, Chebyshev)'
        }
        return descriptions.get(self.method, 'Unknown method')
    
    def _get_method_advantages(self) -> str:
        """Get method advantages"""
        advantages = {
            'finite_difference': 'Simple implementation, efficient for regular grids',
            'finite_element': 'Handles complex geometries, flexible boundary conditions',
            'spectral': 'High accuracy for smooth solutions, fast convergence'
        }
        return advantages.get(self.method, 'Unknown method')
    
    def _get_method_limitations(self) -> str:
        """Get method limitations"""
        limitations = {
            'finite_difference': 'Limited to regular grids, lower order accuracy',
            'finite_element': 'More complex implementation, computational overhead',
            'spectral': 'Requires smooth solutions, difficult for complex geometries'
        }
        return limitations.get(self.method, 'Unknown method')