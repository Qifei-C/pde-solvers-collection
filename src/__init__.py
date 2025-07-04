"""
PDE Solvers Collection
Comprehensive numerical methods for solving partial differential equations
"""

from .pde_solver import PDESolver
from .heat_equation import HeatEquationSolver
from .wave_equation import WaveEquationSolver
from .diffusion import DiffusionSolver

__version__ = "1.0.0"
__author__ = "PDE Solvers Team"

__all__ = [
    'PDESolver',
    'HeatEquationSolver',
    'WaveEquationSolver',
    'DiffusionSolver'
]