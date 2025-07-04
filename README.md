# PDE Solvers Collection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-orange.svg)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.7+-blue.svg)](https://scipy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Comprehensive collection of numerical methods for solving Partial Differential Equations (PDEs) including heat equations, wave equations, and diffusion problems. Features finite difference, finite element, and spectral methods with professional visualization.

## 🎯 Features

- **Multiple PDE Types**: Heat, wave, diffusion, and advection equations
- **Numerical Methods**: Finite difference, finite element, spectral methods
- **Advanced Schemes**: Implicit, explicit, and Crank-Nicolson methods
- **Boundary Conditions**: Dirichlet, Neumann, and Robin conditions
- **3D Visualization**: Interactive plots and animations

## 🚀 Quick Start

```python
from src.pde_solver import PDESolver

# Solve 2D heat equation
solver = PDESolver()
solution = solver.solve_heat_equation_2d(
    domain_size=(1.0, 1.0),
    grid_points=(100, 100),
    time_steps=1000,
    thermal_diffusivity=0.01,
    initial_condition='gaussian_peak',
    boundary_conditions='dirichlet_zero'
)

# Visualize results
solver.animate_solution(solution, save_path='heat_diffusion.gif')
```

## 🔬 Supported PDEs

### Heat Equation
```
∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²)
```
- 1D, 2D, and 3D implementations
- Variable thermal diffusivity
- Multiple boundary conditions

### Wave Equation
```
∂²u/∂t² = c²(∂²u/∂x² + ∂²u/∂y²)
```
- Acoustic wave propagation
- Vibrating membrane simulation
- Standing wave analysis

### Diffusion-Advection Equation
```
∂u/∂t + v·∇u = D∇²u
```
- Contaminant transport
- Population dynamics
- Reaction-diffusion systems

## 📁 Project Structure

```
pde-solvers-collection/
├── src/                     # Source code
│   ├── pde_solver.py       # Main solver interface
│   ├── heat_equation.py    # Heat equation methods
│   ├── wave_equation.py    # Wave equation methods
│   └── diffusion.py        # Diffusion solvers
├── examples/               # Example problems
├── notebooks/              # Jupyter tutorials
├── tests/                  # Test suite
├── docs/                   # Documentation
└── README.md              # This file
```

## 🛠 Numerical Methods

### Finite Difference Methods
- Forward, backward, and central differences
- Stability analysis and CFL conditions
- High-order accurate schemes

### Finite Element Methods
- Linear and quadratic elements
- Weak form implementations
- Adaptive mesh refinement

### Spectral Methods
- Fourier spectral methods
- Chebyshev polynomials
- Fast transforms for efficiency

## 📊 Example Problems

### Heat Transfer in Materials
```python
# Solve heat conduction in composite material
result = solver.solve_composite_heat_transfer(
    materials=['steel', 'aluminum'],
    interface_position=0.5,
    heat_source=lambda x, t: 100 * np.sin(np.pi * t)
)
```

### Membrane Vibrations
```python
# Simulate drumhead vibrations
vibration = solver.solve_drumhead_vibration(
    radius=1.0,
    initial_displacement='bessel_mode',
    damping_coefficient=0.01
)
```

## 🎨 Visualization Features

- Real-time solution animation
- 3D surface plots with contours
- Cross-sectional analysis
- Convergence diagnostics

## 📈 Performance

- Optimized NumPy operations
- Sparse matrix solvers
- Parallel processing support
- Memory-efficient algorithms

## 🔬 Scientific Applications

- **Engineering**: Heat transfer, structural analysis
- **Physics**: Wave propagation, quantum mechanics
- **Biology**: Population dynamics, pattern formation
- **Finance**: Black-Scholes equation, risk modeling

---

🌊 **Numerical Solutions for Complex Physics**