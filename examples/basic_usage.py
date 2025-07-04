"""
Basic Usage Examples for PDE Solvers Collection
Simple examples to get started with the library
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from src.pde_solver import PDESolver


def basic_heat_equation():
    """Basic heat equation example"""
    print("=== Basic Heat Equation Example ===")
    
    # Create solver
    solver = PDESolver()
    
    # Solve 2D heat equation
    result = solver.solve_heat_equation_2d(
        domain_size=(1.0, 1.0),
        grid_points=(50, 50),
        time_steps=1000,
        thermal_diffusivity=0.01,
        initial_condition='gaussian_peak',
        boundary_conditions='dirichlet_zero',
        end_time=1.0
    )
    
    # Plot final solution
    solver.plot_solution_2d(result, title="Heat Diffusion")
    
    # Create animation
    print("Creating animation...")
    anim = solver.animate_solution(result, save_path='heat_diffusion.gif')
    
    print("Heat equation solved successfully!")


def basic_wave_equation():
    """Basic wave equation example"""
    print("=== Basic Wave Equation Example ===")
    
    # Create solver
    solver = PDESolver()
    
    # Solve 1D wave equation
    result = solver.solve_wave_equation_1d(
        length=1.0,
        nx=100,
        nt=500,
        c=1.0,
        initial_displacement='gaussian',
        boundary_conditions='dirichlet_zero',
        end_time=2.0
    )
    
    # Plot solution at different times
    solver.plot_solution_1d(result, title="Wave Propagation")
    
    print("Wave equation solved successfully!")


def basic_diffusion_advection():
    """Basic advection-diffusion example"""
    print("=== Basic Advection-Diffusion Example ===")
    
    # Create solver
    solver = PDESolver()
    
    # Solve advection-diffusion equation
    result = solver.solve_diffusion_advection(
        domain_size=(1.0, 1.0),
        grid_points=(50, 50),
        time_steps=1000,
        diffusion_coeff=0.005,
        velocity_field=(0.1, 0.05),  # Constant velocity
        initial_condition='gaussian_peak',
        boundary_conditions='periodic',
        end_time=5.0
    )
    
    # Plot evolution
    times_to_plot = [0, len(result['times'])//4, len(result['times'])//2, -1]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    X = result['grid']['X']
    Y = result['grid']['Y']
    
    for i, time_idx in enumerate(times_to_plot):
        im = axes[i].contourf(X, Y, result['solution'][time_idx], levels=15, cmap='viridis')
        axes[i].set_title(f'Time: {result["times"][time_idx]:.2f}')
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('y')
        plt.colorbar(im, ax=axes[i])
        
        # Add velocity arrows
        vx = result['velocity_field']['vx']
        vy = result['velocity_field']['vy']
        step = 5
        axes[i].quiver(X[::step, ::step], Y[::step, ::step], 
                      vx[::step, ::step], vy[::step, ::step], 
                      alpha=0.7, color='white', scale=2)
    
    plt.suptitle('Advection-Diffusion Evolution')
    plt.tight_layout()
    plt.show()
    
    print("Advection-diffusion solved successfully!")


def stability_analysis_demo():
    """Demonstrate stability analysis"""
    print("=== Stability Analysis Demo ===")
    
    solver = PDESolver()
    
    # Test different grid resolutions
    dx_values = [0.1, 0.05, 0.02, 0.01]
    alpha = 0.01  # thermal diffusivity
    c = 1.0       # wave speed
    
    print("Grid Resolution Analysis:")
    print("dx    | Heat Eq. Max dt | Wave Eq. Max dt")
    print("-" * 40)
    
    for dx in dx_values:
        # Fixed dt for analysis
        dt = 1e-4
        
        analysis = solver.analyze_stability(dx, dt, 
                                          diffusion_coeff=alpha, 
                                          wave_speed=c)
        
        print(f"{dx:5.3f} | {analysis['heat_max_dt']:13.1e} | {analysis['wave_max_dt']:13.1e}")
    
    print("\nStability Conditions:")
    print("- Heat equation (explicit): r = α*dt/dx² ≤ 0.5")
    print("- Wave equation: CFL = c*dt/dx ≤ 1.0")


def method_comparison():
    """Compare different numerical methods"""
    print("=== Method Comparison ===")
    
    methods = ['finite_difference']  # Add more methods when implemented
    
    for method in methods:
        print(f"\nMethod: {method}")
        solver = PDESolver(method=method)
        info = solver.get_method_info()
        
        print(f"Description: {info['description']}")
        print(f"Advantages: {info['advantages']}")
        print(f"Limitations: {info['limitations']}")


def main():
    """Run all basic examples"""
    print("PDE Solvers Collection - Basic Usage Examples\n")
    
    try:
        # Run examples
        basic_heat_equation()
        print()
        
        basic_wave_equation()
        print()
        
        basic_diffusion_advection()
        print()
        
        stability_analysis_demo()
        print()
        
        method_comparison()
        
        print("\n=== All Examples Completed Successfully! ===")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()