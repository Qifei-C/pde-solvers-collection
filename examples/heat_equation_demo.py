"""
Heat Equation Demo
Demonstrates solving and visualizing heat equation solutions
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from src.pde_solver import PDESolver


def main():
    """Run heat equation demonstrations"""
    print("=== Heat Equation Demonstration ===\n")
    
    # Initialize solver
    solver = PDESolver(method='finite_difference')
    
    # Demo 1: 1D Heat Equation
    print("1. Solving 1D Heat Equation...")
    result_1d = solver.solve_heat_equation_1d(
        length=1.0,
        nx=100,
        nt=500,
        alpha=0.01,
        initial_condition='gaussian',
        boundary_conditions='dirichlet_zero',
        end_time=1.0
    )
    
    # Plot 1D results
    solver.plot_solution_1d(result_1d, title="1D Heat Equation Solution")
    
    # Demo 2: 2D Heat Equation
    print("2. Solving 2D Heat Equation...")
    result_2d = solver.solve_heat_equation_2d(
        domain_size=(1.0, 1.0),
        grid_points=(50, 50),
        time_steps=1000,
        thermal_diffusivity=0.01,
        initial_condition='gaussian_peak',
        boundary_conditions='dirichlet_zero',
        end_time=0.5
    )
    
    # Plot 2D results
    solver.plot_solution_2d(result_2d, title="2D Heat Equation Solution")
    
    # Demo 3: Composite Material Heat Transfer
    print("3. Solving Composite Material Heat Transfer...")
    
    # Define heat source
    def heat_source(x, t):
        return 1000 * np.exp(-((x - 0.2)**2) / 0.01) * np.sin(2 * np.pi * t)
    
    result_composite = solver.solve_composite_heat_transfer(
        materials=['steel', 'aluminum'],
        interface_position=0.5,
        heat_source=heat_source,
        domain_size=1.0,
        nx=100,
        nt=1000,
        end_time=2.0
    )
    
    # Plot composite results
    x = result_composite['grid']['x']
    u = result_composite['solution']
    times = result_composite['times']
    
    plt.figure(figsize=(12, 8))
    
    # Temperature evolution
    plt.subplot(2, 2, 1)
    time_indices = [0, len(times)//4, len(times)//2, 3*len(times)//4, -1]
    for idx in time_indices:
        plt.plot(x, u[idx], label=f't = {times[idx]:.2f}s')
    plt.axvline(x=0.5, color='r', linestyle='--', alpha=0.7, label='Interface')
    plt.xlabel('Position (m)')
    plt.ylabel('Temperature (°C)')
    plt.title('Composite Material Heat Transfer')
    plt.legend()
    plt.grid(True)
    
    # Temperature contour
    plt.subplot(2, 2, 2)
    T, X = np.meshgrid(times, x)
    plt.contourf(T, X, u.T, levels=20, cmap='hot')
    plt.colorbar(label='Temperature (°C)')
    plt.axhline(y=0.5, color='white', linestyle='--', alpha=0.7)
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title('Temperature Evolution')
    
    # Material properties visualization
    plt.subplot(2, 2, 3)
    materials = result_composite['materials']
    props = result_composite['material_properties']
    
    alpha_steel = props[materials[0]]['thermal_diffusivity']
    alpha_aluminum = props[materials[1]]['thermal_diffusivity']
    
    alpha_profile = np.ones_like(x)
    interface_idx = int(0.5 * len(x))
    alpha_profile[:interface_idx] = alpha_steel
    alpha_profile[interface_idx:] = alpha_aluminum
    
    plt.plot(x, alpha_profile * 1e6, 'b-', linewidth=2)
    plt.axvline(x=0.5, color='r', linestyle='--', alpha=0.7)
    plt.xlabel('Position (m)')
    plt.ylabel('Thermal Diffusivity (×10⁻⁶ m²/s)')
    plt.title('Material Properties')
    plt.grid(True)
    
    # Final temperature profile
    plt.subplot(2, 2, 4)
    plt.plot(x, u[-1], 'r-', linewidth=2, label='Final temperature')
    plt.axvline(x=0.5, color='k', linestyle='--', alpha=0.7, label='Interface')
    plt.xlabel('Position (m)')
    plt.ylabel('Temperature (°C)')
    plt.title(f'Final Temperature (t = {times[-1]:.2f}s)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Demo 4: Stability Analysis
    print("4. Stability Analysis...")
    dx = 0.01
    alpha = 0.01
    
    # Test different time steps
    dt_values = np.array([1e-5, 5e-5, 1e-4, 5e-4, 1e-3])
    
    print("Time Step (dt) | r = α*dt/dx² | Stable?")
    print("-" * 40)
    for dt in dt_values:
        analysis = solver.analyze_stability(dx, dt, diffusion_coeff=alpha)
        stable = "Yes" if analysis['heat_stable'] else "No"
        print(f"{dt:11.1e} | {analysis['heat_r_parameter']:10.3f} | {stable}")
    
    print(f"\nRecommended max dt: {solver.analyze_stability(dx, 1, diffusion_coeff=alpha)['heat_max_dt']:.1e}")
    
    print("\n=== Heat Equation Demo Complete ===")


if __name__ == "__main__":
    main()