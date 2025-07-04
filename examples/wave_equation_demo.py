"""
Wave Equation Demo
Demonstrates solving and visualizing wave equation solutions
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from src.pde_solver import PDESolver


def main():
    """Run wave equation demonstrations"""
    print("=== Wave Equation Demonstration ===\n")
    
    # Initialize solver
    solver = PDESolver(method='finite_difference')
    
    # Demo 1: 1D Wave Equation - Plucked String
    print("1. Solving 1D Wave Equation (Plucked String)...")
    result_1d = solver.solve_wave_equation_1d(
        length=1.0,
        nx=100,
        nt=1000,
        c=1.0,
        initial_displacement='pluck',
        initial_velocity='zero',
        boundary_conditions='dirichlet_zero',
        end_time=2.0
    )
    
    # Plot 1D results
    solver.plot_solution_1d(result_1d, title="1D Wave Equation - Plucked String")
    
    # Create animation for 1D wave
    print("Creating 1D wave animation...")
    anim_1d = solver.animate_solution(result_1d)
    
    # Demo 2: 2D Wave Equation - Membrane Vibration
    print("2. Solving 2D Wave Equation (Vibrating Membrane)...")
    result_2d = solver.solve_wave_equation_2d(
        domain_size=(1.0, 1.0),
        grid_points=(50, 50),
        time_steps=500,
        wave_speed=1.0,
        initial_displacement='circular_wave',
        initial_velocity='zero',
        boundary_conditions='dirichlet_zero',
        end_time=1.0
    )
    
    # Plot 2D results
    solver.plot_solution_2d(result_2d, title="2D Wave Equation - Vibrating Membrane")
    
    # Demo 3: Circular Drumhead
    print("3. Solving Circular Drumhead Vibration...")
    result_drum = solver.solve_drumhead_vibration(
        radius=1.0,
        nr=50,
        nt=1000,
        wave_speed=1.0,
        initial_displacement='bessel_mode',
        damping_coefficient=0.02,
        end_time=3.0
    )
    
    # Plot drumhead results
    r = result_drum['grid']['r']
    u = result_drum['solution']
    times = result_drum['times']
    
    plt.figure(figsize=(15, 10))
    
    # Radial displacement evolution
    plt.subplot(2, 3, 1)
    time_indices = [0, len(times)//6, len(times)//3, len(times)//2, 2*len(times)//3, -1]
    for idx in time_indices:
        plt.plot(r, u[idx], label=f't = {times[idx]:.2f}s')
    plt.xlabel('Radius (m)')
    plt.ylabel('Displacement')
    plt.title('Drumhead Displacement vs Radius')
    plt.legend()
    plt.grid(True)
    
    # Time-radius evolution
    plt.subplot(2, 3, 2)
    T, R = np.meshgrid(times, r)
    plt.contourf(T, R, u.T, levels=20, cmap='RdBu_r')
    plt.colorbar(label='Displacement')
    plt.xlabel('Time (s)')
    plt.ylabel('Radius (m)')
    plt.title('Drumhead Evolution')
    
    # 2D drumhead visualization
    ax = plt.subplot(2, 3, 3, projection='3d')
    X = result_drum['grid']['X']
    Y = result_drum['grid']['Y']
    U_2d = result_drum['solution_2d'][-1]  # Final time
    ax.plot_surface(X, Y, U_2d, cmap='viridis', alpha=0.8)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Displacement')
    ax.set_title('Drumhead 3D View')
    
    # Frequency analysis
    plt.subplot(2, 3, 4)
    center_displacement = u[:, 0]  # Displacement at center
    dt = times[1] - times[0]
    freqs = np.fft.fftfreq(len(center_displacement), dt)
    fft_center = np.abs(np.fft.fft(center_displacement))
    
    # Only plot positive frequencies
    pos_freqs = freqs[:len(freqs)//2]
    pos_fft = fft_center[:len(freqs)//2]
    
    plt.plot(pos_freqs, pos_fft)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('Frequency Spectrum at Center')
    plt.grid(True)
    plt.xlim(0, 10)
    
    # Energy evolution
    plt.subplot(2, 3, 5)
    dr = r[1] - r[0]
    energy = []
    for n in range(len(times)):
        # Kinetic + potential energy approximation
        if n > 0:
            kinetic = 0.5 * np.sum(((u[n] - u[n-1])/dt)**2 * r * dr)
            potential = 0.5 * np.sum(u[n]**2 * r * dr)
            total_energy = kinetic + potential
        else:
            total_energy = 0.5 * np.sum(u[n]**2 * r * dr)
        energy.append(total_energy)
    
    plt.plot(times, energy)
    plt.xlabel('Time (s)')
    plt.ylabel('Total Energy')
    plt.title('Energy Evolution (with damping)')
    plt.grid(True)
    
    # Damping effect
    plt.subplot(2, 3, 6)
    max_displacement = np.max(np.abs(u), axis=1)
    plt.semilogy(times, max_displacement)
    plt.xlabel('Time (s)')
    plt.ylabel('Max Displacement (log scale)')
    plt.title('Damping Effect')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Demo 4: Standing Wave Analysis
    print("4. Standing Wave Mode Analysis...")
    modes_analysis = solver.solve_standing_wave_analysis(
        length=1.0,
        nx=200,
        wave_speed=1.0
    )
    
    # Plot standing wave modes
    plt.figure(figsize=(12, 8))
    
    x = modes_analysis['grid']['x']
    modes = modes_analysis['modes']
    
    # Plot first 4 modes
    for i, (mode_name, mode_data) in enumerate(list(modes.items())[:4]):
        plt.subplot(2, 2, i+1)
        plt.plot(x, mode_data['mode_shape'], 'b-', linewidth=2)
        plt.title(f'Mode {i+1}: f = {mode_data["frequency"]:.2f} Hz')
        plt.xlabel('Position (m)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    plt.suptitle('Standing Wave Modes for Fixed-End String')
    plt.tight_layout()
    plt.show()
    
    # Print mode information
    print("\nStanding Wave Mode Analysis:")
    print("Mode | Frequency (Hz) | Wavelength (m) | Wave Number")
    print("-" * 50)
    for i, (mode_name, mode_data) in enumerate(modes.items()):
        print(f"{i+1:4d} | {mode_data['frequency']:12.3f} | {mode_data['wavelength']:12.3f} | {mode_data['wave_number']:10.3f}")
    
    # Demo 5: Wave Interference
    print("5. Wave Interference Pattern...")
    
    # Custom initial condition for interference
    def interference_initial(x):
        # Two gaussian waves
        wave1 = 0.5 * np.exp(-100 * (x - 0.3)**2)
        wave2 = 0.5 * np.exp(-100 * (x - 0.7)**2)
        return wave1 + wave2
    
    result_interference = solver.solve_wave_equation_1d(
        length=1.0,
        nx=200,
        nt=800,
        c=1.0,
        initial_displacement=interference_initial,
        initial_velocity='zero',
        boundary_conditions='periodic',
        end_time=2.0
    )
    
    # Plot interference evolution
    x_int = result_interference['grid']['x']
    u_int = result_interference['solution']
    times_int = result_interference['times']
    
    plt.figure(figsize=(12, 6))
    
    # Space-time plot
    plt.subplot(1, 2, 1)
    T_int, X_int = np.meshgrid(times_int, x_int)
    plt.contourf(T_int, X_int, u_int.T, levels=20, cmap='RdBu_r')
    plt.colorbar(label='Displacement')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title('Wave Interference Pattern')
    
    # Selected time snapshots
    plt.subplot(1, 2, 2)
    interference_times = [0, len(times_int)//8, len(times_int)//4, 3*len(times_int)//8]
    for idx in interference_times:
        plt.plot(x_int, u_int[idx], label=f't = {times_int[idx]:.3f}s', linewidth=2)
    plt.xlabel('Position (m)')
    plt.ylabel('Displacement')
    plt.title('Wave Interference Snapshots')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== Wave Equation Demo Complete ===")


if __name__ == "__main__":
    main()