"""
Vibration Analysis Demo: Natural Frequencies of a 1D Bar
Demonstrates eigenvalue methods applied to mechanical vibrations
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy.linalg import eigh as scipy_eigh

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from iterative_methods import power_iteration, inverse_power_iteration
from direct_methods import qr_algorithm_eigenvalues
from test_matrices import create_laplacian_1d

def vibrating_bar_matrix(n, E=1.0, rho=1.0, L=1.0):
    """
    Create the stiffness matrix for a 1D vibrating bar.

    The equation of motion is: M * u_tt = K * u
    where M is mass matrix, K is stiffness matrix, u is displacement.

    Eigenvalue problem: K * phi = omega^2 * M * phi
    where phi are mode shapes, omega are natural frequencies.

    For uniform bar with fixed-fixed ends, this simplifies to a Laplacian.

    Parameters:
    -----------
    n : int
        Number of nodes (interior points)
    E : float
        Young's modulus
    rho : float
        Density
    L : float
        Length of bar

    Returns:
    --------
    K : ndarray
        Stiffness matrix
    M : ndarray
        Mass matrix
    omega_exact : ndarray
        Exact natural frequencies (for validation)
    """
    # Spatial step
    h = L / (n + 1)

    # Stiffness matrix (discrete Laplacian)
    K = create_laplacian_1d(n) * (E / h**2)

    # Mass matrix (lumped mass)
    M = np.eye(n) * (rho * h)

    # Exact natural frequencies (continuous bar, fixed-fixed)
    # omega_k = k * pi * sqrt(E/rho) / L, k = 1, 2, 3, ...
    omega_exact = np.array([k * np.pi * np.sqrt(E / rho) / L for k in range(1, n+1)])

    return K, M, omega_exact


def solve_eigenvalue_problem(K, M):
    """
    Solve generalized eigenvalue problem: K * phi = omega^2 * M * phi

    Uses scipy.linalg.eigh for accurate generalized eigenvalue solution.
    This is more stable than M^{-1}K which amplifies conditioning errors.

    Parameters:
    -----------
    K : ndarray
        Stiffness matrix (symmetric positive definite)
    M : ndarray
        Mass matrix (symmetric positive definite)

    Returns:
    --------
    omega : ndarray
        Natural frequencies (sorted ascending)
    phi : ndarray
        Mode shapes (eigenvectors, columns correspond to frequencies)
    """
    # Use scipy's specialized generalized eigenvalue solver
    # This solves K*phi = lambda*M*phi where lambda = omega^2
    # More accurate than M^{-1}K which has condition number kappa(M)*kappa(K)
    eigenvalues, eigenvectors = scipy_eigh(K, M)

    # omega^2 = eigenvalues, so omega = sqrt(eigenvalues)
    omega = np.sqrt(np.abs(eigenvalues))  # abs for numerical safety

    # Already sorted by scipy.linalg.eigh (ascending order)
    phi = eigenvectors.real

    return omega, phi


def plot_mode_shapes(phi, L=1.0, n_modes=4, title="Mode Shapes", output_file=None):
    """
    Plot the first few mode shapes.

    Parameters:
    -----------
    phi : ndarray (n x n)
        Mode shape matrix (column i is mode i)
    L : float
        Length of bar
    n_modes : int
        Number of modes to plot
    title : str
        Plot title
    output_file : str
        Output filename
    """
    n = phi.shape[0]
    x_interior = np.linspace(0, L, n + 2)[1:-1]  # Interior points
    x_full = np.linspace(0, L, n + 2)  # Including boundaries

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    for i in range(min(n_modes, 4)):
        ax = axes[i]

        # Add boundary conditions (fixed-fixed: displacement = 0 at ends)
        phi_full = np.concatenate([[0], phi[:, i], [0]])

        ax.plot(x_full, phi_full, 'b-', linewidth=2, label=f'Mode {i+1}')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.scatter(x_interior, phi[:, i], color='red', s=30, zorder=5)

        ax.set_xlabel('Position along bar (x/L)', fontsize=11)
        ax.set_ylabel('Displacement', fontsize=11)
        ax.set_title(f'Mode {i+1} Shape', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved mode shapes to {output_file}")
    else:
        plt.show()

    plt.close()


def compare_methods_vibration(n=20):
    """
    Compare different eigenvalue methods on vibration problem.

    Parameters:
    -----------
    n : int
        Number of interior nodes

    Returns:
    --------
    results : dict
        Comparison results
    """
    print(f"Vibration Analysis: 1D Bar with {n} nodes")
    print("=" * 70)

    # Create problem
    K, M, omega_exact = vibrating_bar_matrix(n)
    M_inv = np.diag(1.0 / np.diag(M))
    A = M_inv @ K

    print(f"\nProblem setup:")
    print(f"  Stiffness matrix K: {n}x{n}")
    print(f"  Mass matrix M: {n}x{n} (diagonal)")
    print(f"  Standard form: A = M^{{-1}} K")

    # Method 1: QR Algorithm
    print("\n" + "-" * 70)
    print("Method 1: QR Algorithm (Direct Method)")
    print("-" * 70)
    import time
    start = time.time()
    results_qr = qr_algorithm_eigenvalues(A, max_iter=1000, tol=1e-10)
    time_qr = time.time() - start

    omega_qr = np.sqrt(results_qr[0])
    omega_qr = np.sort(omega_qr)

    print(f"Time: {time_qr:.6f} seconds")
    print(f"Iterations: {results_qr[2]}")
    print(f"First 5 natural frequencies:")
    for i in range(min(5, len(omega_qr))):
        exact = omega_exact[i]
        error = abs(omega_qr[i] - exact)
        print(f"  omega_{i+1}: {omega_qr[i]:.6f} (exact: {exact:.6f}, error: {error:.2e})")

    # Method 2: Power Iteration (highest frequency)
    print("\n" + "-" * 70)
    print("Method 2: Power Iteration (Highest Frequency Only)")
    print("-" * 70)
    start = time.time()
    eig_power, vec_power, iters_power, conv_power = power_iteration(A, max_iter=1000, tol=1e-10)
    time_power = time.time() - start

    omega_power = np.sqrt(eig_power)
    exact_highest = omega_exact[-1]  # Highest frequency

    print(f"Time: {time_power:.6f} seconds")
    print(f"Iterations: {iters_power}")
    print(f"Converged: {conv_power}")
    print(f"Highest frequency: {omega_power:.6f}")
    print(f"Exact highest: {exact_highest:.6f}")
    print(f"Error: {abs(omega_power - exact_highest):.2e}")

    # Method 3: Inverse Power Iteration (lowest frequency)
    print("\n" + "-" * 70)
    print("Method 3: Inverse Power Iteration (Lowest Frequency Only)")
    print("-" * 70)
    start = time.time()
    eig_inv, vec_inv, iters_inv, conv_inv = inverse_power_iteration(A, mu=0.0, max_iter=1000, tol=1e-10)
    time_inv = time.time() - start

    omega_inv = np.sqrt(eig_inv)
    exact_lowest = omega_exact[0]  # Lowest frequency

    print(f"Time: {time_inv:.6f} seconds")
    print(f"Iterations: {iters_inv}")
    print(f"Converged: {conv_inv}")
    print(f"Lowest frequency: {omega_inv:.6f}")
    print(f"Exact lowest: {exact_lowest:.6f}")
    print(f"Error: {abs(omega_inv - exact_lowest):.2e}")

    # Method 4: NumPy (for comparison)
    print("\n" + "-" * 70)
    print("Method 4: NumPy (LAPACK)")
    print("-" * 70)
    start = time.time()
    omega_np, phi_np = solve_eigenvalue_problem(K, M)
    time_np = time.time() - start

    print(f"Time: {time_np:.6f} seconds")
    print(f"First 5 natural frequencies:")
    for i in range(min(5, len(omega_np))):
        exact = omega_exact[i]
        error = abs(omega_np[i] - exact)
        print(f"  omega_{i+1}: {omega_np[i]:.6f} (exact: {exact:.6f}, error: {error:.2e})")

    # Summary comparison
    print("\n" + "-" * 70)
    print("Performance Comparison:")
    print("-" * 70)
    print(f"{'Method':<30} {'Time (s)':<12} {'Relative Speed':<15}")
    print("-" * 70)
    baseline = time_np
    print(f"{'QR Algorithm':<30} {time_qr:<12.6f} {time_qr/baseline:<15.2f}x")
    print(f"{'Power Iteration (1 mode)':<30} {time_power:<12.6f} {time_power/baseline:<15.2f}x")
    print(f"{'Inverse Power (1 mode)':<30} {time_inv:<12.6f} {time_inv/baseline:<15.2f}x")
    print(f"{'NumPy (all modes)':<30} {time_np:<12.6f} {time_np/baseline:<15.2f}x")

    return {
        'omega_qr': omega_qr,
        'omega_np': omega_np,
        'omega_exact': omega_exact,
        'phi_np': phi_np,
        'time_qr': time_qr,
        'time_np': time_np,
        'time_power': time_power,
        'time_inv': time_inv
    }


def main():
    print("Vibration Analysis Demo: Natural Frequencies of 1D Bar")
    print("=" * 70)

    # Run comparison
    n = 30
    results = compare_methods_vibration(n=n)

    # Plot mode shapes
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output_results')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'vibration_mode_shapes.png')

    plot_mode_shapes(results['phi_np'], L=1.0, n_modes=4,
                    title=f"First 4 Vibration Modes of 1D Bar ({n} nodes)",
                    output_file=output_file)

    # Create frequency comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Frequency comparison
    modes = np.arange(1, min(11, len(results['omega_exact'])+1))
    ax1.plot(modes, results['omega_exact'][:len(modes)], 'ko-', label='Exact', linewidth=2, markersize=8)
    ax1.plot(modes, results['omega_qr'][:len(modes)], 'bs--', label='QR Algorithm', linewidth=1.5, markersize=6)
    ax1.plot(modes, results['omega_np'][:len(modes)], 'r^--', label='NumPy', linewidth=1.5, markersize=6)

    ax1.set_xlabel('Mode Number', fontsize=12)
    ax1.set_ylabel('Natural Frequency (omega)', fontsize=12)
    ax1.set_title('Natural Frequencies Comparison', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Error
    error_qr = np.abs(results['omega_qr'][:len(modes)] - results['omega_exact'][:len(modes)])
    error_np = np.abs(results['omega_np'][:len(modes)] - results['omega_exact'][:len(modes)])

    ax2.semilogy(modes, error_qr, 'bs-', label='QR Algorithm', linewidth=2, markersize=6)
    ax2.semilogy(modes, error_np, 'r^-', label='NumPy', linewidth=2, markersize=6)

    ax2.set_xlabel('Mode Number', fontsize=12)
    ax2.set_ylabel('Absolute Error', fontsize=12)
    ax2.set_title('Error in Natural Frequencies', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    output_file2 = os.path.join(output_dir, 'vibration_frequency_comparison.png')
    plt.savefig(output_file2, dpi=300, bbox_inches='tight')
    print(f"\nSaved frequency comparison to {output_file2}")
    plt.close()

    print("\n" + "=" * 70)
    print("Vibration analysis demo completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
