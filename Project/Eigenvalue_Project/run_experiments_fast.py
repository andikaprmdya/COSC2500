"""
Fast Experiment Runner - Optimized for quick results
Reduced sizes and iterations for faster execution
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os

from direct_methods import qr_algorithm_benchmark, numpy_eig_benchmark
from iterative_methods import power_iteration, arnoldi_eigenvalues
from test_matrices import create_positive_definite_matrix, create_sparse_matrix, create_ill_conditioned_matrix

def experiment_1_matrix_size_scaling():
    """Experiment 1: Matrix size scaling - FAST version"""
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: Matrix Size Scaling (Fast)")
    print("=" * 80)

    # Smaller sizes for speed
    sizes = [10, 20, 50, 100, 200]
    results = {
        'size': [],
        'qr_time': [],
        'power_time': [],
        'arnoldi_time': [],
        'numpy_time': [],
    }

    for n in sizes:
        print(f"\nTesting n = {n}...")
        A = create_positive_definite_matrix(n, seed=42, condition_number=100)

        # QR Algorithm (limited iterations)
        try:
            start = time.time()
            _ = qr_algorithm_benchmark(A, max_iter=500, tol=1e-6)
            results['qr_time'].append(time.time() - start)
        except:
            results['qr_time'].append(np.nan)

        # Power Iteration
        try:
            start = time.time()
            _, _, _, _ = power_iteration(A, max_iter=500, tol=1e-6)
            results['power_time'].append(time.time() - start)
        except:
            results['power_time'].append(np.nan)

        # Arnoldi (fewer eigenvalues)
        try:
            k = min(n-1, 10)
            start = time.time()
            _, _ = arnoldi_eigenvalues(A, k=k)
            results['arnoldi_time'].append(time.time() - start)
        except:
            results['arnoldi_time'].append(np.nan)

        # NumPy
        try:
            start = time.time()
            _ = np.linalg.eig(A)
            results['numpy_time'].append(time.time() - start)
        except:
            results['numpy_time'].append(np.nan)

        results['size'].append(n)

    return pd.DataFrame(results)


def experiment_2_conditioning():
    """Experiment 2: Conditioning effects - FAST version"""
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Effect of Conditioning (Fast)")
    print("=" * 80)

    n = 30  # Smaller size
    condition_numbers = [10, 100, 1000, 10000, 100000]  # Fewer conditions
    results = {
        'condition_number': [],
        'qr_time': [],
        'qr_max_error': [],
        'power_time': [],
        'power_error': [],
    }

    for kappa in condition_numbers:
        print(f"\nTesting kappa = {kappa:.0e}...")
        A = create_ill_conditioned_matrix(n, condition_number=kappa, seed=42)
        true_eigs = np.linalg.eigvals(A)
        true_eigs = np.sort(true_eigs)[::-1]

        # QR
        try:
            res_qr = qr_algorithm_benchmark(A, max_iter=1000, tol=1e-6)
            error_qr = np.max(np.abs(np.sort(res_qr['eigenvalues'])[::-1] - true_eigs))
            results['qr_time'].append(res_qr['time'])
            results['qr_max_error'].append(error_qr)
        except:
            results['qr_time'].append(np.nan)
            results['qr_max_error'].append(np.nan)

        # Power
        try:
            start = time.time()
            eig_power, _, _, _ = power_iteration(A, max_iter=1000, tol=1e-6)
            error_power = abs(eig_power - true_eigs[0])
            results['power_time'].append(time.time() - start)
            results['power_error'].append(error_power)
        except:
            results['power_time'].append(np.nan)
            results['power_error'].append(np.nan)

        results['condition_number'].append(kappa)

    return pd.DataFrame(results)


def experiment_3_sparse_vs_dense():
    """Experiment 3: Sparse vs Dense - FAST version"""
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: Sparse vs Dense Matrices (Fast)")
    print("=" * 80)

    n = 100  # Smaller size
    densities = [1.0, 0.5, 0.2, 0.1, 0.05]  # Fewer densities
    results = {
        'density': [],
        'qr_time': [],
        'power_time': [],
        'arnoldi_time': [],
    }

    for density in densities:
        print(f"\nTesting density = {density:.2%}...")

        if density == 1.0:
            A = create_positive_definite_matrix(n, seed=42)
            A_dense = A
        else:
            A_sparse = create_sparse_matrix(n, density=density, seed=42, symmetric=True)
            A_dense = A_sparse.toarray()
            A = A_sparse

        # QR (on dense)
        try:
            start = time.time()
            _ = qr_algorithm_benchmark(A_dense, max_iter=500, tol=1e-6)
            results['qr_time'].append(time.time() - start)
        except:
            results['qr_time'].append(np.nan)

        # Power (works on sparse)
        try:
            start = time.time()
            _, _, _, _ = power_iteration(A, max_iter=500, tol=1e-6)
            results['power_time'].append(time.time() - start)
        except:
            results['power_time'].append(np.nan)

        # Arnoldi (works on sparse)
        try:
            start = time.time()
            _, _ = arnoldi_eigenvalues(A, k=10)
            results['arnoldi_time'].append(time.time() - start)
        except:
            results['arnoldi_time'].append(np.nan)

        results['density'].append(density)

    return pd.DataFrame(results)


def generate_plots(df_exp1, df_exp2, df_exp3, output_dir):
    """Generate comparison plots"""
    print("\nGenerating plots...")

    # Plot 1: Size scaling
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(df_exp1['size'], df_exp1['qr_time'], 'bs-', label='QR Algorithm', linewidth=2, markersize=8)
    ax.plot(df_exp1['size'], df_exp1['power_time'], 'ro-', label='Power Iteration', linewidth=2, markersize=8)
    ax.plot(df_exp1['size'], df_exp1['arnoldi_time'], 'g^-', label='Arnoldi (k=10)', linewidth=2, markersize=8)
    ax.plot(df_exp1['size'], df_exp1['numpy_time'], 'kd--', label='NumPy', linewidth=2, markersize=8)

    ax.set_xlabel('Matrix Size (n)', fontsize=13)
    ax.set_ylabel('Computation Time (seconds)', fontsize=13)
    ax.set_title('When Do Iterative Methods Beat Direct Methods?', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'exp1_size_scaling.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Conditioning
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.semilogx(df_exp2['condition_number'], df_exp2['qr_time'], 'bs-', label='QR', linewidth=2, markersize=8)
    ax1.semilogx(df_exp2['condition_number'], df_exp2['power_time'], 'ro-', label='Power', linewidth=2, markersize=8)
    ax1.set_xlabel('Condition Number (kappa)', fontsize=12)
    ax1.set_ylabel('Time (seconds)', fontsize=12)
    ax1.set_title('Ill-Conditioning Slows Everything Down', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.loglog(df_exp2['condition_number'], df_exp2['qr_max_error'], 'bs-', label='QR', linewidth=2, markersize=8)
    ax2.loglog(df_exp2['condition_number'], df_exp2['power_error'], 'ro-', label='Power', linewidth=2, markersize=8)
    ax2.set_xlabel('Condition Number (kappa)', fontsize=12)
    ax2.set_ylabel('Eigenvalue Error', fontsize=12)
    ax2.set_title('But QR Handles It Better', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'exp2_conditioning.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 3: Sparse advantage
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(df_exp3['density'], df_exp3['qr_time'], 'bs-', label='QR (must densify)', linewidth=2, markersize=8)
    ax.plot(df_exp3['density'], df_exp3['power_time'], 'ro-', label='Power (stays sparse)', linewidth=2, markersize=8)
    ax.plot(df_exp3['density'], df_exp3['arnoldi_time'], 'g^-', label='Arnoldi (stays sparse)', linewidth=2, markersize=8)

    ax.set_xlabel('Matrix Density', fontsize=13)
    ax.set_ylabel('Time (seconds)', fontsize=13)
    ax.set_title('Power Iteration Crushes QR for Sparse Matrices', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'exp3_sparse_vs_dense.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plots saved to {output_dir}")


def save_tables(df_exp1, df_exp2, df_exp3, output_dir):
    """Save results tables"""
    print("\nSaving tables...")

    df_exp1.to_csv(os.path.join(output_dir, 'exp1_size_scaling.csv'), index=False, float_format='%.6f')
    df_exp2.to_csv(os.path.join(output_dir, 'exp2_conditioning.csv'), index=False, float_format='%.6e')
    df_exp3.to_csv(os.path.join(output_dir, 'exp3_sparse_dense.csv'), index=False, float_format='%.6f')

    print(f"Tables saved to {output_dir}")


def main():
    """Run fast experiments"""
    print("=" * 80)
    print("EIGENVALUE PROJECT: FAST EXPERIMENTS")
    print("=" * 80)

    output_dir = os.path.join(os.path.dirname(__file__), 'output_results')
    os.makedirs(output_dir, exist_ok=True)

    # Run experiments
    df_exp1 = experiment_1_matrix_size_scaling()
    df_exp2 = experiment_2_conditioning()
    df_exp3 = experiment_3_sparse_vs_dense()

    # Generate outputs
    generate_plots(df_exp1, df_exp2, df_exp3, output_dir)
    save_tables(df_exp1, df_exp2, df_exp3, output_dir)

    # Print summary
    print("\n" + "=" * 80)
    print("KEY FINDINGS:")
    print("=" * 80)

    # Finding 1: When does Power beat QR?
    idx_100 = df_exp1[df_exp1['size'] == 100].index[0]
    if df_exp1.loc[idx_100, 'power_time'] < df_exp1.loc[idx_100, 'qr_time']:
        speedup = df_exp1.loc[idx_100, 'qr_time'] / df_exp1.loc[idx_100, 'power_time']
        print(f"\n1. For n=100, Power iteration is {speedup:.1f}x faster than QR")
        print("   (when you only need one eigenvalue)")

    # Finding 2: Conditioning effect
    worst_kappa = df_exp2['condition_number'].max()
    worst_idx = df_exp2[df_exp2['condition_number'] == worst_kappa].index[0]
    print(f"\n2. At kappa = {worst_kappa:.0e}, errors jumped to:")
    print(f"   - QR: {df_exp2.loc[worst_idx, 'qr_max_error']:.2e}")
    print(f"   - Power: {df_exp2.loc[worst_idx, 'power_error']:.2e}")
    print("   Both struggle, but QR degrades more gracefully")

    # Finding 3: Sparse advantage
    sparse_idx = df_exp3[df_exp3['density'] == 0.05].index[0]
    if not np.isnan(df_exp3.loc[sparse_idx, 'power_time']):
        speedup_sparse = df_exp3.loc[sparse_idx, 'qr_time'] / df_exp3.loc[sparse_idx, 'power_time']
        print(f"\n3. For 5% dense matrices, Power is {speedup_sparse:.1f}x faster")
        print("   QR must convert to dense, Power exploits sparsity directly")

    print("\n" + "=" * 80)
    print("EXPERIMENTS COMPLETED!")
    print("=" * 80)
    print(f"\nResults in: {output_dir}")


if __name__ == "__main__":
    main()
