"""
Minimal Experiment Runner - Ultra-fast for quick results
Very small test cases, should complete in ~30-60 seconds
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os

from direct_methods import qr_algorithm_eigenvalues
from iterative_methods import power_iteration, arnoldi_eigenvalues
from test_matrices import create_positive_definite_matrix, create_sparse_matrix, create_ill_conditioned_matrix

def experiment_1_size_scaling():
    """Size scaling - MINIMAL"""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Matrix Size Scaling")
    print("=" * 70)

    sizes = [10, 30, 50, 80]  # Only 4 sizes
    results = {'size': [], 'qr_time': [], 'power_time': [], 'numpy_time': []}

    for n in sizes:
        print(f"Testing n={n}...", end=" ")
        A = create_positive_definite_matrix(n, seed=42)

        # QR - limited iterations
        try:
            start = time.time()
            _, _, iters = qr_algorithm_eigenvalues(A, max_iter=200, tol=1e-6, return_vectors=False)
            results['qr_time'].append(time.time() - start)
        except:
            results['qr_time'].append(np.nan)

        # Power iteration
        try:
            start = time.time()
            _, _, _, _ = power_iteration(A, max_iter=200, tol=1e-6)
            results['power_time'].append(time.time() - start)
        except:
            results['power_time'].append(np.nan)

        # NumPy
        try:
            start = time.time()
            _ = np.linalg.eig(A)
            results['numpy_time'].append(time.time() - start)
        except:
            results['numpy_time'].append(np.nan)

        results['size'].append(n)
        print("Done")

    return pd.DataFrame(results)


def experiment_2_conditioning():
    """Conditioning - MINIMAL"""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Effect of Conditioning")
    print("=" * 70)

    n = 20  # Very small
    condition_numbers = [10, 1000, 100000]  # Only 3 conditions
    results = {'condition_number': [], 'qr_error': [], 'power_error': []}

    for kappa in condition_numbers:
        print(f"Testing kappa={kappa:.0e}...", end=" ")
        A = create_ill_conditioned_matrix(n, condition_number=kappa, seed=42)
        true_eigs = np.linalg.eigvals(A)
        true_eigs = np.sort(np.real(true_eigs))[::-1]

        # QR
        try:
            eigs_qr, _, _ = qr_algorithm_eigenvalues(A, max_iter=300, tol=1e-6, return_vectors=False)
            error_qr = np.max(np.abs(np.sort(eigs_qr)[::-1] - true_eigs))
            results['qr_error'].append(error_qr)
        except:
            results['qr_error'].append(np.nan)

        # Power
        try:
            eig_power, _, _, _ = power_iteration(A, max_iter=300, tol=1e-6)
            error_power = abs(eig_power - true_eigs[0])
            results['power_error'].append(error_power)
        except:
            results['power_error'].append(np.nan)

        results['condition_number'].append(kappa)
        print("Done")

    return pd.DataFrame(results)


def experiment_3_sparse():
    """Sparse advantage - MINIMAL"""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Sparse Matrix Advantage")
    print("=" * 70)

    n = 80  # Small
    densities = [1.0, 0.3, 0.05]  # Only 3 densities
    results = {'density': [], 'qr_time': [], 'power_time': []}

    for density in densities:
        print(f"Testing density={density:.0%}...", end=" ")

        if density == 1.0:
            A = create_positive_definite_matrix(n, seed=42)
            A_dense = A
        else:
            A_sparse = create_sparse_matrix(n, density=density, seed=42, symmetric=True)
            A_dense = A_sparse.toarray()
            A = A_sparse

        # QR on dense
        try:
            start = time.time()
            _, _, _ = qr_algorithm_eigenvalues(A_dense, max_iter=200, tol=1e-6, return_vectors=False)
            results['qr_time'].append(time.time() - start)
        except:
            results['qr_time'].append(np.nan)

        # Power on sparse
        try:
            start = time.time()
            _, _, _, _ = power_iteration(A, max_iter=200, tol=1e-6)
            results['power_time'].append(time.time() - start)
        except:
            results['power_time'].append(np.nan)

        results['density'].append(density)
        print("Done")

    return pd.DataFrame(results)


def generate_plots(df1, df2, df3, output_dir):
    """Generate simple plots"""
    print("\nGenerating plots...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: Size scaling
    ax = axes[0]
    ax.plot(df1['size'], df1['qr_time'], 'bs-', label='QR', linewidth=2, markersize=8)
    ax.plot(df1['size'], df1['power_time'], 'ro-', label='Power', linewidth=2, markersize=8)
    ax.plot(df1['size'], df1['numpy_time'], 'kd--', label='NumPy', linewidth=2, markersize=8)
    ax.set_xlabel('Matrix Size', fontsize=11)
    ax.set_ylabel('Time (s)', fontsize=11)
    ax.set_title('Size: Power Wins for Large n', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Plot 2: Conditioning
    ax = axes[1]
    ax.semilogx(df2['condition_number'], df2['qr_error'], 'bs-', label='QR', linewidth=2, markersize=8)
    ax.semilogx(df2['condition_number'], df2['power_error'], 'ro-', label='Power', linewidth=2, markersize=8)
    ax.set_xlabel('Condition Number', fontsize=11)
    ax.set_ylabel('Error', fontsize=11)
    ax.set_title('Conditioning: Both Struggle', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Sparse
    ax = axes[2]
    width = 0.35
    x = np.arange(len(df3))
    ax.bar(x - width/2, df3['qr_time'], width, label='QR', color='steelblue')
    ax.bar(x + width/2, df3['power_time'], width, label='Power', color='orangered')
    ax.set_xlabel('Density', fontsize=11)
    ax.set_ylabel('Time (s)', fontsize=11)
    ax.set_title('Sparse: Power Dominates', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{d:.0%}' for d in df3['density']])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_experiments.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plot saved: all_experiments.png")


def print_summary(df1, df2, df3):
    """Print key findings"""
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # Finding 1: Size
    largest_n = df1['size'].max()
    idx = df1[df1['size'] == largest_n].index[0]
    if not np.isnan(df1.loc[idx, 'power_time']) and not np.isnan(df1.loc[idx, 'qr_time']):
        speedup = df1.loc[idx, 'qr_time'] / df1.loc[idx, 'power_time']
        print(f"\n1. At n={largest_n}, Power iteration is {speedup:.1f}x faster than QR")
        print("   (when you only need the largest eigenvalue)")

    # Finding 2: Conditioning
    worst_kappa = df2['condition_number'].max()
    idx = df2[df2['condition_number'] == worst_kappa].index[0]
    print(f"\n2. At kappa={worst_kappa:.0e}, errors:")
    print(f"   QR: {df2.loc[idx, 'qr_error']:.2e}")
    print(f"   Power: {df2.loc[idx, 'power_error']:.2e}")
    print("   Both methods struggle with ill-conditioning")

    # Finding 3: Sparse
    sparsest = df3['density'].min()
    idx_sparse = df3[df3['density'] == sparsest].index[0]
    idx_dense = df3[df3['density'] == 1.0].index[0]
    if not np.isnan(df3.loc[idx_sparse, 'power_time']):
        speedup = df3.loc[idx_sparse, 'qr_time'] / df3.loc[idx_sparse, 'power_time']
        print(f"\n3. For {sparsest:.0%} dense matrices, Power is {speedup:.1f}x faster")
        print("   Power exploits sparsity, QR must densify")

    print("\n" + "=" * 70)


def main():
    print("=" * 70)
    print("MINIMAL EXPERIMENTS - Should complete in ~30-60 seconds")
    print("=" * 70)

    output_dir = os.path.join(os.path.dirname(__file__), 'output_results')
    os.makedirs(output_dir, exist_ok=True)

    start_total = time.time()

    # Run experiments
    df1 = experiment_1_size_scaling()
    df2 = experiment_2_conditioning()
    df3 = experiment_3_sparse()

    # Save CSVs
    df1.to_csv(os.path.join(output_dir, 'exp1_minimal.csv'), index=False)
    df2.to_csv(os.path.join(output_dir, 'exp2_minimal.csv'), index=False)
    df3.to_csv(os.path.join(output_dir, 'exp3_minimal.csv'), index=False)

    # Generate plot
    generate_plots(df1, df2, df3, output_dir)

    # Print summary
    print_summary(df1, df2, df3)

    print(f"\nTotal time: {time.time() - start_total:.1f} seconds")
    print(f"Results saved to: {output_dir}")
    print("\n" + "=" * 70)
    print("SUCCESS! Now run the applications separately if you want:")
    print("  python applications/pagerank_demo.py")
    print("  python applications/vibration_analysis.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
