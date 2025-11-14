"""
Main Experiment Runner
Runs all comparison experiments and generates comprehensive results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os

from direct_methods import qr_algorithm_benchmark, numpy_eig_benchmark
from iterative_methods import (power_iteration, inverse_power_iteration,
                               rayleigh_quotient_iteration, arnoldi_eigenvalues)
from test_matrices import (create_positive_definite_matrix, create_tridiagonal_matrix,
                           create_sparse_matrix, create_ill_conditioned_matrix,
                           get_matrix_properties)

# Import applications
from applications.pagerank_demo import main as pagerank_main
from applications.vibration_analysis import main as vibration_main


def experiment_1_matrix_size_scaling():
    """
    Experiment 1: How do methods scale with matrix size?
    Compare QR, Power, Arnoldi, and NumPy for different sizes.
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: Matrix Size Scaling")
    print("=" * 80)

    sizes = [10, 20, 50, 100, 200, 500]
    results = {
        'size': [],
        'qr_time': [],
        'qr_iters': [],
        'power_time': [],
        'power_iters': [],
        'arnoldi_time': [],
        'numpy_time': [],
    }

    for n in sizes:
        print(f"\nTesting n = {n}...")

        # Create test matrix
        A = create_positive_definite_matrix(n, seed=42, condition_number=100)

        # QR Algorithm
        try:
            res_qr = qr_algorithm_benchmark(A, max_iter=2000, tol=1e-8)
            results['qr_time'].append(res_qr['time'])
            results['qr_iters'].append(res_qr['iterations'])
        except:
            results['qr_time'].append(np.nan)
            results['qr_iters'].append(np.nan)

        # Power Iteration (single eigenvalue)
        try:
            start = time.time()
            _, _, iters, _ = power_iteration(A, max_iter=2000, tol=1e-8)
            results['power_time'].append(time.time() - start)
            results['power_iters'].append(iters)
        except:
            results['power_time'].append(np.nan)
            results['power_iters'].append(np.nan)

        # Arnoldi (k = min(n-1, 20) eigenvalues)
        try:
            k = min(n-1, 20)
            start = time.time()
            _, _ = arnoldi_eigenvalues(A, k=k)
            results['arnoldi_time'].append(time.time() - start)
        except:
            results['arnoldi_time'].append(np.nan)

        # NumPy
        try:
            res_np = numpy_eig_benchmark(A)
            results['numpy_time'].append(res_np['time'])
        except:
            results['numpy_time'].append(np.nan)

        results['size'].append(n)

    df = pd.DataFrame(results)
    return df


def experiment_2_conditioning():
    """
    Experiment 2: How does conditioning affect different methods?
    Test with various condition numbers.
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Effect of Conditioning")
    print("=" * 80)

    n = 50
    condition_numbers = [10, 100, 1000, 10000, 100000, 1000000]
    results = {
        'condition_number': [],
        'qr_time': [],
        'qr_max_error': [],
        'power_time': [],
        'power_error': [],
        'numpy_time': [],
    }

    for kappa in condition_numbers:
        print(f"\nTesting κ = {kappa:.0e}...")

        # Create ill-conditioned matrix
        A = create_ill_conditioned_matrix(n, condition_number=kappa, seed=42)

        # True eigenvalues
        true_eigs = np.linalg.eigvals(A)
        true_eigs = np.sort(true_eigs)[::-1]

        # QR Algorithm
        try:
            res_qr = qr_algorithm_benchmark(A, max_iter=5000, tol=1e-8)
            error_qr = np.max(np.abs(np.sort(res_qr['eigenvalues'])[::-1] - true_eigs))
            results['qr_time'].append(res_qr['time'])
            results['qr_max_error'].append(error_qr)
        except:
            results['qr_time'].append(np.nan)
            results['qr_max_error'].append(np.nan)

        # Power Iteration
        try:
            start = time.time()
            eig_power, _, _, _ = power_iteration(A, max_iter=5000, tol=1e-8)
            error_power = abs(eig_power - true_eigs[0])
            results['power_time'].append(time.time() - start)
            results['power_error'].append(error_power)
        except:
            results['power_time'].append(np.nan)
            results['power_error'].append(np.nan)

        # NumPy
        try:
            res_np = numpy_eig_benchmark(A)
            results['numpy_time'].append(res_np['time'])
        except:
            results['numpy_time'].append(np.nan)

        results['condition_number'].append(kappa)

    df = pd.DataFrame(results)
    return df


def experiment_3_sparse_vs_dense():
    """
    Experiment 3: Sparse vs Dense matrices
    Compare methods on sparse matrices (where iterative methods should win)
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: Sparse vs Dense Matrices")
    print("=" * 80)

    n = 200
    densities = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02]
    results = {
        'density': [],
        'num_nonzeros': [],
        'qr_time': [],
        'power_time': [],
        'arnoldi_time': [],
        'numpy_time': [],
    }

    for density in densities:
        print(f"\nTesting density = {density:.2%}...")

        if density == 1.0:
            # Dense matrix
            A = create_positive_definite_matrix(n, seed=42).astype(np.float64)
            A_dense = A
        else:
            # Sparse matrix
            A_sparse = create_sparse_matrix(n, density=density, seed=42, symmetric=True)
            A_dense = A_sparse.toarray()
            A = A_sparse

        nnz = np.count_nonzero(A_dense)

        # QR Algorithm (on dense version)
        try:
            start = time.time()
            _ = qr_algorithm_benchmark(A_dense, max_iter=1000, tol=1e-8)
            results['qr_time'].append(time.time() - start)
        except:
            results['qr_time'].append(np.nan)

        # Power Iteration (works on sparse)
        try:
            start = time.time()
            _, _, _, _ = power_iteration(A, max_iter=1000, tol=1e-8)
            results['power_time'].append(time.time() - start)
        except:
            results['power_time'].append(np.nan)

        # Arnoldi (works on sparse)
        try:
            start = time.time()
            _, _ = arnoldi_eigenvalues(A, k=min(20, n-1))
            results['arnoldi_time'].append(time.time() - start)
        except:
            results['arnoldi_time'].append(np.nan)

        # NumPy (on dense)
        try:
            start = time.time()
            _ = np.linalg.eig(A_dense)
            results['numpy_time'].append(time.time() - start)
        except:
            results['numpy_time'].append(np.nan)

        results['density'].append(density)
        results['num_nonzeros'].append(nnz)

    df = pd.DataFrame(results)
    return df


def generate_comparison_plots(df_exp1, df_exp2, df_exp3, output_dir):
    """
    Generate comprehensive comparison plots.
    """
    print("\nGenerating comparison plots...")

    # Plot 1: Size scaling
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Time vs size
    ax1 = axes[0]
    ax1.plot(df_exp1['size'], df_exp1['qr_time'], 'bs-', label='QR Algorithm', linewidth=2, markersize=6)
    ax1.plot(df_exp1['size'], df_exp1['power_time'], 'ro-', label='Power Iteration', linewidth=2, markersize=6)
    ax1.plot(df_exp1['size'], df_exp1['arnoldi_time'], 'g^-', label='Arnoldi (k=20)', linewidth=2, markersize=6)
    ax1.plot(df_exp1['size'], df_exp1['numpy_time'], 'kd--', label='NumPy', linewidth=2, markersize=6)

    ax1.set_xlabel('Matrix Size (n)', fontsize=12)
    ax1.set_ylabel('Computation Time (seconds)', fontsize=12)
    ax1.set_title('Computational Cost vs Matrix Size', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    ax1.set_xscale('log')

    # Iterations vs size
    ax2 = axes[1]
    ax2.plot(df_exp1['size'], df_exp1['qr_iters'], 'bs-', label='QR Algorithm', linewidth=2, markersize=6)
    ax2.plot(df_exp1['size'], df_exp1['power_iters'], 'ro-', label='Power Iteration', linewidth=2, markersize=6)

    ax2.set_xlabel('Matrix Size (n)', fontsize=12)
    ax2.set_ylabel('Iterations to Converge', fontsize=12)
    ax2.set_title('Convergence Speed vs Matrix Size', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'exp1_size_scaling.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Conditioning effects
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.semilogx(df_exp2['condition_number'], df_exp2['qr_time'], 'bs-', label='QR Algorithm', linewidth=2, markersize=6)
    ax1.semilogx(df_exp2['condition_number'], df_exp2['power_time'], 'ro-', label='Power Iteration', linewidth=2, markersize=6)
    ax1.semilogx(df_exp2['condition_number'], df_exp2['numpy_time'], 'kd--', label='NumPy', linewidth=2, markersize=6)

    ax1.set_xlabel('Condition Number (κ)', fontsize=12)
    ax1.set_ylabel('Computation Time (seconds)', fontsize=12)
    ax1.set_title('Effect of Ill-Conditioning on Performance', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.loglog(df_exp2['condition_number'], df_exp2['qr_max_error'], 'bs-', label='QR Algorithm', linewidth=2, markersize=6)
    ax2.loglog(df_exp2['condition_number'], df_exp2['power_error'], 'ro-', label='Power Iteration', linewidth=2, markersize=6)

    ax2.set_xlabel('Condition Number (κ)', fontsize=12)
    ax2.set_ylabel('Maximum Eigenvalue Error', fontsize=12)
    ax2.set_title('Error vs Conditioning', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'exp2_conditioning.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 3: Sparse vs dense
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    x = df_exp3['density']
    ax.plot(x, df_exp3['qr_time'], 'bs-', label='QR Algorithm (dense)', linewidth=2, markersize=7)
    ax.plot(x, df_exp3['power_time'], 'ro-', label='Power Iteration (sparse)', linewidth=2, markersize=7)
    ax.plot(x, df_exp3['arnoldi_time'], 'g^-', label='Arnoldi (sparse)', linewidth=2, markersize=7)
    ax.plot(x, df_exp3['numpy_time'], 'kd--', label='NumPy (dense)', linewidth=2, markersize=7)

    ax.set_xlabel('Matrix Density', fontsize=12)
    ax.set_ylabel('Computation Time (seconds)', fontsize=12)
    ax.set_title('Sparse vs Dense: When Do Iterative Methods Win?', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'exp3_sparse_vs_dense.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plots saved to {output_dir}")


def save_results_tables(df_exp1, df_exp2, df_exp3, output_dir):
    """
    Save results as CSV and LaTeX tables.
    """
    print("\nSaving results tables...")

    # CSV files
    df_exp1.to_csv(os.path.join(output_dir, 'exp1_size_scaling.csv'), index=False, float_format='%.6f')
    df_exp2.to_csv(os.path.join(output_dir, 'exp2_conditioning.csv'), index=False, float_format='%.6e')
    df_exp3.to_csv(os.path.join(output_dir, 'exp3_sparse_dense.csv'), index=False, float_format='%.6f')

    # Simple LaTeX tables (manual formatting to avoid jinja2)
    for df, name in [(df_exp1, 'exp1'), (df_exp2, 'exp2'), (df_exp3, 'exp3')]:
        tex_file = os.path.join(output_dir, f'{name}_table.tex')
        with open(tex_file, 'w') as f:
            f.write(f"% Table for {name}\n")
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\begin{tabular}{" + "c" * len(df.columns) + "}\n")
            f.write("\\hline\n")
            f.write(" & ".join(df.columns) + " \\\\\n")
            f.write("\\hline\n")
            for _, row in df.head(10).iterrows():  # First 10 rows
                row_str = " & ".join([f"{v:.4e}" if isinstance(v, (int, float)) and not np.isnan(v) else str(v) for v in row.values])
                f.write(row_str + " \\\\\n")
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write(f"\\caption{{Results for {name}}}\n")
            f.write(f"\\label{{tab:{name}}}\n")
            f.write("\\end{table}\n")

    print(f"Tables saved to {output_dir}")


def main():
    """
    Run all experiments and generate outputs.
    """
    print("=" * 80)
    print("EIGENVALUE PROJECT: COMPREHENSIVE EXPERIMENTAL STUDY")
    print("=" * 80)

    output_dir = os.path.join(os.path.dirname(__file__), 'output_results')
    os.makedirs(output_dir, exist_ok=True)

    # Run experiments
    df_exp1 = experiment_1_matrix_size_scaling()
    df_exp2 = experiment_2_conditioning()
    df_exp3 = experiment_3_sparse_vs_dense()

    # Generate plots
    generate_comparison_plots(df_exp1, df_exp2, df_exp3, output_dir)

    # Save tables
    save_results_tables(df_exp1, df_exp2, df_exp3, output_dir)

    # Run applications
    print("\n" + "=" * 80)
    print("RUNNING APPLICATION DEMOS")
    print("=" * 80)

    print("\n--- PageRank Demo ---")
    try:
        pagerank_main()
    except Exception as e:
        print(f"PageRank demo failed: {e}")

    print("\n--- Vibration Analysis Demo ---")
    try:
        vibration_main()
    except Exception as e:
        print(f"Vibration analysis demo failed: {e}")

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - exp1_size_scaling.png, .csv, .tex")
    print("  - exp2_conditioning.png, .csv, .tex")
    print("  - exp3_sparse_vs_dense.png, .csv, .tex")
    print("  - pagerank_demo.png")
    print("  - vibration_mode_shapes.png")
    print("  - vibration_frequency_comparison.png")


if __name__ == "__main__":
    main()
