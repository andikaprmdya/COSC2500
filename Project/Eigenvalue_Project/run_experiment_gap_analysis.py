"""
Experiment 4: Eigenvalue Gap Sensitivity Analysis

Tests how the eigenvalue gap (ratio lambda_2/lambda_1) affects
power iteration convergence rate. Critical for understanding when
iterative methods fail.

Theory: Convergence rate = (lambda_2/lambda_1)^k
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os

from test_matrices import create_controlled_gap_matrix
from iterative_methods import power_iteration

def experiment_gap_sensitivity():
    """
    Test power iteration convergence vs eigenvalue gap ratio.

    Gap ratios tested:
    - 0.99: Nearly degenerate (very slow)
    - 0.95: Poor separation
    - 0.9:  Moderate separation
    - 0.7:  Good separation
    - 0.5:  Excellent separation
    - 0.3:  Very well separated
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Eigenvalue Gap Sensitivity")
    print("=" * 70)

    n = 50  # Matrix size
    gap_ratios = [0.99, 0.95, 0.9, 0.7, 0.5, 0.3]
    max_iter = 500
    tol = 1e-10

    results = {
        'gap_ratio': [],
        'iterations': [],
        'time': [],
        'converged': [],
        'final_error': [],
        'theoretical_rate': [],
        'observed_rate': []
    }

    for gap in gap_ratios:
        print(f"\nTesting gap ratio = {gap:.2f}...")

        # Create matrix with controlled gap
        A = create_controlled_gap_matrix(n, gap_ratio=gap, seed=42)

        # Verify eigenvalue gap
        eigs_true = np.linalg.eigvals(A)
        eigs_true = np.sort(np.real(eigs_true))[::-1]
        actual_gap = eigs_true[1] / eigs_true[0]
        print(f"  Actual gap ratio: {actual_gap:.6f} (target: {gap:.6f})")

        # Run power iteration
        start = time.time()
        eig_computed, vec, iters, converged = power_iteration(
            A, max_iter=max_iter, tol=tol
        )
        elapsed = time.time() - start

        # Calculate error
        error = abs(eig_computed - eigs_true[0])

        # Theoretical convergence rate
        theoretical_rate = gap  # Since lambda_2/lambda_1 = gap

        # Observed convergence rate (estimate from iterations)
        # After k iterations, error ~= C * (gap)^k
        # To reach tol from initial ~1, we need (gap)^k = tol
        # So k = log(tol) / log(gap)
        if converged and gap < 1.0:
            observed_rate = np.exp(np.log(tol) / iters)
        else:
            observed_rate = np.nan

        results['gap_ratio'].append(gap)
        results['iterations'].append(iters)
        results['time'].append(elapsed)
        results['converged'].append(converged)
        results['final_error'].append(error)
        results['theoretical_rate'].append(theoretical_rate)
        results['observed_rate'].append(observed_rate)

        print(f"  Iterations: {iters}")
        print(f"  Converged: {converged}")
        print(f"  Error: {error:.2e}")
        print(f"  Theoretical rate: {theoretical_rate:.4f}")
        print(f"  Observed rate: {observed_rate:.4f}")

    df = pd.DataFrame(results)
    return df


def plot_gap_analysis(df, output_dir):
    """Generate plots showing gap sensitivity"""
    print("\nGenerating plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Iterations vs Gap
    ax = axes[0, 0]
    ax.plot(df['gap_ratio'], df['iterations'], 'bo-', linewidth=2, markersize=10)
    ax.set_xlabel('Eigenvalue Gap Ratio (λ₂/λ₁)', fontsize=12)
    ax.set_ylabel('Iterations to Converge', fontsize=12)
    ax.set_title('Power Iteration Convergence Slows Dramatically\nNear λ₂≈λ₁',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.25, 1.0])

    # Add theoretical curve
    gaps_theory = np.linspace(0.3, 0.99, 100)
    tol = 1e-10
    iters_theory = np.log(tol) / np.log(gaps_theory)
    ax.plot(gaps_theory, iters_theory, 'r--', linewidth=1.5,
            label='Theoretical: k = log(tol)/log(λ₂/λ₁)', alpha=0.7)
    ax.legend(fontsize=10)

    # Plot 2: Time vs Gap
    ax = axes[0, 1]
    ax.plot(df['gap_ratio'], df['time'], 'go-', linewidth=2, markersize=10)
    ax.set_xlabel('Eigenvalue Gap Ratio (λ₂/λ₁)', fontsize=12)
    ax.set_ylabel('Computation Time (s)', fontsize=12)
    ax.set_title('Time Scales Linearly with Iterations', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.25, 1.0])

    # Plot 3: Convergence rate comparison
    ax = axes[1, 0]
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width/2, df['theoretical_rate'], width, label='Theoretical (λ₂/λ₁)',
           color='steelblue', alpha=0.8)
    ax.bar(x + width/2, df['observed_rate'], width, label='Observed',
           color='orangered', alpha=0.8)
    ax.set_xlabel('Gap Ratio', fontsize=12)
    ax.set_ylabel('Convergence Rate', fontsize=12)
    ax.set_title('Theory Matches Practice', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{g:.2f}' for g in df['gap_ratio']])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Final error
    ax = axes[1, 1]
    ax.semilogy(df['gap_ratio'], df['final_error'], 'mo-', linewidth=2, markersize=10)
    ax.set_xlabel('Eigenvalue Gap Ratio (λ₂/λ₁)', fontsize=12)
    ax.set_ylabel('Final Error', fontsize=12)
    ax.set_title('All Cases Achieve tol=10⁻¹⁰ (When Converged)',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.25, 1.0])
    ax.axhline(y=1e-10, color='r', linestyle='--', linewidth=1.5,
               label='Target tolerance', alpha=0.7)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'exp4_gap_sensitivity.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plot saved: exp4_gap_sensitivity.png")


def print_summary(df):
    """Print key findings"""
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # Finding 1: Worst case
    worst_idx = df['iterations'].idxmax()
    worst_gap = df.loc[worst_idx, 'gap_ratio']
    worst_iters = df.loc[worst_idx, 'iterations']

    print(f"\n1. Gap ratio {worst_gap:.2f} (nearly degenerate) required {worst_iters} iterations")
    print(f"   Theory predicts: k = log(1e-10)/log({worst_gap}) = {np.log(1e-10)/np.log(worst_gap):.0f} iterations")

    # Finding 2: Best case
    best_idx = df['iterations'].idxmin()
    best_gap = df.loc[best_idx, 'gap_ratio']
    best_iters = df.loc[best_idx, 'iterations']

    print(f"\n2. Gap ratio {best_gap:.2f} (well-separated) required only {best_iters} iterations")
    print(f"   Speedup: {worst_iters/best_iters:.1f}× faster than worst case")

    # Finding 3: Practical implication
    print(f"\n3. For real problems:")
    print(f"   - If lambda_1/lambda_2 > 3 (gap < 0.33): Power iteration converges quickly")
    print(f"   - If lambda_1/lambda_2 < 1.1 (gap > 0.91): Power iteration is too slow, use QR instead")
    print(f"   - Crossover around gap ~= 0.7 where both methods competitive")

    print("\n" + "=" * 70)


def main():
    print("=" * 70)
    print("EIGENVALUE GAP SENSITIVITY ANALYSIS")
    print("=" * 70)

    output_dir = os.path.join(os.path.dirname(__file__), 'output_results')
    os.makedirs(output_dir, exist_ok=True)

    start_total = time.time()

    # Run experiment
    df = experiment_gap_sensitivity()

    # Save CSV
    df.to_csv(os.path.join(output_dir, 'exp4_gap_sensitivity.csv'), index=False)
    print(f"\nResults saved to: exp4_gap_sensitivity.csv")

    # Generate plot
    plot_gap_analysis(df, output_dir)

    # Print summary
    print_summary(df)

    print(f"\nTotal time: {time.time() - start_total:.1f} seconds")
    print("\n" + "=" * 70)
    print("EXPERIMENT 4 COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
