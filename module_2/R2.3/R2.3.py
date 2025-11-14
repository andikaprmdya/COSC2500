#!/usr/bin/env python3
"""
COSC2500 Module 2, Exercise R2.3: Numerical Stability and Condition Numbers

This script analyzes the numerical stability of linear system solving using:
(a) Hilbert matrices H_ij = 1/(i+j-1) for n = 2, 5, 10
(b) Matrices A_ij = |i-j| + 1 for n = 100, 200, 300, 400, 500

For each matrix, we:
1. Create system Ax = b where x = ones(n) and b = Ax
2. Solve using naive Gaussian elimination and numpy.linalg.solve
3. Compute forward error, error magnification factor, and condition number

Author: COSC2500 Student
Date: 2025
"""

import numpy as np
import warnings
from typing import Tuple, List
import sys
import os

# Import the existing Gaussian elimination function
sys.path.append('..')  # Add parent directory to path
from gauselim import gauss_solve

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import matplotlib.patches as patches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Matplotlib not available. Plotting functionality will be disabled.")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class LinearSystemAnalyzer:
    """Comprehensive linear system stability analyzer."""

    def __init__(self):
        self.machine_eps = np.finfo(float).eps

    def create_hilbert_matrix(self, n: int) -> np.ndarray:
        """
        Create n×n Hilbert matrix with H_ij = 1/(i+j-1).

        Args:
            n: Size of the matrix

        Returns:
            n×n Hilbert matrix
        """
        H = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                H[i, j] = 1.0 / (i + j + 1)  # +1 because indices start from 0
        return H

    def create_matrix_b(self, n: int) -> np.ndarray:
        """
        Create n×n matrix with A_ij = |i-j| + 1.

        Args:
            n: Size of the matrix

        Returns:
            n×n matrix with specified pattern
        """
        A = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                A[i, j] = abs(i - j) + 1
        return A

    def naive_gaussian_elimination(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Solve Ax = b using naive Gaussian elimination (no pivoting).
        Uses the existing gauss_solve function from gauselim.py

        Args:
            A: Coefficient matrix
            b: Right-hand side vector

        Returns:
            Solution vector x
        """
        try:
            # Use the existing gauss_solve function
            x = gauss_solve(A, b)
            return x
        except Exception as e:
            # Re-raise with more specific error information
            raise ValueError(f"Gaussian elimination failed: {str(e)}")

    def compute_condition_number_inf(self, A: np.ndarray) -> float:
        """
        Compute the infinity norm condition number of matrix A.

        Args:
            A: Input matrix

        Returns:
            Condition number in infinity norm
        """
        try:
            # Use numpy's built-in condition number computation
            return np.linalg.cond(A, p=np.inf)
        except (np.linalg.LinAlgError, ValueError):
            return float('inf')  # Singular matrix

    def analyze_linear_system(self, A: np.ndarray, matrix_name: str, n: int) -> dict:
        """
        Comprehensive analysis of a linear system Ax = b.

        Args:
            A: Coefficient matrix
            matrix_name: Name of the matrix type for reporting
            n: Size of the matrix

        Returns:
            Dictionary containing all analysis results
        """
        print(f"\nAnalyzing {matrix_name} (n = {n})")
        print("-" * 60)

        # Create exact solution x = ones(n) and compute b = Ax
        x_exact = np.ones(n)
        b = A @ x_exact

        results = {
            'matrix_name': matrix_name,
            'n': n,
            'x_exact': x_exact,
            'b': b,
            'condition_number': None,
            'naive_gaussian': {},
            'numpy_solve': {},
            'analysis': {}
        }

        # Compute condition number
        try:
            cond_num = self.compute_condition_number_inf(A)
            results['condition_number'] = cond_num
            print(f"Condition number (inf-norm): {cond_num:.2e}")
        except Exception as e:
            print(f"Error computing condition number: {e}")
            results['condition_number'] = float('inf')

        # Method 1: Naive Gaussian Elimination
        print("\nMethod 1: Naive Gaussian Elimination")
        try:
            x_naive = self.naive_gaussian_elimination(A, b)

            # Compute forward error
            forward_error_naive = np.max(np.abs(x_exact - x_naive))

            # Compute error magnification factor
            relative_forward_error = forward_error_naive / np.max(np.abs(x_exact))
            error_magnification_naive = relative_forward_error / self.machine_eps

            results['naive_gaussian'] = {
                'solution': x_naive,
                'forward_error': forward_error_naive,
                'error_magnification': error_magnification_naive,
                'success': True
            }

            print(f"  Forward error ||x - x_computed||_inf: {forward_error_naive:.2e}")
            print(f"  Error magnification factor: {error_magnification_naive:.2e}")

            # Check for significant digits
            if forward_error_naive > 0:
                significant_digits_naive = max(0, -np.log10(forward_error_naive / np.max(np.abs(x_exact))))
                print(f"  Approximate significant digits: {significant_digits_naive:.1f}")
                results['naive_gaussian']['significant_digits'] = significant_digits_naive
            else:
                results['naive_gaussian']['significant_digits'] = 16  # Machine precision limit

        except Exception as e:
            print(f"  Error: {e}")
            results['naive_gaussian'] = {
                'solution': None,
                'forward_error': float('inf'),
                'error_magnification': float('inf'),
                'success': False,
                'error': str(e),
                'significant_digits': 0
            }

        # Method 2: NumPy's linalg.solve
        print("\nMethod 2: NumPy linalg.solve")
        try:
            x_numpy = np.linalg.solve(A, b)

            # Compute forward error
            forward_error_numpy = np.max(np.abs(x_exact - x_numpy))

            # Compute error magnification factor
            relative_forward_error = forward_error_numpy / np.max(np.abs(x_exact))
            error_magnification_numpy = relative_forward_error / self.machine_eps

            results['numpy_solve'] = {
                'solution': x_numpy,
                'forward_error': forward_error_numpy,
                'error_magnification': error_magnification_numpy,
                'success': True
            }

            print(f"  Forward error ||x - x_computed||_inf: {forward_error_numpy:.2e}")
            print(f"  Error magnification factor: {error_magnification_numpy:.2e}")

            # Check for significant digits
            if forward_error_numpy > 0:
                significant_digits_numpy = max(0, -np.log10(forward_error_numpy / np.max(np.abs(x_exact))))
                print(f"  Approximate significant digits: {significant_digits_numpy:.1f}")
                results['numpy_solve']['significant_digits'] = significant_digits_numpy
            else:
                results['numpy_solve']['significant_digits'] = 16  # Machine precision limit

        except Exception as e:
            print(f"  Error: {e}")
            results['numpy_solve'] = {
                'solution': None,
                'forward_error': float('inf'),
                'error_magnification': float('inf'),
                'success': False,
                'error': str(e),
                'significant_digits': 0
            }

        return results

    def analyze_hilbert_matrices(self) -> List[dict]:
        """Analyze Hilbert matrices for n = 2, 5, 10."""
        print("="*80)
        print("PART (A): HILBERT MATRICES ANALYSIS")
        print("="*80)
        print("Analyzing H_ij = 1/(i+j-1) for n = 2, 5, 10")

        results = []
        ns = [2, 5, 10]

        for n in ns:
            H = self.create_hilbert_matrix(n)
            result = self.analyze_linear_system(H, f"Hilbert Matrix", n)
            results.append(result)

        return results

    def analyze_matrix_b(self) -> List[dict]:
        """Analyze matrices A_ij = |i-j| + 1 for n = 100, 200, 300, 400, 500."""
        print("\n" + "="*80)
        print("PART (B): MATRICES A_ij = |i-j| + 1 ANALYSIS")
        print("="*80)
        print("Analyzing A_ij = |i-j| + 1 for n = 100, 200, 300, 400, 500")

        results = []
        ns = [100, 200, 300, 400, 500]

        for n in ns:
            A = self.create_matrix_b(n)
            result = self.analyze_linear_system(A, f"Matrix |i-j|+1", n)
            results.append(result)

        return results

    def create_summary_table(self, results_hilbert: List[dict], results_matrix_b: List[dict]):
        """Create comprehensive summary tables."""
        print("\n" + "="*120)
        print("COMPREHENSIVE SUMMARY TABLES")
        print("="*120)

        # Hilbert matrices summary
        print("\nTable 1: Hilbert Matrices (H_ij = 1/(i+j-1))")
        print("-" * 120)
        header = f"{'n':<3} {'Cond(A)':<12} {'Naive: Forward Error':<20} {'Naive: Error Mag':<15} {'Naive: Sig Digits':<15} {'NumPy: Forward Error':<20} {'NumPy: Error Mag':<15} {'NumPy: Sig Digits':<15}"
        print(header)
        print("-" * 120)

        for result in results_hilbert:
            n = result['n']
            cond = result['condition_number']

            # Naive Gaussian results
            naive = result['naive_gaussian']
            naive_fe = naive['forward_error'] if naive['success'] else float('inf')
            naive_em = naive['error_magnification'] if naive['success'] else float('inf')
            naive_sd = naive.get('significant_digits', 0) if naive['success'] else 0

            # NumPy results
            numpy = result['numpy_solve']
            numpy_fe = numpy['forward_error'] if numpy['success'] else float('inf')
            numpy_em = numpy['error_magnification'] if numpy['success'] else float('inf')
            numpy_sd = numpy.get('significant_digits', 0) if numpy['success'] else 0

            row = f"{n:<3} {cond:<12.2e} {naive_fe:<20.2e} {naive_em:<15.2e} {naive_sd:<15.1f} {numpy_fe:<20.2e} {numpy_em:<15.2e} {numpy_sd:<15.1f}"
            print(row)

        # Matrix B summary
        print("\nTable 2: Matrices A_ij = |i-j| + 1")
        print("-" * 120)
        print(header)
        print("-" * 120)

        for result in results_matrix_b:
            n = result['n']
            cond = result['condition_number']

            # Naive Gaussian results
            naive = result['naive_gaussian']
            naive_fe = naive['forward_error'] if naive['success'] else float('inf')
            naive_em = naive['error_magnification'] if naive['success'] else float('inf')
            naive_sd = naive.get('significant_digits', 0) if naive['success'] else 0

            # NumPy results
            numpy = result['numpy_solve']
            numpy_fe = numpy['forward_error'] if numpy['success'] else float('inf')
            numpy_em = numpy['error_magnification'] if numpy['success'] else float('inf')
            numpy_sd = numpy.get('significant_digits', 0) if numpy['success'] else 0

            row = f"{n:<3} {cond:<12.2e} {naive_fe:<20.2e} {naive_em:<15.2e} {naive_sd:<15.1f} {numpy_fe:<20.2e} {numpy_em:<15.2e} {numpy_sd:<15.1f}"
            print(row)

    def analyze_significant_digits(self, results_hilbert: List[dict], results_matrix_b: List[dict]):
        """Analyze and report cases with no correct significant digits."""
        print("\n" + "="*80)
        print("PART (C): ANALYSIS OF SIGNIFICANT DIGITS")
        print("="*80)

        print("Cases with no correct significant digits (< 1 significant digit):")
        print("-" * 80)

        no_digits_found = False

        # Check Hilbert matrices
        print("\nHilbert Matrices:")
        for result in results_hilbert:
            n = result['n']
            naive_sd = result['naive_gaussian'].get('significant_digits', 0)
            numpy_sd = result['numpy_solve'].get('significant_digits', 0)

            if naive_sd < 1 or numpy_sd < 1:
                print(f"  n = {n}:")
                if naive_sd < 1:
                    print(f"    Naive Gaussian: {naive_sd:.1f} significant digits")
                if numpy_sd < 1:
                    print(f"    NumPy solve: {numpy_sd:.1f} significant digits")
                no_digits_found = True

        # Check matrix B
        print("\nMatrices A_ij = |i-j| + 1:")
        for result in results_matrix_b:
            n = result['n']
            naive_sd = result['naive_gaussian'].get('significant_digits', 0)
            numpy_sd = result['numpy_solve'].get('significant_digits', 0)

            if naive_sd < 1 or numpy_sd < 1:
                print(f"  n = {n}:")
                if naive_sd < 1:
                    print(f"    Naive Gaussian: {naive_sd:.1f} significant digits")
                if numpy_sd < 1:
                    print(f"    NumPy solve: {numpy_sd:.1f} significant digits")
                no_digits_found = True

        if not no_digits_found:
            print("All tested cases maintain at least 1 correct significant digit.")

        # Additional analysis
        print("\nKey Observations:")
        print("-" * 40)

        # Find worst condition numbers
        worst_hilbert = max(results_hilbert, key=lambda x: x['condition_number'])
        worst_matrix_b = max(results_matrix_b, key=lambda x: x['condition_number'])

        print(f"Worst conditioned Hilbert matrix: n = {worst_hilbert['n']}, cond = {worst_hilbert['condition_number']:.2e}")
        print(f"Worst conditioned |i-j|+1 matrix: n = {worst_matrix_b['n']}, cond = {worst_matrix_b['condition_number']:.2e}")

        # Relationship between condition number and accuracy
        print(f"\nCondition Number vs Accuracy Relationship:")
        print(f"- Hilbert matrices show exponential growth in condition number with n")
        print(f"- Matrix |i-j|+1 shows more moderate condition number growth")
        print(f"- Error magnification ~= condition_number * machine_epsilon (theoretical bound)")
        print(f"- NumPy's solver generally outperforms naive Gaussian elimination")

    def create_comprehensive_visualizations(self, results_hilbert: List[dict], results_matrix_b: List[dict]):
        """Create comprehensive visualizations of the numerical stability analysis."""
        if not MATPLOTLIB_AVAILABLE:
            print("\nMatplotlib not available. Skipping visualizations.")
            return

        print("Creating multiple visualization windows for better readability...")

        # Window 1: Primary Analysis (3 plots)
        self.create_primary_analysis_plots(results_hilbert, results_matrix_b)

        # Window 2: Error Analysis (3 plots)
        self.create_error_analysis_plots(results_hilbert, results_matrix_b)

        # Window 3: Advanced Analysis (3 plots)
        self.create_advanced_analysis_plots(results_hilbert, results_matrix_b)

    def create_primary_analysis_plots(self, results_hilbert: List[dict], results_matrix_b: List[dict]):
        """Create primary analysis plots (Window 1)."""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        # Plot 1: Condition Number Growth
        self.plot_condition_number_growth(ax1, results_hilbert, results_matrix_b)

        # Plot 2: Forward Error Comparison
        self.plot_forward_error_comparison(ax2, results_hilbert, results_matrix_b)

        # Plot 3: Significant Digits Analysis
        self.plot_significant_digits(ax3, results_hilbert, results_matrix_b)

        plt.tight_layout()
        plt.suptitle('Primary Numerical Stability Analysis',
                     fontsize=14, fontweight='bold', y=0.98)

        # Save the plot
        plt.savefig('R2_3_primary_analysis.png', dpi=300, bbox_inches='tight')
        print("  Saved: R2_3_primary_analysis.png")
        plt.close()

    def create_error_analysis_plots(self, results_hilbert: List[dict], results_matrix_b: List[dict]):
        """Create error analysis plots (Window 2)."""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        # Plot 1: Error Magnification Factor
        self.plot_error_magnification(ax1, results_hilbert, results_matrix_b)

        # Plot 2: Condition Number vs Error Relationship
        self.plot_condition_vs_error(ax2, results_hilbert, results_matrix_b)

        # Plot 3: Theoretical vs Actual Error
        self.plot_theoretical_vs_actual(ax3, results_hilbert, results_matrix_b)

        plt.tight_layout()
        plt.suptitle('Error Analysis and Theoretical Validation',
                     fontsize=14, fontweight='bold', y=0.98)

        # Save the plot
        plt.savefig('R2_3_error_analysis.png', dpi=300, bbox_inches='tight')
        print("  Saved: R2_3_error_analysis.png")
        plt.close()

    def create_advanced_analysis_plots(self, results_hilbert: List[dict], results_matrix_b: List[dict]):
        """Create advanced analysis plots (Window 3)."""
        fig = plt.figure(figsize=(18, 6))

        # Plot 1: Method Performance Comparison
        ax1 = plt.subplot(1, 3, 1)
        self.plot_method_comparison(ax1, results_hilbert, results_matrix_b)

        # Plot 2: Matrix Condition Heatmap
        ax2 = plt.subplot(1, 3, 2)
        self.plot_matrix_condition_heatmap(ax2, results_hilbert, results_matrix_b)

        # Plot 3: Summary Statistics
        ax3 = plt.subplot(1, 3, 3)
        self.plot_summary_statistics(ax3, results_hilbert, results_matrix_b)

        plt.tight_layout()
        plt.suptitle('Method Comparison and Summary Analysis',
                     fontsize=14, fontweight='bold', y=0.98)

        # Save the plot
        plt.savefig('R2_3_advanced_analysis.png', dpi=300, bbox_inches='tight')
        print("  Saved: R2_3_advanced_analysis.png")
        plt.close()

    def plot_condition_number_growth(self, ax, results_hilbert, results_matrix_b):
        """Plot condition number growth for both matrix types."""
        # Hilbert matrices
        hilbert_n = [r['n'] for r in results_hilbert]
        hilbert_cond = [r['condition_number'] for r in results_hilbert]

        # Matrix B
        matrix_b_n = [r['n'] for r in results_matrix_b]
        matrix_b_cond = [r['condition_number'] for r in results_matrix_b]

        ax.semilogy(hilbert_n, hilbert_cond, 'ro-', linewidth=2, markersize=8,
                   label='Hilbert Matrix', markerfacecolor='red')
        ax.semilogy(matrix_b_n, matrix_b_cond, 'bs-', linewidth=2, markersize=8,
                   label='Matrix |i-j|+1', markerfacecolor='blue')

        ax.set_xlabel('Matrix Size (n)')
        ax.set_ylabel('Condition Number (log scale)')
        ax.set_title('Condition Number Growth')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add annotations for key points
        for i, (n, cond) in enumerate(zip(hilbert_n, hilbert_cond)):
            ax.annotate(f'{cond:.1e}', (n, cond), xytext=(5, 5),
                       textcoords='offset points', fontsize=8)

    def plot_forward_error_comparison(self, ax, results_hilbert, results_matrix_b):
        """Plot forward error comparison between methods."""
        all_results = results_hilbert + results_matrix_b

        n_values = [r['n'] for r in all_results]
        naive_errors = [r['naive_gaussian']['forward_error'] if r['naive_gaussian']['success'] else np.nan
                       for r in all_results]
        numpy_errors = [r['numpy_solve']['forward_error'] if r['numpy_solve']['success'] else np.nan
                       for r in all_results]

        # Separate Hilbert and Matrix B
        hilbert_count = len(results_hilbert)

        # Plot Hilbert results
        ax.semilogy(n_values[:hilbert_count], naive_errors[:hilbert_count],
                   'ro-', label='Naive (Hilbert)', markersize=8)
        ax.semilogy(n_values[:hilbert_count], numpy_errors[:hilbert_count],
                   'r^-', label='NumPy (Hilbert)', markersize=8)

        # Plot Matrix B results
        ax.semilogy(n_values[hilbert_count:], naive_errors[hilbert_count:],
                   'bo-', label='Naive (|i-j|+1)', markersize=6)
        ax.semilogy(n_values[hilbert_count:], numpy_errors[hilbert_count:],
                   'b^-', label='NumPy (|i-j|+1)', markersize=6)

        ax.set_xlabel('Matrix Size (n)')
        ax.set_ylabel('Forward Error (log scale)')
        ax.set_title('Forward Error Comparison')
        ax.grid(True, alpha=0.3)
        ax.legend()

    def plot_error_magnification(self, ax, results_hilbert, results_matrix_b):
        """Plot error magnification factors."""
        all_results = results_hilbert + results_matrix_b

        n_values = [r['n'] for r in all_results]
        naive_mag = [r['naive_gaussian']['error_magnification'] if r['naive_gaussian']['success'] else np.nan
                    for r in all_results]
        numpy_mag = [r['numpy_solve']['error_magnification'] if r['numpy_solve']['success'] else np.nan
                    for r in all_results]

        hilbert_count = len(results_hilbert)

        # Create bar chart
        x_pos = np.arange(len(n_values))
        width = 0.35

        bars1 = ax.bar(x_pos - width/2, np.log10(naive_mag), width,
                      label='Naive Gaussian', alpha=0.7, color='orange')
        bars2 = ax.bar(x_pos + width/2, np.log10(numpy_mag), width,
                      label='NumPy solve', alpha=0.7, color='green')

        ax.set_xlabel('Matrix Cases')
        ax.set_ylabel('Error Magnification (log₁₀)')
        ax.set_title('Error Magnification Factor')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'n={n}' for n in n_values], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    def plot_significant_digits(self, ax, results_hilbert, results_matrix_b):
        """Plot significant digits retained."""
        all_results = results_hilbert + results_matrix_b
        matrix_types = ['Hilbert']*len(results_hilbert) + ['|i-j|+1']*len(results_matrix_b)

        n_values = [r['n'] for r in all_results]
        naive_digits = [r['naive_gaussian'].get('significant_digits', 0) if r['naive_gaussian']['success'] else 0
                       for r in all_results]
        numpy_digits = [r['numpy_solve'].get('significant_digits', 0) if r['numpy_solve']['success'] else 0
                       for r in all_results]

        x_pos = np.arange(len(n_values))
        width = 0.35

        colors_naive = ['red' if mt == 'Hilbert' else 'blue' for mt in matrix_types]
        colors_numpy = ['darkred' if mt == 'Hilbert' else 'darkblue' for mt in matrix_types]

        bars1 = ax.bar(x_pos - width/2, naive_digits, width,
                      label='Naive Gaussian', alpha=0.7, color=colors_naive)
        bars2 = ax.bar(x_pos + width/2, numpy_digits, width,
                      label='NumPy solve', alpha=0.7, color=colors_numpy)

        # Add horizontal line at 1 significant digit threshold
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.7,
                  label='Loss of accuracy threshold')

        ax.set_xlabel('Matrix Cases')
        ax.set_ylabel('Significant Digits Retained')
        ax.set_title('Significant Digits Analysis')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'{mt}\nn={n}' for mt, n in zip(matrix_types, n_values)], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    def plot_method_comparison(self, ax, results_hilbert, results_matrix_b):
        """Plot method performance comparison."""
        # Calculate success rates and average errors
        hilbert_naive_success = sum(1 for r in results_hilbert if r['naive_gaussian']['success'])
        hilbert_numpy_success = sum(1 for r in results_hilbert if r['numpy_solve']['success'])
        matrix_b_naive_success = sum(1 for r in results_matrix_b if r['naive_gaussian']['success'])
        matrix_b_numpy_success = sum(1 for r in results_matrix_b if r['numpy_solve']['success'])

        categories = ['Hilbert\nNaive', 'Hilbert\nNumPy', '|i-j|+1\nNaive', '|i-j|+1\nNumPy']
        success_rates = [
            hilbert_naive_success / len(results_hilbert) * 100,
            hilbert_numpy_success / len(results_hilbert) * 100,
            matrix_b_naive_success / len(results_matrix_b) * 100,
            matrix_b_numpy_success / len(results_matrix_b) * 100
        ]

        colors = ['lightcoral', 'darkred', 'lightblue', 'darkblue']
        bars = ax.bar(categories, success_rates, color=colors, alpha=0.7)

        # Add value labels on bars
        for bar, rate in zip(bars, success_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{rate:.0f}%', ha='center', va='bottom', fontweight='bold')

        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Method Success Rates')
        ax.set_ylim(0, 110)
        ax.grid(True, alpha=0.3, axis='y')

    def plot_condition_vs_error(self, ax, results_hilbert, results_matrix_b):
        """Plot condition number vs error relationship."""
        all_results = results_hilbert + results_matrix_b

        cond_numbers = [r['condition_number'] for r in all_results]
        naive_errors = [r['naive_gaussian']['forward_error'] if r['naive_gaussian']['success'] else np.nan
                       for r in all_results]
        numpy_errors = [r['numpy_solve']['forward_error'] if r['numpy_solve']['success'] else np.nan
                       for r in all_results]

        # Filter out invalid values
        valid_naive = [(c, e) for c, e in zip(cond_numbers, naive_errors) if not np.isnan(e)]
        valid_numpy = [(c, e) for c, e in zip(cond_numbers, numpy_errors) if not np.isnan(e)]

        if valid_naive:
            cond_naive, err_naive = zip(*valid_naive)
            ax.loglog(cond_naive, err_naive, 'ro', markersize=8, label='Naive Gaussian')

        if valid_numpy:
            cond_numpy, err_numpy = zip(*valid_numpy)
            ax.loglog(cond_numpy, err_numpy, 'bs', markersize=8, label='NumPy solve')

        # Add theoretical line: error ~ condition_number * machine_epsilon
        if valid_naive or valid_numpy:
            all_conds = list(cond_numbers)
            theoretical_error = [c * self.machine_eps for c in all_conds]
            ax.loglog(all_conds, theoretical_error, 'g--', alpha=0.7,
                     label='Theoretical: κ(A)·ε')

        ax.set_xlabel('Condition Number')
        ax.set_ylabel('Forward Error')
        ax.set_title('Condition Number vs Error')
        ax.grid(True, alpha=0.3)
        ax.legend()

    def plot_matrix_condition_heatmap(self, ax, results_hilbert, results_matrix_b):
        """Create a heatmap showing matrix conditioning severity."""
        # Create a visualization of condition number severity
        matrix_types = ['Hilbert'] * len(results_hilbert) + ['|i-j|+1'] * len(results_matrix_b)
        n_values = [r['n'] for r in results_hilbert + results_matrix_b]
        cond_values = [r['condition_number'] for r in results_hilbert + results_matrix_b]

        # Create a 2D representation
        unique_types = list(set(matrix_types))
        unique_n = sorted(list(set(n_values)))

        # Create grid for heatmap
        grid = np.zeros((len(unique_types), len(unique_n)))

        for i, result in enumerate(results_hilbert + results_matrix_b):
            type_idx = unique_types.index(matrix_types[i])
            n_idx = unique_n.index(result['n'])
            grid[type_idx, n_idx] = np.log10(result['condition_number'])

        im = ax.imshow(grid, cmap='Reds', aspect='auto')

        # Set labels
        ax.set_xticks(range(len(unique_n)))
        ax.set_xticklabels(unique_n)
        ax.set_yticks(range(len(unique_types)))
        ax.set_yticklabels(unique_types)
        ax.set_xlabel('Matrix Size (n)')
        ax.set_title('Condition Number Severity\n(log₁₀ scale)')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('log₁₀(Condition Number)')

        # Add text annotations
        for i in range(len(unique_types)):
            for j in range(len(unique_n)):
                text = ax.text(j, i, f'{grid[i, j]:.1f}',
                             ha="center", va="center", color="black", fontweight='bold')

    def plot_theoretical_vs_actual(self, ax, results_hilbert, results_matrix_b):
        """Plot theoretical vs actual error bounds."""
        all_results = results_hilbert + results_matrix_b

        # Calculate theoretical and actual error bounds
        theoretical_bounds = [r['condition_number'] * self.machine_eps for r in all_results]
        actual_naive = [r['naive_gaussian']['forward_error'] if r['naive_gaussian']['success'] else np.nan
                       for r in all_results]
        actual_numpy = [r['numpy_solve']['forward_error'] if r['numpy_solve']['success'] else np.nan
                       for r in all_results]

        x_pos = np.arange(len(all_results))
        width = 0.25

        ax.semilogy(x_pos - width, theoretical_bounds, 'go-', label='Theoretical Bound', markersize=6)
        ax.semilogy(x_pos, actual_naive, 'ro-', label='Naive Gaussian', markersize=6)
        ax.semilogy(x_pos + width, actual_numpy, 'bo-', label='NumPy solve', markersize=6)

        labels = [f'n={r["n"]}' for r in all_results]
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45)
        ax.set_ylabel('Error (log scale)')
        ax.set_title('Theoretical vs Actual Error')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def plot_summary_statistics(self, ax, results_hilbert, results_matrix_b):
        """Plot summary statistics."""
        ax.axis('off')  # Turn off axis for text-based summary

        # Calculate summary statistics
        hilbert_worst_cond = max(r['condition_number'] for r in results_hilbert)
        hilbert_best_digits = max(min(r['naive_gaussian'].get('significant_digits', 0),
                                     r['numpy_solve'].get('significant_digits', 0))
                                 for r in results_hilbert)

        matrix_b_worst_cond = max(r['condition_number'] for r in results_matrix_b)
        matrix_b_best_digits = max(min(r['naive_gaussian'].get('significant_digits', 0),
                                      r['numpy_solve'].get('significant_digits', 0))
                                  for r in results_matrix_b)

        summary_text = f"""NUMERICAL STABILITY SUMMARY

Hilbert Matrices:
• Worst condition number: {hilbert_worst_cond:.1e}
• Best significant digits: {hilbert_best_digits:.1f}
• Exponential conditioning growth

Matrix |i-j|+1:
• Worst condition number: {matrix_b_worst_cond:.1e}
• Best significant digits: {matrix_b_best_digits:.1f}
• Polynomial conditioning growth

Key Insights:
• NumPy generally outperforms naive method
• Condition number predicts accuracy loss
• Hilbert matrices are severely ill-conditioned
• Matrix structure affects numerical stability

Recommendations:
• Use pivoting for stability
• Monitor condition numbers
• Consider iterative refinement
• Apply regularization for ill-conditioned systems"""

        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

def run_comprehensive_analysis():
    """Main function to run the comprehensive linear system analysis."""
    print("COSC2500 Module 2, Exercise R2.3: Numerical Stability Analysis")
    print("="*80)
    print(f"Machine epsilon: {np.finfo(float).eps:.2e}")
    print("="*80)

    # Initialize analyzer
    analyzer = LinearSystemAnalyzer()

    # Run analyses
    results_hilbert = analyzer.analyze_hilbert_matrices()
    results_matrix_b = analyzer.analyze_matrix_b()

    # Create summary tables
    analyzer.create_summary_table(results_hilbert, results_matrix_b)

    # Analyze significant digits
    analyzer.analyze_significant_digits(results_hilbert, results_matrix_b)

    # Generate comprehensive visualizations
    print("\n" + "="*80)
    print("GENERATING COMPREHENSIVE VISUALIZATIONS")
    print("="*80)
    analyzer.create_comprehensive_visualizations(results_hilbert, results_matrix_b)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("Key Findings:")
    print("1. Hilbert matrices become severely ill-conditioned as n increases")
    print("2. Matrix |i-j|+1 maintains reasonable conditioning for tested sizes")
    print("3. NumPy's solver (with pivoting) generally outperforms naive Gaussian elimination")
    print("4. Error magnification factor relates directly to condition number")
    print("5. Condition number predicts loss of significant digits in solutions")

    return results_hilbert, results_matrix_b

if __name__ == "__main__":
    # Run the comprehensive analysis
    hilbert_results, matrix_b_results = run_comprehensive_analysis()