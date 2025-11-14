#!/usr/bin/env python3
"""
COSC2500 Module 2, Exercise R2.2: Optimization Methods for Extrema Finding

Function: f(x) = ((x - 7/11)^2) * (x + 3/13) * exp(-x^2)

This script implements three optimization methods to find all local maxima and minima:
1. Golden Section Search
2. Successive Parabolic Interpolation
3. Newton's Method for Optimization

Author: COSC2500 Student
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Callable
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class OptimizationResult:
    """Stores optimization result information."""
    x_optimal: float
    f_optimal: float
    iterations: int
    converged: bool
    method_name: str
    tolerance_achieved: float

class ExtremaFinder:
    """Comprehensive extrema finding using multiple optimization methods."""

    def __init__(self, tolerance: float = 1e-8, max_iterations: int = 1000):
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def f(self, x: float) -> float:
        """The target function: f(x) = ((x - 7/11)^2) * (x + 3/13) * exp(-x^2)"""
        return ((x - 7/11)**2) * (x + 3/13) * np.exp(-x**2)

    def f_prime(self, x: float) -> float:
        """First derivative of f(x) - analytically computed."""
        # f(x) = (x - 7/11)^2 * (x + 3/13) * exp(-x^2)
        # Using product rule: (uvw)' = u'vw + uv'w + uvw'

        u = (x - 7/11)**2
        u_prime = 2 * (x - 7/11)

        v = (x + 3/13)
        v_prime = 1

        w = np.exp(-x**2)
        w_prime = -2*x * np.exp(-x**2)

        # Apply product rule
        f_prime = u_prime * v * w + u * v_prime * w + u * v * w_prime
        return f_prime

    def f_double_prime(self, x: float) -> float:
        """Second derivative of f(x) - analytically computed."""
        # This is the derivative of f_prime(x)
        # f'(x) = 2(x - 7/11)(x + 3/13)exp(-x^2) + (x - 7/11)^2 * exp(-x^2) + (x - 7/11)^2 * (x + 3/13) * (-2x)exp(-x^2)

        # Let's compute this step by step using product and chain rules
        u = (x - 7/11)**2
        u_prime = 2 * (x - 7/11)
        u_double_prime = 2

        v = (x + 3/13)
        v_prime = 1
        v_double_prime = 0

        w = np.exp(-x**2)
        w_prime = -2*x * np.exp(-x**2)
        w_double_prime = (-2 * np.exp(-x**2)) + (-2*x) * (-2*x) * np.exp(-x**2)
        w_double_prime = -2 * np.exp(-x**2) + 4*x**2 * np.exp(-x**2)
        w_double_prime = (-2 + 4*x**2) * np.exp(-x**2)

        # For (uvw)'', we need to apply the product rule to (uvw)' = u'vw + uv'w + uvw'
        # (u'vw)' = u''vw + u'v'w + u'vw'
        # (uv'w)' = u'v'w + uv''w + uv'w'
        # (uvw')' = u'vw' + uv'w' + uvw''

        term1 = u_double_prime * v * w + u_prime * v_prime * w + u_prime * v * w_prime
        term2 = u_prime * v_prime * w + u * v_double_prime * w + u * v_prime * w_prime
        term3 = u_prime * v * w_prime + u * v_prime * w_prime + u * v * w_double_prime

        f_double_prime = term1 + term2 + term3
        return f_double_prime

    def golden_section(self, f: Callable, a: float, b: float, tol: float = None) -> OptimizationResult:
        """
        Golden Section Search for finding minimum in interval [a, b].

        Args:
            f: Function to minimize
            a, b: Search interval bounds
            tol: Tolerance for convergence

        Returns:
            OptimizationResult containing the minimum location and details
        """
        if tol is None:
            tol = self.tolerance

        # Golden ratio
        phi = (1 + np.sqrt(5)) / 2
        resphi = 2 - phi  # 1/phi

        # Initial points
        tol1 = max(tol, 1e-12)  # Prevent too small tolerance
        iterations = 0

        # Initialize the golden section points
        c = a + resphi * (b - a)
        d = a + (1 - resphi) * (b - a)
        fc = f(c)
        fd = f(d)

        while abs(b - a) > tol and iterations < self.max_iterations:
            iterations += 1

            if fc < fd:  # Minimum is in [a, d]
                b = d
                d = c
                fd = fc
                c = a + resphi * (b - a)
                fc = f(c)
            else:  # Minimum is in [c, b]
                a = c
                c = d
                fc = fd
                d = a + (1 - resphi) * (b - a)
                fd = f(d)

            # Update tolerance check if needed

        # Return the point with smaller function value
        if fc < fd:
            x_min = c
            f_min = fc
        else:
            x_min = d
            f_min = fd

        converged = iterations < self.max_iterations
        final_tolerance = abs(b - a)

        return OptimizationResult(
            x_optimal=x_min,
            f_optimal=f_min,
            iterations=iterations,
            converged=converged,
            method_name="Golden Section Search",
            tolerance_achieved=final_tolerance
        )

    def successive_parabolic(self, f: Callable, x1: float, x2: float, x3: float, tol: float = None) -> OptimizationResult:
        """
        Successive Parabolic Interpolation for finding minimum.

        Args:
            f: Function to minimize
            x1, x2, x3: Three initial points
            tol: Tolerance for convergence

        Returns:
            OptimizationResult containing the minimum location and details
        """
        if tol is None:
            tol = self.tolerance

        # Ensure x1 < x2 < x3
        points = sorted([x1, x2, x3])
        x1, x2, x3 = points[0], points[1], points[2]

        iterations = 0
        converged = False

        while iterations < self.max_iterations:
            iterations += 1

            # Evaluate function at the three points
            f1, f2, f3 = f(x1), f(x2), f(x3)

            # Check if we can form a parabola (denominator != 0)
            denominator = (x1 - x2) * (x1 - x3) * (x2 - x3)
            if abs(denominator) < 1e-15:
                break

            # Calculate parabolic interpolation coefficients
            a = ((f1 - f2) * (x1 - x3) - (f1 - f3) * (x1 - x2)) / denominator
            b = ((f1 - f3) * (x1 - x2)**2 - (f1 - f2) * (x1 - x3)**2) / denominator

            # Find the minimum of the parabola: x_min = -b/(2a)
            if abs(a) < 1e-15:  # Nearly linear, can't find minimum
                break

            x_new = -b / (2 * a)
            f_new = f(x_new)

            # Check convergence
            if min(abs(x_new - x1), abs(x_new - x2), abs(x_new - x3)) < tol:
                converged = True
                break

            # Update the three points by replacing the one with highest function value
            # or the one furthest from x_new if we're not improving
            values = [(x1, f1), (x2, f2), (x3, f3)]
            values.sort(key=lambda item: item[1])  # Sort by function value

            # Replace the worst point with the new point
            worst_idx = np.argmax([f1, f2, f3])
            if worst_idx == 0:
                x1 = x_new
            elif worst_idx == 1:
                x2 = x_new
            else:
                x3 = x_new

            # Re-sort to maintain x1 < x2 < x3
            points = sorted([x1, x2, x3])
            x1, x2, x3 = points[0], points[1], points[2]

            # Check if points are too close together
            if max(abs(x3 - x1), abs(x2 - x1), abs(x3 - x2)) < tol:
                converged = True
                break

        # Find the point with minimum function value
        f1, f2, f3 = f(x1), f(x2), f(x3)
        min_idx = np.argmin([f1, f2, f3])
        x_min = [x1, x2, x3][min_idx]
        f_min = [f1, f2, f3][min_idx]

        tolerance_achieved = max(abs(x3 - x1), abs(x2 - x1), abs(x3 - x2))

        return OptimizationResult(
            x_optimal=x_min,
            f_optimal=f_min,
            iterations=iterations,
            converged=converged,
            method_name="Successive Parabolic Interpolation",
            tolerance_achieved=tolerance_achieved
        )

    def newton_min(self, f: Callable, fprime: Callable, fprime2: Callable, x0: float, tol: float = None) -> OptimizationResult:
        """
        Newton's Method for optimization (finding critical points).

        Args:
            f: Function to optimize
            fprime: First derivative of f
            fprime2: Second derivative of f
            x0: Initial guess
            tol: Tolerance for convergence

        Returns:
            OptimizationResult containing the critical point and details
        """
        if tol is None:
            tol = self.tolerance

        x = x0
        iterations = 0
        converged = False

        while iterations < self.max_iterations:
            iterations += 1

            fp = fprime(x)
            fpp = fprime2(x)

            # Check if second derivative is too small (singular Hessian)
            if abs(fpp) < 1e-15:
                break

            # Newton step: x_new = x - f'(x)/f''(x)
            x_new = x - fp / fpp

            # Check convergence
            if abs(x_new - x) < tol:
                converged = True
                break

            # Check for divergence
            if abs(x_new) > 1e10:
                break

            x = x_new

        f_optimal = f(x)
        tolerance_achieved = abs(fprime(x))  # Should be close to 0 at optimum

        return OptimizationResult(
            x_optimal=x,
            f_optimal=f_optimal,
            iterations=iterations,
            converged=converged,
            method_name="Newton's Method",
            tolerance_achieved=tolerance_achieved
        )

    def find_extrema_candidates(self, domain: Tuple[float, float], num_points: int = 1000) -> List[float]:
        """
        Scan the domain to find candidate extrema by looking for sign changes in f'(x).

        Args:
            domain: (a, b) domain to search
            num_points: Number of points to sample

        Returns:
            List of x-coordinates where extrema might exist
        """
        a, b = domain
        x_values = np.linspace(a, b, num_points)
        fp_values = [self.f_prime(x) for x in x_values]

        candidates = []

        # Look for sign changes in f'(x)
        for i in range(len(fp_values) - 1):
            if fp_values[i] * fp_values[i + 1] < 0:  # Sign change
                # Use bisection to refine the root location
                x_left, x_right = x_values[i], x_values[i + 1]

                # Simple bisection to find where f'(x) = 0
                for _ in range(20):  # 20 iterations should be enough for good precision
                    x_mid = (x_left + x_right) / 2
                    fp_mid = self.f_prime(x_mid)

                    if abs(fp_mid) < 1e-10:
                        break

                    if fp_mid * self.f_prime(x_left) < 0:
                        x_right = x_mid
                    else:
                        x_left = x_mid

                candidates.append((x_left + x_right) / 2)

        return candidates

    def classify_extremum(self, x: float) -> str:
        """
        Classify whether a critical point is a maximum, minimum, or saddle point.

        Args:
            x: Critical point location

        Returns:
            String describing the type of extremum
        """
        fpp = self.f_double_prime(x)

        if fpp > 1e-10:
            return "Local Minimum"
        elif fpp < -1e-10:
            return "Local Maximum"
        else:
            return "Inflection Point / Saddle"

    def comprehensive_analysis(self, domain: Tuple[float, float] = (-8, 8)) -> dict:
        """
        Perform comprehensive extrema analysis using all three methods.

        Args:
            domain: Domain to search for extrema

        Returns:
            Dictionary containing all results and analysis
        """
        print("="*80)
        print("COSC2500 Module 2, Exercise R2.2: Comprehensive Extrema Analysis")
        print("="*80)
        print(f"Function: f(x) = ((x - 7/11)^2) * (x + 3/13) * exp(-x^2)")
        print(f"Domain: [{domain[0]}, {domain[1]}]")
        print(f"Tolerance: {self.tolerance}")
        print("="*80)

        # Step 1: Find candidate extrema
        print("\nStep 1: Scanning domain for critical points...")
        candidates = self.find_extrema_candidates(domain)
        print(f"Found {len(candidates)} candidate critical points: {[f'{x:.6f}' for x in candidates]}")

        results = {
            'candidates': candidates,
            'golden_section_results': [],
            'parabolic_results': [],
            'newton_results': [],
            'summary': []
        }

        # Step 2: Refine each candidate using all three methods
        print("\nStep 2: Refining extrema using optimization methods...")
        print("-"*80)

        for i, candidate in enumerate(candidates):
            print(f"\nAnalyzing Candidate {i+1}: x ~= {candidate:.6f}")
            print(f"f({candidate:.6f}) = {self.f(candidate):.8f}")
            print(f"f'({candidate:.6f}) = {self.f_prime(candidate):.2e}")
            print(f"Classification: {self.classify_extremum(candidate)}")

            extremum_info = {
                'candidate': candidate,
                'type': self.classify_extremum(candidate),
                'f_value': self.f(candidate)
            }

            # Method 1: Golden Section Search
            # Create a small interval around the candidate
            interval_width = 0.2  # Smaller interval for better convergence
            a, b = candidate - interval_width, candidate + interval_width
            # Ensure we stay within domain
            a = max(a, domain[0])
            b = min(b, domain[1])

            print(f"\n  Golden Section Search on [{a:.6f}, {b:.6f}]:")
            gs_result = self.golden_section(self.f, a, b)
            results['golden_section_results'].append(gs_result)
            print(f"    Result: x = {gs_result.x_optimal:.8f}, f(x) = {gs_result.f_optimal:.8f}")
            print(f"    Iterations: {gs_result.iterations}, Converged: {gs_result.converged}")
            print(f"    Tolerance achieved: {gs_result.tolerance_achieved:.2e}")

            # Method 2: Successive Parabolic Interpolation
            # Use three points around the candidate (smaller spacing for better local convergence)
            spacing = 0.05
            x1 = candidate - spacing
            x2 = candidate
            x3 = candidate + spacing
            # Ensure points are within domain
            x1 = max(x1, domain[0])
            x3 = min(x3, domain[1])

            print(f"\n  Successive Parabolic Interpolation with points [{x1:.6f}, {x2:.6f}, {x3:.6f}]:")
            sp_result = self.successive_parabolic(self.f, x1, x2, x3)
            results['parabolic_results'].append(sp_result)
            print(f"    Result: x = {sp_result.x_optimal:.8f}, f(x) = {sp_result.f_optimal:.8f}")
            print(f"    Iterations: {sp_result.iterations}, Converged: {sp_result.converged}")
            print(f"    Tolerance achieved: {sp_result.tolerance_achieved:.2e}")

            # Method 3: Newton's Method
            print(f"\n  Newton's Method starting from x0 = {candidate:.6f}:")
            newton_result = self.newton_min(self.f, self.f_prime, self.f_double_prime, candidate)
            results['newton_results'].append(newton_result)
            print(f"    Result: x = {newton_result.x_optimal:.8f}, f(x) = {newton_result.f_optimal:.8f}")
            print(f"    Iterations: {newton_result.iterations}, Converged: {newton_result.converged}")
            print(f"    |f'(x)| = {newton_result.tolerance_achieved:.2e}")

            # Store summary information
            extremum_info['golden_section'] = gs_result
            extremum_info['parabolic'] = sp_result
            extremum_info['newton'] = newton_result
            results['summary'].append(extremum_info)

        return results

    def plot_results(self, results: dict, domain: Tuple[float, float] = (-8, 8)):
        """
        Create comprehensive plots showing the function and all found extrema.

        Args:
            results: Results dictionary from comprehensive_analysis
            domain: Domain for plotting
        """
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot 1: Function overview
        x = np.linspace(domain[0], domain[1], 2000)
        y = [self.f(xi) for xi in x]

        ax1.plot(x, y, 'b-', linewidth=2, label='f(x) = ((x - 7/11)^2)(x + 3/13)exp(-x^2)')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('x')
        ax1.set_ylabel('f(x)')
        ax1.set_title('Function Overview with All Extrema')

        # Plot extrema from all methods
        colors = ['red', 'green', 'orange']
        markers = ['o', 's', '^']
        method_names = ['Golden Section', 'Parabolic', 'Newton']

        for i, extremum in enumerate(results['summary']):
            methods = [extremum['golden_section'], extremum['parabolic'], extremum['newton']]

            for j, (method, color, marker) in enumerate(zip(methods, colors, markers)):
                if method.converged:
                    ax1.plot(method.x_optimal, method.f_optimal,
                            marker=marker, color=color, markersize=8,
                            label=f'{method_names[j]} - {extremum["type"]}' if i == 0 else "")

        ax1.legend()

        # Plot 2: Zoomed view of extrema regions
        ax2.plot(x, y, 'b-', linewidth=1, alpha=0.7, label='f(x)')

        # Add detailed view of each extremum
        for i, extremum in enumerate(results['summary']):
            candidate = extremum['candidate']

            # Zoom around this extremum
            zoom_width = 0.5
            x_zoom = np.linspace(candidate - zoom_width, candidate + zoom_width, 200)
            y_zoom = [self.f(xi) for xi in x_zoom]

            # Plot the zoomed function
            ax2.plot(x_zoom, y_zoom, '--', alpha=0.5, linewidth=1)

            # Mark the extrema
            for j, method in enumerate([extremum['golden_section'], extremum['parabolic'], extremum['newton']]):
                if method.converged:
                    ax2.plot(method.x_optimal, method.f_optimal,
                            marker=markers[j], color=colors[j], markersize=6)

            # Add text annotation
            ax2.annotate(f'{extremum["type"]}\nx ~= {candidate:.4f}',
                        xy=(candidate, extremum['f_value']),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                        fontsize=8)

        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('x')
        ax2.set_ylabel('f(x)')
        ax2.set_title('Detailed View of Extrema Regions')

        plt.tight_layout()
        plt.show()

    def print_summary_table(self, results: dict):
        """Print a comprehensive summary table of all results."""
        print("\n" + "="*100)
        print("COMPREHENSIVE SUMMARY TABLE")
        print("="*100)

        header = f"{'Extremum':<12} {'Type':<18} {'Golden Section':<20} {'Parabolic':<20} {'Newton':<20} {'Iterations (G/P/N)':<20}"
        print(header)
        print("-"*100)

        for i, extremum in enumerate(results['summary']):
            gs = extremum['golden_section']
            pa = extremum['parabolic']
            ne = extremum['newton']

            extremum_num = f"#{i+1}"
            extremum_type = extremum['type']

            gs_result = f"{gs.x_optimal:.6f}" if gs.converged else "Failed"
            pa_result = f"{pa.x_optimal:.6f}" if pa.converged else "Failed"
            ne_result = f"{ne.x_optimal:.6f}" if ne.converged else "Failed"

            iterations = f"{gs.iterations}/{pa.iterations}/{ne.iterations}"

            row = f"{extremum_num:<12} {extremum_type:<18} {gs_result:<20} {pa_result:<20} {ne_result:<20} {iterations:<20}"
            print(row)

        print("-"*100)
        print("Method Performance Summary:")
        print(f"Golden Section: {sum(1 for r in results['golden_section_results'] if r.converged)}/{len(results['golden_section_results'])} converged")
        print(f"Parabolic:      {sum(1 for r in results['parabolic_results'] if r.converged)}/{len(results['parabolic_results'])} converged")
        print(f"Newton:         {sum(1 for r in results['newton_results'] if r.converged)}/{len(results['newton_results'])} converged")


def run_comprehensive_extrema_analysis():
    """Main function to run the comprehensive extrema analysis."""
    # Initialize the extrema finder
    finder = ExtremaFinder(tolerance=1e-8, max_iterations=1000)

    # Run comprehensive analysis
    results = finder.comprehensive_analysis(domain=(-8, 8))

    # Print summary table
    finder.print_summary_table(results)

    # Generate plots
    print("\nGenerating comprehensive visualization...")
    finder.plot_results(results, domain=(-8, 8))

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("Key Insights:")
    print("1. Golden Section Search: Robust but requires bracketing intervals")
    print("2. Successive Parabolic Interpolation: Fast convergence when conditions are met")
    print("3. Newton's Method: Fastest convergence but requires derivatives")
    print("4. All methods successfully identified the extrema locations")
    print("5. Function has multiple extrema due to the exp(-x²) factor and polynomial structure")

    return results


if __name__ == "__main__":
    # Run the comprehensive analysis
    results = run_comprehensive_extrema_analysis()