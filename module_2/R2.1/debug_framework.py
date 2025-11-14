import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Callable, Dict, Any
import math
from dataclasses import dataclass

@dataclass
class DebugResult:
    """Detailed debugging information for method analysis."""
    method_name: str
    function_name: str
    expected_root: float
    found_root: Optional[float]
    converged: bool
    iterations: int
    iteration_history: List[float]
    error_history: List[float]
    final_error: float
    failure_reason: Optional[str]
    theoretical_expectation: str
    actual_vs_expected: str

class RootFindingDebugger:
    """Comprehensive debugging framework for root-finding methods."""

    def __init__(self, tolerance: float = 1e-6, max_iterations: int = 1000):
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.debug_results = []

    def log_debug(self, message: str, level: str = "INFO"):
        """Debug logging with levels."""
        print(f"[{level}] {message}")

    # ============================================================================
    # PHASE 1: METHOD ISOLATION TESTS
    # ============================================================================

    def test_bisection_simple(self) -> List[DebugResult]:
        """Test bisection on simple cases where it should work perfectly."""
        self.log_debug("PHASE 1: Testing Bisection Method Isolation", "HEADER")

        test_cases = [
            {
                'function': lambda x: x,
                'derivative': lambda x: 1,
                'interval': (-1.0, 1.0),
                'expected_root': 0.0,
                'name': 'f(x) = x'
            },
            {
                'function': lambda x: x - 2,
                'derivative': lambda x: 1,
                'interval': (1.0, 3.0),
                'expected_root': 2.0,
                'name': 'f(x) = x - 2'
            },
            {
                'function': lambda x: x**2 - 4,
                'derivative': lambda x: 2*x,
                'interval': (1.0, 3.0),
                'expected_root': 2.0,
                'name': 'f(x) = x² - 4'
            },
            {
                'function': lambda x: np.sin(x),
                'derivative': lambda x: np.cos(x),
                'interval': (2.0, 4.0),
                'expected_root': np.pi,
                'name': 'f(x) = sin(x)'
            }
        ]

        results = []
        for test in test_cases:
            self.log_debug(f"Testing Bisection on {test['name']}")
            result = self.debug_bisection_method(
                test['function'], test['interval'][0], test['interval'][1],
                test['expected_root'], test['name']
            )
            results.append(result)

            # Log detailed analysis
            self.log_debug(f"  Expected: {test['expected_root']:.6f}")
            found_str = f"{result.found_root:.6f}" if result.found_root is not None else "FAILED"
            self.log_debug(f"  Found: {found_str}")
            self.log_debug(f"  Converged: {result.converged}")
            self.log_debug(f"  Iterations: {result.iterations}")
            final_error_str = f"{result.final_error:.2e}" if result.final_error != float('inf') else "inf"
            self.log_debug(f"  Final Error: {final_error_str}")

        return results

    def test_newton_simple(self) -> List[DebugResult]:
        """Test Newton's method on simple cases with known behavior."""
        self.log_debug("PHASE 1: Testing Newton Method Isolation", "HEADER")

        test_cases = [
            {
                'function': lambda x: x**2 - 4,
                'derivative': lambda x: 2*x,
                'x0': 1.0,
                'expected_root': 2.0,
                'name': 'f(x) = x² - 4',
                'convergence_type': 'quadratic'
            },
            {
                'function': lambda x: x**3 - 8,
                'derivative': lambda x: 3*x**2,
                'x0': 1.0,
                'expected_root': 2.0,
                'name': 'f(x) = x³ - 8',
                'convergence_type': 'quadratic'
            },
            {
                'function': lambda x: np.cos(x),
                'derivative': lambda x: -np.sin(x),
                'x0': 1.0,
                'expected_root': np.pi/2,
                'name': 'f(x) = cos(x)',
                'convergence_type': 'quadratic'
            }
        ]

        results = []
        for test in test_cases:
            self.log_debug(f"Testing Newton on {test['name']}")
            result = self.debug_newton_method(
                test['function'], test['derivative'], test['x0'],
                test['expected_root'], test['name']
            )
            results.append(result)

            # Analyze convergence rate
            convergence_rate = self.analyze_convergence_rate(result.iteration_history)
            self.log_debug(f"  Expected: {test['expected_root']:.6f}")
            found_str = f"{result.found_root:.6f}" if result.found_root is not None else "FAILED"
            self.log_debug(f"  Found: {found_str}")
            self.log_debug(f"  Convergence Rate: {convergence_rate}")
            self.log_debug(f"  Expected Rate: {test['convergence_type']}")

        return results

    def test_modified_newton_known_multiple(self) -> List[DebugResult]:
        """Test modified Newton on known multiple roots."""
        self.log_debug("PHASE 1: Testing Modified Newton on Multiple Roots", "HEADER")

        test_cases = [
            {
                'function': lambda x: (x - 1)**2,
                'derivative': lambda x: 2*(x - 1),
                'x0': 0.5,
                'expected_root': 1.0,
                'multiplicity': 2,
                'name': 'f(x) = (x-1)² (double root)'
            },
            {
                'function': lambda x: (x - 1)**3,
                'derivative': lambda x: 3*(x - 1)**2,
                'x0': 0.5,
                'expected_root': 1.0,
                'multiplicity': 3,
                'name': 'f(x) = (x-1)³ (triple root)'
            },
            {
                'function': lambda x: (x - 1)**3 * (x + 2),
                'derivative': lambda x: 3*(x - 1)**2*(x + 2) + (x - 1)**3,
                'x0': 0.5,
                'expected_root': 1.0,
                'multiplicity': 3,
                'name': 'f(x) = (x-1)³(x+2) at x=1'
            }
        ]

        results = []
        for test in test_cases:
            self.log_debug(f"Testing Modified Newton on {test['name']}")

            # Test both standard and modified Newton
            result_standard = self.debug_newton_method(
                test['function'], test['derivative'], test['x0'],
                test['expected_root'], f"{test['name']} - Standard"
            )

            result_modified = self.debug_modified_newton_method(
                test['function'], test['derivative'], test['x0'],
                test['expected_root'], f"{test['name']} - Modified m={test['multiplicity']}",
                test['multiplicity']
            )

            results.extend([result_standard, result_modified])

            self.log_debug(f"  Standard Newton - Converged: {result_standard.converged}, Iterations: {result_standard.iterations}")
            self.log_debug(f"  Modified Newton - Converged: {result_modified.converged}, Iterations: {result_modified.iterations}")

        return results

    def test_fixed_point_conditions(self) -> List[DebugResult]:
        """Test fixed-point method with known convergence conditions."""
        self.log_debug("PHASE 1: Testing Fixed-Point Convergence Conditions", "HEADER")

        test_cases = [
            {
                'g_function': lambda x: 0.5 * x + 1,  # |g'(x)| = 0.5 < 1, should converge
                'g_derivative': lambda x: 0.5,
                'x0': 0.0,
                'expected_root': 2.0,  # Fixed point: x = 0.5x + 1 → x = 2
                'name': 'g(x) = 0.5x + 1 (should converge)',
                'should_converge': True
            },
            {
                'g_function': lambda x: 2 * x + 1,  # |g'(x)| = 2 > 1, should diverge
                'g_derivative': lambda x: 2,
                'x0': 0.0,
                'expected_root': None,
                'name': 'g(x) = 2x + 1 (should diverge)',
                'should_converge': False
            },
            {
                'g_function': lambda x: np.cos(x),  # |g'(x)| = |sin(x)| varies
                'g_derivative': lambda x: -np.sin(x),
                'x0': 0.5,
                'expected_root': 0.7390851332,  # Fixed point of cos(x)
                'name': 'g(x) = cos(x) (conditional)',
                'should_converge': True
            }
        ]

        results = []
        for test in test_cases:
            self.log_debug(f"Testing Fixed-Point: {test['name']}")

            # Check convergence condition first
            gp_at_x0 = abs(test['g_derivative'](test['x0']))
            self.log_debug(f"  |g'(x0)| = {gp_at_x0:.6f}")

            result = self.debug_fixed_point_method(
                test['g_function'], test['x0'],
                test['expected_root'], test['name']
            )
            results.append(result)

            self.log_debug(f"  Should converge: {test['should_converge']}")
            self.log_debug(f"  Actually converged: {result.converged}")
            consistency = "PASS" if result.converged == test['should_converge'] else "FAIL"
            self.log_debug(f"  Consistency: {consistency}")

        return results

    # ============================================================================
    # INDIVIDUAL METHOD DEBUGGING IMPLEMENTATIONS
    # ============================================================================

    def debug_bisection_method(self, f: Callable, a: float, b: float,
                             expected_root: float, function_name: str) -> DebugResult:
        """Debug bisection method with detailed tracking."""

        # Check initial sign condition
        fa, fb = f(a), f(b)
        if fa * fb >= 0:
            return DebugResult(
                "Bisection", function_name, expected_root, None, False, 0,
                [a, b], [float('inf')], float('inf'),
                f"No sign change: f({a}) = {fa:.6f}, f({b}) = {fb:.6f}",
                "Should work with proper brackets",
                "FAILED: No sign change detected"
            )

        iteration_history = [a, b]
        error_history = [abs(b - a)]
        iterations = 0

        while (b - a) / 2 > self.tolerance and iterations < self.max_iterations:
            c = (a + b) / 2
            fc = f(c)
            iteration_history.append(c)

            error = abs(c - expected_root) if expected_root is not None else abs(fc)
            error_history.append(error)

            if abs(fc) < self.tolerance:
                return DebugResult(
                    "Bisection", function_name, expected_root, c, True, iterations + 1,
                    iteration_history, error_history, error,
                    None, "Should converge linearly", "SUCCESS: Converged to root"
                )

            if f(a) * fc < 0:
                b = c
            else:
                a = c
            iterations += 1

        final_root = (a + b) / 2
        final_error = abs(final_root - expected_root) if expected_root is not None else abs(f(final_root))

        converged = iterations < self.max_iterations
        failure_reason = "Max iterations exceeded" if not converged else None

        return DebugResult(
            "Bisection", function_name, expected_root, final_root, converged, iterations,
            iteration_history, error_history, final_error, failure_reason,
            "Should converge linearly",
            "SUCCESS: Converged" if converged else f"FAILED: {failure_reason}"
        )

    def debug_newton_method(self, f: Callable, df: Callable, x0: float,
                          expected_root: float, function_name: str) -> DebugResult:
        """Debug Newton's method with detailed tracking."""

        x = x0
        iteration_history = [x]
        error_history = [abs(x - expected_root) if expected_root is not None else abs(f(x))]
        iterations = 0

        for i in range(self.max_iterations):
            fx = f(x)

            if abs(fx) < self.tolerance:
                return DebugResult(
                    "Newton Standard", function_name, expected_root, x, True, iterations,
                    iteration_history, error_history, error_history[-1],
                    None, "Should converge quadratically", "SUCCESS: Converged to root"
                )

            dfx = df(x)
            if abs(dfx) < 1e-12:
                return DebugResult(
                    "Newton Standard", function_name, expected_root, x, False, iterations,
                    iteration_history, error_history, error_history[-1],
                    "Derivative became zero", "Should work away from critical points",
                    "FAILED: Zero derivative"
                )

            x_new = x - fx / dfx
            iterations += 1

            if abs(x_new) > 1e10:
                return DebugResult(
                    "Newton Standard", function_name, expected_root, x_new, False, iterations,
                    iteration_history, error_history, float('inf'),
                    "Solution diverged", "Should converge with good starting point",
                    "FAILED: Divergence"
                )

            iteration_history.append(x_new)
            error = abs(x_new - expected_root) if expected_root is not None else abs(f(x_new))
            error_history.append(error)

            if abs(x_new - x) < self.tolerance:
                return DebugResult(
                    "Newton Standard", function_name, expected_root, x_new, True, iterations,
                    iteration_history, error_history, error,
                    None, "Should converge quadratically", "SUCCESS: Converged"
                )

            x = x_new

        return DebugResult(
            "Newton Standard", function_name, expected_root, x, False, iterations,
            iteration_history, error_history, error_history[-1],
            "Max iterations exceeded", "Should converge quickly", "FAILED: Too many iterations"
        )

    def debug_modified_newton_method(self, f: Callable, df: Callable, x0: float,
                                   expected_root: float, function_name: str,
                                   multiplicity: int) -> DebugResult:
        """Debug modified Newton's method for multiple roots."""

        x = x0
        iteration_history = [x]
        error_history = [abs(x - expected_root) if expected_root is not None else abs(f(x))]
        iterations = 0

        for i in range(self.max_iterations):
            fx = f(x)

            if abs(fx) < self.tolerance:
                return DebugResult(
                    f"Newton Modified m={multiplicity}", function_name, expected_root, x, True, iterations,
                    iteration_history, error_history, error_history[-1],
                    None, f"Should converge linearly for m={multiplicity} root", "SUCCESS: Converged to root"
                )

            dfx = df(x)
            if abs(dfx) < 1e-12:
                return DebugResult(
                    f"Newton Modified m={multiplicity}", function_name, expected_root, x, False, iterations,
                    iteration_history, error_history, error_history[-1],
                    "Derivative became zero", "Should handle multiple roots better",
                    "FAILED: Zero derivative"
                )

            # Modified Newton: x_new = x - m * f(x) / f'(x)
            x_new = x - multiplicity * fx / dfx
            iterations += 1

            if abs(x_new) > 1e10:
                return DebugResult(
                    f"Newton Modified m={multiplicity}", function_name, expected_root, x_new, False, iterations,
                    iteration_history, error_history, float('inf'),
                    "Solution diverged", "Should be more stable than standard Newton",
                    "FAILED: Divergence"
                )

            iteration_history.append(x_new)
            error = abs(x_new - expected_root) if expected_root is not None else abs(f(x_new))
            error_history.append(error)

            if abs(x_new - x) < self.tolerance:
                return DebugResult(
                    f"Newton Modified m={multiplicity}", function_name, expected_root, x_new, True, iterations,
                    iteration_history, error_history, error,
                    None, f"Should converge better than standard for m={multiplicity}", "SUCCESS: Converged"
                )

            x = x_new

        return DebugResult(
            f"Newton Modified m={multiplicity}", function_name, expected_root, x, False, iterations,
            iteration_history, error_history, error_history[-1],
            "Max iterations exceeded", "Should converge faster than standard", "FAILED: Too many iterations"
        )

    def debug_fixed_point_method(self, g: Callable, x0: float,
                               expected_root: Optional[float], function_name: str) -> DebugResult:
        """Debug fixed-point method with convergence condition checking."""

        x = x0
        iteration_history = [x]
        error_history = []
        iterations = 0

        # Check convergence condition at starting point
        h = 1e-8
        try:
            gp = (g(x0 + h) - g(x0 - h)) / (2 * h)
            if abs(gp) >= 1.0:
                return DebugResult(
                    "Fixed-Point", function_name, expected_root, None, False, 0,
                    iteration_history, [float('inf')], float('inf'),
                    f"Convergence condition violated: |g'(x0)| = {abs(gp):.6f} >= 1",
                    "Should check |g'(x)| < 1", "FAILED: Convergence condition"
                )
        except:
            pass

        for i in range(self.max_iterations):
            try:
                x_new = g(x)
                iterations += 1

                if abs(x_new) > 1e10:
                    return DebugResult(
                        "Fixed-Point", function_name, expected_root, x_new, False, iterations,
                        iteration_history, error_history, float('inf'),
                        "Solution diverged", "Should converge if |g'(x)| < 1",
                        "FAILED: Divergence"
                    )

                iteration_history.append(x_new)
                error = abs(x_new - x)
                error_history.append(error)

                if error < self.tolerance:
                    final_error = abs(x_new - expected_root) if expected_root is not None else error
                    return DebugResult(
                        "Fixed-Point", function_name, expected_root, x_new, True, iterations,
                        iteration_history, error_history, final_error,
                        None, "Should converge if |g'(x)| < 1", "SUCCESS: Converged"
                    )

                x = x_new

            except Exception as e:
                return DebugResult(
                    "Fixed-Point", function_name, expected_root, x, False, iterations,
                    iteration_history, error_history, float('inf'),
                    f"Runtime error: {str(e)}", "Should handle domain carefully",
                    f"FAILED: Runtime error"
                )

        final_error = abs(x - expected_root) if expected_root is not None else abs(iteration_history[-1] - iteration_history[-2])
        return DebugResult(
            "Fixed-Point", function_name, expected_root, x, False, iterations,
            iteration_history, error_history, final_error,
            "Max iterations exceeded", "Should converge if conditions met", "FAILED: Too many iterations"
        )

    # ============================================================================
    # CONVERGENCE ANALYSIS UTILITIES
    # ============================================================================

    def analyze_convergence_rate(self, iteration_history: List[float]) -> str:
        """Analyze convergence rate from iteration history."""
        if len(iteration_history) < 4:
            return "Insufficient data"

        try:
            # Calculate successive errors (assuming converging to last value)
            target = iteration_history[-1]
            errors = [abs(x - target) for x in iteration_history[:-1]]

            # Remove zeros and very small errors
            errors = [e for e in errors if e > 1e-15]
            if len(errors) < 3:
                return "Converged too quickly to analyze"

            # Check for linear convergence: e_{n+1} ≈ C * e_n
            linear_ratios = [errors[i+1] / errors[i] for i in range(len(errors)-1) if errors[i] > 1e-15]

            # Check for quadratic convergence: e_{n+1} ≈ C * e_n^2
            quad_ratios = [errors[i+1] / (errors[i]**2) for i in range(len(errors)-1) if errors[i] > 1e-10]

            if linear_ratios:
                avg_linear = np.mean(linear_ratios)
                std_linear = np.std(linear_ratios)

                if std_linear < 0.1 * avg_linear and avg_linear < 1:
                    return f"Linear (ratio approx {avg_linear:.3f})"

            if quad_ratios:
                avg_quad = np.mean(quad_ratios)
                std_quad = np.std(quad_ratios)

                if len(quad_ratios) >= 2 and std_quad < abs(avg_quad):
                    return f"Quadratic (C approx {avg_quad:.2e})"

            return "Irregular"

        except:
            return "Analysis failed"

    # ============================================================================
    # PHASE 2: ROOT VERIFICATION DEBUGGING
    # ============================================================================

    def debug_root_verification(self) -> Dict[str, Any]:
        """Debug the root verification logic."""
        self.log_debug("PHASE 2: Debugging Root Verification Logic", "HEADER")

        verification_tests = []

        # Test 1: Known correct roots
        test_cases = [
            {'root': 0.0, 'interval': (-1, 1), 'function': lambda x: x, 'should_pass': True},
            {'root': 2.0, 'interval': (1, 3), 'function': lambda x: x - 2, 'should_pass': True},
            {'root': 1.5708, 'interval': (1, 2), 'function': lambda x: np.cos(x), 'should_pass': True},
            {'root': 0.5, 'interval': (1, 2), 'function': lambda x: x, 'should_pass': False},  # Wrong interval
        ]

        margin_tests = [0.01, 0.1, 0.5, 1.0]  # Different margins to test

        for margin in margin_tests:
            self.log_debug(f"Testing verification with margin = {margin}")

            for test in test_cases:
                # Test current verification logic
                in_interval = self.verify_root_in_interval(
                    test['root'], test['interval'][0], test['interval'][1], margin
                )

                # Test function value at root
                function_value = abs(test['function'](test['root']))

                verification_tests.append({
                    'margin': margin,
                    'root': test['root'],
                    'interval': test['interval'],
                    'in_interval': in_interval,
                    'should_pass': test['should_pass'],
                    'function_value': function_value,
                    'consistent': in_interval == test['should_pass']
                })

                self.log_debug(f"  Root {test['root']} in {test['interval']}: {in_interval} (expected: {test['should_pass']})")

        return {'verification_tests': verification_tests}

    def verify_root_in_interval(self, root: float, a: float, b: float, margin: float = 0.1) -> bool:
        """Current root verification logic (for debugging)."""
        return a - margin <= root <= b + margin

    # ============================================================================
    # PHASE 3: STATISTICAL FRAMEWORK VALIDATION
    # ============================================================================

    def debug_success_rate_calculation(self) -> Dict[str, Any]:
        """Debug the success rate calculation methodology."""
        self.log_debug("PHASE 3: Debugging Success Rate Calculation", "HEADER")

        # Test with known outcomes
        test_scenarios = [
            {'successes': 10, 'total': 10, 'expected_rate': 1.0},
            {'successes': 5, 'total': 10, 'expected_rate': 0.5},
            {'successes': 1, 'total': 18, 'expected_rate': 0.0556},  # This might explain the 5.56% pattern
            {'successes': 0, 'total': 10, 'expected_rate': 0.0},
        ]

        calculation_results = []
        for scenario in test_scenarios:
            calculated_rate = scenario['successes'] / scenario['total']
            calculation_results.append({
                'scenario': scenario,
                'calculated_rate': calculated_rate,
                'expected_rate': scenario['expected_rate'],
                'match': abs(calculated_rate - scenario['expected_rate']) < 0.001
            })

            self.log_debug(f"Successes: {scenario['successes']}, Total: {scenario['total']}")
            self.log_debug(f"  Calculated: {calculated_rate:.4f}, Expected: {scenario['expected_rate']:.4f}")

        # Test random starting point generation
        self.log_debug("Testing starting point generation:")
        interval = (1.0, 2.0)
        num_points = 18  # Same as in original implementation

        starting_points = [interval[0] + (interval[1] - interval[0]) * i / (num_points - 1)
                          for i in range(num_points)]

        self.log_debug(f"Generated {len(starting_points)} points in {interval}")
        self.log_debug(f"Points: {starting_points[:5]}...{starting_points[-5:]}")

        # Check if points are actually in interval
        in_interval = [interval[0] <= p <= interval[1] for p in starting_points]
        self.log_debug(f"All points in interval: {all(in_interval)}")

        return {
            'calculation_results': calculation_results,
            'starting_points': starting_points,
            'all_in_interval': all(in_interval)
        }

    # ============================================================================
    # PHASE 4: THEORETICAL CONSISTENCY VALIDATION
    # ============================================================================

    def validate_theoretical_expectations(self) -> Dict[str, Any]:
        """Validate results against known numerical analysis theory."""
        self.log_debug("PHASE 4: Validating Theoretical Consistency", "HEADER")

        theoretical_tests = []

        # Test 1: Standard Newton on multiple root should struggle
        self.log_debug("Testing Newton on multiple root (x-1)³:")
        polynomial_func = lambda x: (x - 1)**3
        polynomial_deriv = lambda x: 3*(x - 1)**2

        newton_multiple = self.debug_newton_method(
            polynomial_func, polynomial_deriv, 0.5, 1.0, "(x-1)³ at x=1"
        )

        # Test 2: Modified Newton should handle multiple roots better
        modified_multiple = self.debug_modified_newton_method(
            polynomial_func, polynomial_deriv, 0.5, 1.0, "(x-1)³ at x=1", 3
        )

        # Test 3: Bisection should work reliably
        bisection_multiple = self.debug_bisection_method(
            polynomial_func, 0.5, 1.5, 1.0, "(x-1)³ at x=1"
        )

        theoretical_tests.extend([newton_multiple, modified_multiple, bisection_multiple])

        # Test 4: Newton on simple root should converge quickly
        simple_func = lambda x: x**2 - 4
        simple_deriv = lambda x: 2*x

        newton_simple = self.debug_newton_method(
            simple_func, simple_deriv, 1.0, 2.0, "x²-4 at x=2"
        )

        theoretical_tests.append(newton_simple)

        # Analyze consistency
        consistency_analysis = {
            'newton_multiple_slow': newton_multiple.iterations > 5,
            'modified_better_than_standard': (modified_multiple.converged and modified_multiple.iterations < newton_multiple.iterations),
            'bisection_reliable': bisection_multiple.converged,
            'newton_simple_fast': newton_simple.converged and newton_simple.iterations <= 5
        }

        self.log_debug("Theoretical Consistency Check:")
        for test, result in consistency_analysis.items():
            status = "PASS" if result else "FAIL"
            self.log_debug(f"  {test}: {status}")

        return {
            'theoretical_tests': theoretical_tests,
            'consistency_analysis': consistency_analysis
        }

    # ============================================================================
    # COMPREHENSIVE DEBUGGING REPORT
    # ============================================================================

    def run_comprehensive_debug(self) -> Dict[str, Any]:
        """Run complete debugging framework and generate report."""
        self.log_debug("="*80, "HEADER")
        self.log_debug("COMPREHENSIVE ROOT-FINDING DEBUGGING FRAMEWORK", "HEADER")
        self.log_debug("="*80, "HEADER")

        debug_report = {}

        # Phase 1: Method Isolation
        debug_report['bisection_simple'] = self.test_bisection_simple()
        debug_report['newton_simple'] = self.test_newton_simple()
        debug_report['modified_newton_multiple'] = self.test_modified_newton_known_multiple()
        debug_report['fixed_point_conditions'] = self.test_fixed_point_conditions()

        # Phase 2: Root Verification
        debug_report['root_verification'] = self.debug_root_verification()

        # Phase 3: Statistical Framework
        debug_report['success_rate_calc'] = self.debug_success_rate_calculation()

        # Phase 4: Theoretical Consistency
        debug_report['theoretical_validation'] = self.validate_theoretical_expectations()

        # Generate summary
        debug_report['summary'] = self.generate_debug_summary(debug_report)

        return debug_report

    def generate_debug_summary(self, debug_report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of debugging findings."""
        self.log_debug("="*60, "HEADER")
        self.log_debug("DEBUGGING SUMMARY AND IDENTIFIED ISSUES", "HEADER")
        self.log_debug("="*60, "HEADER")

        issues_found = []
        fixes_needed = []

        # Analyze bisection results
        bisection_results = debug_report['bisection_simple']
        bisection_success_rate = sum(1 for r in bisection_results if r.converged) / len(bisection_results)

        if bisection_success_rate < 0.95:
            issues_found.append(f"Bisection success rate too low: {bisection_success_rate:.2%}")
            fixes_needed.append("Fix bisection sign checking and interval handling")

        # Analyze Newton results
        newton_results = debug_report['newton_simple']
        newton_success_rate = sum(1 for r in newton_results if r.converged) / len(newton_results)

        if newton_success_rate < 0.8:
            issues_found.append(f"Newton success rate too low: {newton_success_rate:.2%}")
            fixes_needed.append("Debug Newton iteration formula and convergence criteria")

        # Analyze modified Newton
        modified_results = [r for r in debug_report['modified_newton_multiple'] if 'Modified' in r.method_name]
        if modified_results:
            modified_success_rate = sum(1 for r in modified_results if r.converged) / len(modified_results)
            if modified_success_rate == 0:
                issues_found.append("Modified Newton completely failing")
                fixes_needed.append("Debug modified Newton formula: x_new = x - m*f(x)/f'(x)")

        # Analyze success rate calculation
        if debug_report['success_rate_calc']['calculation_results']:
            calc_test = debug_report['success_rate_calc']['calculation_results'][2]  # The 1/18 test
            if abs(calc_test['calculated_rate'] - 0.0556) < 0.001:
                issues_found.append("Found source of 5.56% pattern: 1 success out of 18 attempts")
                fixes_needed.append("Debug why only 1 out of 18 attempts succeeds")

        summary = {
            'total_issues': len(issues_found),
            'issues_found': issues_found,
            'fixes_needed': fixes_needed,
            'bisection_success_rate': bisection_success_rate,
            'newton_success_rate': newton_success_rate,
            'critical_failures': [issue for issue in issues_found if 'completely failing' in issue]
        }

        self.log_debug(f"TOTAL ISSUES FOUND: {len(issues_found)}")
        for i, issue in enumerate(issues_found, 1):
            self.log_debug(f"{i}. {issue}")

        self.log_debug(f"\nFIXES NEEDED:")
        for i, fix in enumerate(fixes_needed, 1):
            self.log_debug(f"{i}. {fix}")

        return summary

def run_debugging_framework():
    """Main function to run the debugging framework."""
    debugger = RootFindingDebugger(tolerance=1e-6, max_iterations=1000)
    debug_report = debugger.run_comprehensive_debug()

    return debug_report

if __name__ == "__main__":
    run_debugging_framework()