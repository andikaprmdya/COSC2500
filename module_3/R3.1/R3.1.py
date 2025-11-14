import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os
import time
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap

# ============================================================================
# PART 0: CREATE OUTPUT DIRECTORY
# ============================================================================

# Create directory for saving figures
output_dir = r'C:\Users\Dika\UQ\Anum\module_3\R3.1'
os.makedirs(output_dir, exist_ok=True)
print(f"\n📁 Output directory created: {output_dir}")
print(f"   All figures will be saved here.\n")

# ============================================================================
# PART 1: ENHANCED METHOD IMPLEMENTATIONS WITH FUNCTION COUNTING
# ============================================================================

def euler_method(f, t_span, y0, h):
    """
    Euler's method for solving ODEs with function evaluation counting
    
    Parameters:
    -----------
    f : function
        Derivative function dy/dt = f(t, y)
    t_span : tuple
        (t_initial, t_final)
    y0 : float
        Initial condition
    h : float
        Step size
    
    Returns:
    --------
    t : ndarray
        Time points
    y : ndarray
        Solution values
    func_evals : int
        Number of function evaluations
    """
    t_initial, t_final = t_span
    n_steps = int((t_final - t_initial) / h)
    
    t = np.linspace(t_initial, t_final, n_steps + 1)
    y = np.zeros(n_steps + 1)
    y[0] = y0
    
    # Track function evaluations
    func_evals = 0
    
    for i in range(n_steps):
        y[i + 1] = y[i] + h * f(t[i], y[i])
        func_evals += 1  # Euler: 1 evaluation per step
    
    return t, y, func_evals


def rk4_method(f, t_span, y0, h):
    """
    4th-order Runge-Kutta method for solving ODEs with function evaluation counting
    
    Parameters:
    -----------
    f : function
        Derivative function dy/dt = f(t, y)
    t_span : tuple
        (t_initial, t_final)
    y0 : float
        Initial condition
    h : float
        Step size
    
    Returns:
    --------
    t : ndarray
        Time points
    y : ndarray
        Solution values
    func_evals : int
        Number of function evaluations
    """
    t_initial, t_final = t_span
    n_steps = int((t_final - t_initial) / h)
    
    t = np.linspace(t_initial, t_final, n_steps + 1)
    y = np.zeros(n_steps + 1)
    y[0] = y0
    
    # Track function evaluations
    func_evals = 0
    
    for i in range(n_steps):
        k1 = h * f(t[i], y[i])
        k2 = h * f(t[i] + 0.5*h, y[i] + 0.5*k1)
        k3 = h * f(t[i] + 0.5*h, y[i] + 0.5*k2)
        k4 = h * f(t[i] + h, y[i] + k3)
        
        y[i + 1] = y[i] + (1/6) * (k1 + 2*k2 + 2*k3 + k4)
        func_evals += 4  # RK4: 4 evaluations per step
    
    return t, y, func_evals


# ============================================================================
# PART 2: RICHARDSON EXTRAPOLATION FOR PREDICTIVE ANALYSIS
# ============================================================================

def richardson_extrapolation(step_sizes, errors):
    """
    Perform Richardson extrapolation to predict error behavior
    and estimate step size requirements for target accuracy
    
    Parameters:
    -----------
    step_sizes : list
        List of step sizes used
    errors : list
        Corresponding errors
        
    Returns:
    --------
    dict containing:
        - 'coefficient': Error constant C (Error = C*h^p)
        - 'order': Convergence order p
        - 'h_for_target': Function to calculate h for target error
        - 'h_predict': Array of h values for plotting
        - 'error_predict': Predicted errors for plotting
    """
    # Use first two points to estimate convergence behavior
    h1, h2 = step_sizes[0], step_sizes[1]
    e1, e2 = errors[0], errors[1]
    
    # Avoid division by zero or log of zero
    if e1 <= 0 or e2 <= 0 or h1 <= 0 or h2 <= 0:
        # Return default values
        return {
            'coefficient': np.nan,
            'order': np.nan,
            'h_for_target': lambda target: np.nan,
            'h_predict': step_sizes,
            'error_predict': errors
        }
    
    # Calculate convergence order: p = log(e1/e2) / log(h1/h2)
    order = np.log(e1/e2) / np.log(h1/h2)
    
    # Calculate error coefficient: C = e1 / h1^p
    coefficient = e1 / (h1 ** order)
    
    # Function to predict step size for target error
    def h_for_target_error(target_error):
        """Calculate h needed to achieve target error"""
        if target_error <= 0 or coefficient <= 0:
            return np.nan
        return (target_error / coefficient) ** (1/order)
    
    # Generate predictions for finer step sizes (for plotting)
    h_min = step_sizes[-1] / 10  # Extend beyond measured range
    h_max = step_sizes[0] * 2
    h_predict = np.logspace(np.log10(h_min), np.log10(h_max), 100)
    error_predict = coefficient * h_predict ** order
    
    return {
        'coefficient': coefficient,
        'order': order,
        'h_for_target': h_for_target_error,
        'h_predict': h_predict,
        'error_predict': error_predict
    }


# ============================================================================
# PART 3: COMPREHENSIVE RESULTS PRINTER
# ============================================================================

def print_comprehensive_results(problem_name, equation, step_sizes, 
                                euler_errors, rk4_errors, 
                                euler_evals, rk4_evals, 
                                adaptive_error, adaptive_steps,
                                euler_extrap, rk4_extrap):
    """
    Print comprehensive results table with extrapolation analysis
    """
    
    print(f"\n{'='*120}")
    print(f"PROBLEM {problem_name}: {equation}")
    print(f"{'='*120}")
    
    # Main results table
    print(f"\n{'Step Size h':<12} {'Euler Error':<15} {'RK4 Error':<15} {'Error Ratio':<12} "
          f"{'Euler Evals':<12} {'RK4 Evals':<12} {'Efficiency':<12}")
    print(f"{'-'*120}")
    
    for i, h in enumerate(step_sizes):
        ratio = euler_errors[i] / rk4_errors[i] if rk4_errors[i] > 1e-16 else np.inf
        
        # Efficiency = (RK4 evals) / (Euler evals) for same accuracy
        # Since RK4 is much more accurate, we show raw ratio
        efficiency = rk4_evals[i] / euler_evals[i]
        
        print(f"{h:<12.6f} {euler_errors[i]:<15.3e} {rk4_errors[i]:<15.3e} "
              f"{ratio:<12.2e} {euler_evals[i]:<12} {rk4_evals[i]:<12} {efficiency:<12.1f}×")
    
    print(f"{'-'*120}")
    print(f"{'Adaptive RK45:':<12} Error = {adaptive_error:.3e}, Steps = {adaptive_steps}, "
          f"Evals ≈ {adaptive_steps * 6} (estimated)")
    
    # Convergence rates
    if len(euler_errors) > 1:
        euler_rates = []
        rk4_rates = []
        
        for i in range(len(euler_errors)-1):
            if euler_errors[i] > 1e-16 and euler_errors[i+1] > 1e-16:
                euler_rate = np.log(euler_errors[i]/euler_errors[i+1]) / np.log(step_sizes[i]/step_sizes[i+1])
                euler_rates.append(euler_rate)
            
            if rk4_errors[i] > 1e-16 and rk4_errors[i+1] > 1e-16:
                rk4_rate = np.log(rk4_errors[i]/rk4_errors[i+1]) / np.log(step_sizes[i]/step_sizes[i+1])
                rk4_rates.append(rk4_rate)
        
        if euler_rates:
            print(f"\n{'Observed Convergence Rates:'}")
            print(f"{'Transition':<30} {'Euler Rate':<15} {'RK4 Rate':<15}")
            print(f"{'-'*60}")
            
            for i in range(min(len(euler_rates), len(rk4_rates))):
                print(f"{step_sizes[i]:.6f} → {step_sizes[i+1]:.6f}     "
                      f"{euler_rates[i]:<15.4f} {rk4_rates[i]:<15.4f}")
            
            print(f"\n{'Average Convergence Rates:'}")
            print(f"  Euler: {np.mean(euler_rates):.4f} (Expected: 1.0)")
            if rk4_rates:
                print(f"  RK4:   {np.mean(rk4_rates):.4f} (Expected: 4.0)")
    
    # Richardson extrapolation analysis
    print(f"\n{'Richardson Extrapolation Analysis:'}")
    if not np.isnan(euler_extrap['coefficient']):
        print(f"  Euler: Error ≈ {euler_extrap['coefficient']:.4e} × h^{euler_extrap['order']:.3f}")
    else:
        print(f"  Euler: Cannot extrapolate (insufficient data or machine precision)")
    
    if not np.isnan(rk4_extrap['coefficient']):
        print(f"  RK4:   Error ≈ {rk4_extrap['coefficient']:.4e} × h^{rk4_extrap['order']:.3f}")
    else:
        print(f"  RK4:   Cannot extrapolate (machine precision reached)")
    
    # Practical step size requirements
    print(f"\n{'Practical Step Size Requirements for Target Accuracy:'}")
    targets = [1e-4, 1e-6, 1e-8, 1e-10]
    
    print(f"{'Target Error':<15} {'Euler h':<15} {'Euler Steps':<15} "
          f"{'RK4 h':<15} {'RK4 Steps':<15} {'Efficiency Gain':<18}")
    print(f"{'-'*120}")
    
    for target in targets:
        h_euler = euler_extrap['h_for_target'](target)
        h_rk4 = rk4_extrap['h_for_target'](target)
        
        if not np.isnan(h_euler) and h_euler > 0:
            steps_euler = int(np.ceil(1 / h_euler))
            evals_euler = steps_euler * 1  # 1 eval per step
        else:
            steps_euler = np.inf
            evals_euler = np.inf
            h_euler = np.nan
        
        if not np.isnan(h_rk4) and h_rk4 > 0:
            steps_rk4 = int(np.ceil(1 / h_rk4))
            evals_rk4 = steps_rk4 * 4  # 4 evals per step
        else:
            steps_rk4 = np.inf
            evals_rk4 = np.inf
            h_rk4 = np.nan
        
        # Efficiency: How many times fewer function evaluations does RK4 need?
        if not np.isinf(evals_euler) and not np.isinf(evals_rk4) and evals_rk4 > 0:
            efficiency = evals_euler / evals_rk4
            efficiency_str = f"{efficiency:.1f}×"
        else:
            efficiency_str = "N/A"
        
        # Format output
        h_euler_str = f"{h_euler:.2e}" if not np.isnan(h_euler) else "N/A"
        h_rk4_str = f"{h_rk4:.2e}" if not np.isnan(h_rk4) else "N/A"
        steps_euler_str = f"{steps_euler}" if not np.isinf(steps_euler) else "N/A"
        steps_rk4_str = f"{steps_rk4}" if not np.isinf(steps_rk4) else "N/A"
        
        print(f"{target:<15.0e} {h_euler_str:<15} {steps_euler_str:<15} "
              f"{h_rk4_str:<15} {steps_rk4_str:<15} {efficiency_str:<18}")
    
    print(f"\n{'Key Insight:'} For high accuracy (e.g., error < 1e-8), RK4 requires dramatically fewer")
    print(f"              function evaluations than Euler, making it far more efficient overall.")
    
    print(f"{'='*120}\n")


# ============================================================================
# PART 4: DEFINE THE THREE ODE PROBLEMS
# ============================================================================

# Problem (a): y' = t
def f_a(t, y):
    return t

def exact_a(t):
    """Exact solution: y = t²/2 + 1"""
    return 0.5 * t**2 + 1


# Problem (b): y' = 2(t+1)y
def f_b(t, y):
    return 2 * (t + 1) * y

def exact_b(t):
    """Exact solution: y = exp(t² + 2t)"""
    return np.exp(t**2 + 2*t)


# Problem (c): y' = 1/y²
def f_c(t, y):
    if abs(y) < 1e-10:  # Prevent division by zero
        return 1e10
    return 1 / y**2

def exact_c(t):
    """Exact solution: y = (3t + 1)^(1/3)"""
    return (3*t + 1)**(1/3)


# ============================================================================
# PART 5: ENHANCED SOLVE FUNCTION WITH EXTRAPOLATION
# ============================================================================

def solve_problem(problem_name, equation, f, exact_solution, t_span, y0, save_prefix):
    """
    Solve ODE problem with multiple step sizes, perform extrapolation analysis,
    and create comprehensive visualizations
    
    Parameters:
    -----------
    problem_name : str
        Name of the problem (e.g., '(a)')
    equation : str
        Equation string for display
    f : function
        Derivative function
    exact_solution : function
        Analytical solution
    t_span : tuple
        Time span (t_initial, t_final)
    y0 : float
        Initial condition
    save_prefix : str
        Prefix for saved figure filenames
        
    Returns:
    --------
    dict : Comprehensive results dictionary
    """
    
    # Step sizes: h = 0.1 * 2^(-k) for k = 0, 1, 2, 3, 4, 5
    step_sizes = [0.1 * 2**(-k) for k in range(6)]
    
    # Storage for errors and function evaluations
    euler_errors = []
    rk4_errors = []
    euler_func_evals = []
    rk4_func_evals = []
    
    # ========================================================================
    # FIGURE 1: Solution comparison for h = 0.1, 0.05, 0.025
    # ========================================================================
    
    fig1, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig1.suptitle(f'Problem {problem_name}: {equation}', 
                  fontsize=18, fontweight='bold')
    
    for idx, h in enumerate([0.1, 0.05, 0.025]):
        ax = axes[idx]
        
        # Solve with Euler
        t_euler, y_euler, euler_evals = euler_method(f, t_span, y0, h)
        
        # Solve with RK4
        t_rk4, y_rk4, rk4_evals = rk4_method(f, t_span, y0, h)
        
        # Exact solution
        t_exact = np.linspace(t_span[0], t_span[1], 1000)
        y_exact = exact_solution(t_exact)
        
        # Plot
        ax.plot(t_exact, y_exact, 'k-', linewidth=2.5, label='Exact Solution', zorder=1)
        ax.plot(t_euler, y_euler, 'ro-', markersize=6, linewidth=1.5, 
                label='Euler', alpha=0.7, zorder=3)
        ax.plot(t_rk4, y_rk4, 'bs-', markersize=5, linewidth=1.5, 
                label='RK4', alpha=0.7, zorder=2)
        
        ax.set_xlabel('t', fontsize=13, fontweight='bold')
        ax.set_ylabel('y(t)', fontsize=13, fontweight='bold')
        ax.set_title(f'Step size h = {h}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add error annotation
        euler_err = abs(y_euler[-1] - exact_solution(t_span[1]))
        rk4_err = abs(y_rk4[-1] - exact_solution(t_span[1]))
        
        textstr = f'Error at t=1:\nEuler: {euler_err:.2e}\nRK4: {rk4_err:.2e}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props, family='monospace')
    
    plt.tight_layout()
    
    # Save Figure 1
    fig1_path = os.path.join(output_dir, f'{save_prefix}_solution_comparison.png')
    fig1.savefig(fig1_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {fig1_path}")
    plt.close(fig1)
    
    # ========================================================================
    # Compute errors at t = 1 for all step sizes
    # ========================================================================
    
    for h in step_sizes:
        # Euler solution
        t_euler, y_euler, euler_evals = euler_method(f, t_span, y0, h)
        euler_error = abs(y_euler[-1] - exact_solution(t_span[1]))
        euler_errors.append(euler_error)
        euler_func_evals.append(euler_evals)
        
        # RK4 solution
        t_rk4, y_rk4, rk4_evals = rk4_method(f, t_span, y0, h)
        rk4_error = abs(y_rk4[-1] - exact_solution(t_span[1]))
        rk4_errors.append(rk4_error)
        rk4_func_evals.append(rk4_evals)
    
    # Solve with scipy's solve_ivp (adaptive RK45)
    sol_adaptive = solve_ivp(lambda t, y: f(t, y[0]), t_span, [y0], 
                              method='RK45', rtol=1e-10, atol=1e-10)
    adaptive_error = abs(sol_adaptive.y[0, -1] - exact_solution(t_span[1]))
    adaptive_steps = len(sol_adaptive.t)
    
    # ========================================================================
    # Perform Richardson extrapolation
    # ========================================================================
    
    euler_extrap = richardson_extrapolation(step_sizes, euler_errors)
    rk4_extrap = richardson_extrapolation(step_sizes, rk4_errors)
    
    # ========================================================================
    # FIGURE 2: Enhanced convergence plot with extrapolation
    # ========================================================================
    
    fig2, ax = plt.subplots(figsize=(14, 9))
    
    # Measured points
    ax.loglog(step_sizes, euler_errors, 'ro', markersize=12, 
              label='Euler (measured)', zorder=5, markeredgewidth=2, markeredgecolor='darkred')
    ax.loglog(step_sizes, rk4_errors, 'bs', markersize=12, 
              label='RK4 (measured)', zorder=5, markeredgewidth=2, markeredgecolor='darkblue')
    
    # Extrapolation lines
    if not np.isnan(euler_extrap['coefficient']):
        ax.loglog(euler_extrap['h_predict'], euler_extrap['error_predict'], 
                  'r--', linewidth=2.5, alpha=0.6, label='Euler (extrapolated)', zorder=2)
    
    if not np.isnan(rk4_extrap['coefficient']):
        ax.loglog(rk4_extrap['h_predict'], rk4_extrap['error_predict'], 
                  'b--', linewidth=2.5, alpha=0.6, label='RK4 (extrapolated)', zorder=2)
    
    # Target accuracy lines
    target_errors = [1e-4, 1e-6, 1e-8, 1e-10]
    for target in target_errors:
        ax.axhline(y=target, color='gray', linestyle=':', alpha=0.4, linewidth=1)
        ax.text(step_sizes[0]*1.5, target, f'{target:.0e}', 
                fontsize=10, va='bottom', ha='left', color='gray', fontweight='bold')
    
    # Mark required step sizes for 1e-8 target
    if not np.isnan(euler_extrap['coefficient']):
        h_euler_1e8 = euler_extrap['h_for_target'](1e-8)
        if not np.isnan(h_euler_1e8) and h_euler_1e8 > 0:
            ax.plot(h_euler_1e8, 1e-8, 'r*', markersize=20, 
                    label=f'Euler h for 1e-8: {h_euler_1e8:.2e}', zorder=6)
    
    if not np.isnan(rk4_extrap['coefficient']):
        h_rk4_1e8 = rk4_extrap['h_for_target'](1e-8)
        if not np.isnan(h_rk4_1e8) and h_rk4_1e8 > 0:
            ax.plot(h_rk4_1e8, 1e-8, 'b*', markersize=20, 
                    label=f'RK4 h for 1e-8: {h_rk4_1e8:.2e}', zorder=6)
    
    # Reference lines
    h_ref = np.array(step_sizes)
    euler_ref = euler_errors[0] * (h_ref / h_ref[0])
    rk4_ref = rk4_errors[0] * (h_ref / h_ref[0])**4
    
    ax.loglog(h_ref, euler_ref, 'k--', linewidth=2, alpha=0.4, 
              label='O(h) reference', zorder=1)
    ax.loglog(h_ref, rk4_ref, 'k:', linewidth=2, alpha=0.4, 
              label='O(h⁴) reference', zorder=1)
    
    ax.set_xlabel('Step Size h', fontsize=15, fontweight='bold')
    ax.set_ylabel('Error at t = 1', fontsize=15, fontweight='bold')
    ax.set_title(f'Problem {problem_name}: Convergence Analysis with Richardson Extrapolation', 
                 fontsize=17, fontweight='bold')
    ax.legend(fontsize=11, loc='best', framealpha=0.95)
    ax.grid(True, which='both', alpha=0.3, linestyle='--')
    
    # Enhanced information box
    extrap_info = ""
    if not np.isnan(euler_extrap['coefficient']):
        extrap_info += f"Euler: Error ≈ {euler_extrap['coefficient']:.2e} × h^{euler_extrap['order']:.2f}\n"
    else:
        extrap_info += f"Euler: Machine precision reached\n"
    
    if not np.isnan(rk4_extrap['coefficient']):
        extrap_info += f"RK4: Error ≈ {rk4_extrap['coefficient']:.2e} × h^{rk4_extrap['order']:.2f}\n"
    else:
        extrap_info += f"RK4: Machine precision reached\n"
    
    # Calculate required step sizes
    req_info = "\nFor Error < 1e-8:"
    if not np.isnan(euler_extrap['coefficient']):
        h_euler_req = euler_extrap['h_for_target'](1e-8)
        if not np.isnan(h_euler_req) and h_euler_req > 0:
            steps_euler_req = int(np.ceil(1/h_euler_req))
            req_info += f"\n  Euler: h < {h_euler_req:.2e} ({steps_euler_req} steps)"
    
    if not np.isnan(rk4_extrap['coefficient']):
        h_rk4_req = rk4_extrap['h_for_target'](1e-8)
        if not np.isnan(h_rk4_req) and h_rk4_req > 0:
            steps_rk4_req = int(np.ceil(1/h_rk4_req))
            req_info += f"\n  RK4: h < {h_rk4_req:.2e} ({steps_rk4_req} steps)"
    
    textstr = (f'Adaptive RK45:\n'
               f'  Error: {adaptive_error:.2e}\n'
               f'  Steps: {adaptive_steps}\n'
               f'  Evals: ~{adaptive_steps * 6}\n\n'
               f'Richardson Extrapolation:\n'
               f'{extrap_info}'
               f'{req_info}')
    
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.95, 
                 edgecolor='navy', linewidth=2)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, family='monospace')
    
    plt.tight_layout()
    
    # Save Figure 2
    fig2_path = os.path.join(output_dir, f'{save_prefix}_convergence_extrapolation.png')
    fig2.savefig(fig2_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {fig2_path}")
    plt.close(fig2)
    
    # ========================================================================
    # Print comprehensive results
    # ========================================================================
    
    print_comprehensive_results(problem_name, equation, step_sizes, 
                                euler_errors, rk4_errors,
                                euler_func_evals, rk4_func_evals,
                                adaptive_error, adaptive_steps,
                                euler_extrap, rk4_extrap)
    
    # Return comprehensive results
    return {
        'euler_errors': euler_errors,
        'rk4_errors': rk4_errors,
        'euler_evals': euler_func_evals,
        'rk4_evals': rk4_func_evals,
        'adaptive_error': adaptive_error,
        'adaptive_steps': adaptive_steps,
        'euler_extrap': euler_extrap,
        'rk4_extrap': rk4_extrap,
        'step_sizes': step_sizes
    }


# ============================================================================
# PART 6: UNIFIED COMPREHENSIVE SUMMARY
# ============================================================================

def create_comprehensive_summary(results_a, results_b, results_c):
    """
    Create unified summary table and heatmap visualization
    Addresses Module 2 feedback about redundant table structure
    """
    print(f"\n{'='*140}")
    print("UNIFIED COMPREHENSIVE SUMMARY: ALL THREE PROBLEMS")
    print(f"{'='*140}")
    
    problems = [
        ('(a) y\'=t', results_a),
        ('(b) y\'=2(t+1)y', results_b),
        ('(c) y\'=1/y²', results_c)
    ]
    
    # Analysis at h=0.05 (index 1)
    h_idx = 1
    h_value = 0.05
    
    print(f"\n{'='*140}")
    print(f"COMPARISON AT h = {h_value}")
    print(f"{'='*140}")
    print(f"{'Problem':<20} {'Euler Error':<15} {'RK4 Error':<15} {'Error Ratio':<13} "
          f"{'Euler Evals':<13} {'RK4 Evals':<13} {'Avg Conv Rate':<15}")
    print(f"{'-'*140}")
    
    summary_data = []
    
    for name, results in problems:
        euler_err = results['euler_errors'][h_idx]
        rk4_err = results['rk4_errors'][h_idx]
        ratio = euler_err / rk4_err if rk4_err > 1e-16 else np.inf
        
        euler_evals = results['euler_evals'][h_idx]
        rk4_evals = results['rk4_evals'][h_idx]
        
        # Calculate average convergence rate for Euler
        euler_rates = []
        for i in range(len(results['euler_errors'])-1):
            if results['euler_errors'][i] > 1e-16 and results['euler_errors'][i+1] > 1e-16:
                rate = np.log(results['euler_errors'][i]/results['euler_errors'][i+1]) / \
                       np.log(results['step_sizes'][i]/results['step_sizes'][i+1])
                euler_rates.append(rate)
        
        avg_rate = np.mean(euler_rates) if euler_rates else np.nan
        
        print(f"{name:<20} {euler_err:<15.3e} {rk4_err:<15.3e} {ratio:<13.2e} "
              f"{euler_evals:<13} {rk4_evals:<13} {avg_rate:<15.4f}")
        
        summary_data.append({
            'name': name,
            'euler_error': euler_err,
            'rk4_error': rk4_err,
            'ratio': ratio,
            'euler_evals': euler_evals,
            'rk4_evals': rk4_evals,
            'conv_rate': avg_rate
        })
    
    print(f"{'-'*140}")
    
    # Error magnitude heatmap data
    print(f"\n{'='*140}")
    print("ERROR MAGNITUDE ANALYSIS (log10 scale)")
    print(f"{'='*140}")
    print(f"{'Problem':<20} {'Euler log10(Error)':<25} {'RK4 log10(Error)':<25} {'Magnitude Gap':<20}")
    print(f"{'-'*140}")
    
    for data in summary_data:
        euler_log = np.log10(data['euler_error']) if data['euler_error'] > 0 else -np.inf
        rk4_log = np.log10(data['rk4_error']) if data['rk4_error'] > 0 else -np.inf
        gap = euler_log - rk4_log
        
        print(f"{data['name']:<20} {euler_log:<25.2f} {rk4_log:<25.2f} {gap:<20.2f}")
    
    # Extrapolation comparison
    print(f"\n{'='*140}")
    print("RICHARDSON EXTRAPOLATION COEFFICIENTS")
    print(f"{'='*140}")
    print(f"{'Problem':<20} {'Euler C':<15} {'Euler p':<10} {'RK4 C':<15} {'RK4 p':<10}")
    print(f"{'-'*140}")
    
    for name, results in problems:
        euler_C = results['euler_extrap']['coefficient']
        euler_p = results['euler_extrap']['order']
        rk4_C = results['rk4_extrap']['coefficient']
        rk4_p = results['rk4_extrap']['order']
        
        euler_C_str = f"{euler_C:.2e}" if not np.isnan(euler_C) else "N/A"
        euler_p_str = f"{euler_p:.3f}" if not np.isnan(euler_p) else "N/A"
        rk4_C_str = f"{rk4_C:.2e}" if not np.isnan(rk4_C) else "N/A"
        rk4_p_str = f"{rk4_p:.3f}" if not np.isnan(rk4_p) else "N/A"
        
        print(f"{name:<20} {euler_C_str:<15} {euler_p_str:<10} {rk4_C_str:<15} {rk4_p_str:<10}")
    
    print(f"{'='*140}\n")
    
    # ========================================================================
    # Create visual heatmap
    # ========================================================================
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Comprehensive Method Comparison Across All Problems', 
                 fontsize=18, fontweight='bold')
    
    # Heatmap 1: Error magnitude
    ax1 = axes[0]
    problem_names = ['Problem (a)\ny\'=t', 'Problem (b)\ny\'=2(t+1)y', 'Problem (c)\ny\'=1/y²']
    methods = ['Euler', 'RK4']
    
    # Create error matrix (log10)
    error_matrix = np.zeros((len(methods), len(problems)))
    for j, (name, results) in enumerate(problems):
        error_matrix[0, j] = np.log10(results['euler_errors'][h_idx])
        rk4_err = results['rk4_errors'][h_idx]
        error_matrix[1, j] = np.log10(rk4_err) if rk4_err > 1e-16 else -16    
    # Create heatmap
    im1 = ax1.imshow(error_matrix, cmap='RdYlGn_r', aspect='auto', vmin=-10, vmax=0)
    
    # Set ticks
    ax1.set_xticks(np.arange(len(problem_names)))
    ax1.set_yticks(np.arange(len(methods)))
    ax1.set_xticklabels(problem_names, fontsize=11)
    ax1.set_yticklabels(methods, fontsize=12, fontweight='bold')
    
    # Rotate x labels
    plt.setp(ax1.get_xticklabels(), rotation=0, ha="center")
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('log₁₀(Error)', fontsize=12, fontweight='bold')
    
    # Add text annotations
    for i in range(len(methods)):
        for j in range(len(problem_names)):
            text = ax1.text(j, i, f'{error_matrix[i, j]:.1f}',
                           ha="center", va="center", color="black", fontsize=12, fontweight='bold')
    
    ax1.set_title(f'Error Magnitude at h={h_value}', fontsize=14, fontweight='bold')
    
    # Heatmap 2: Function evaluations
    ax2 = axes[1]
    
    # Create evaluations matrix
    evals_matrix = np.zeros((len(methods), len(problems)))
    for j, (name, results) in enumerate(problems):
        evals_matrix[0, j] = results['euler_evals'][h_idx]
        evals_matrix[1, j] = results['rk4_evals'][h_idx]
    
    # Create heatmap
    im2 = ax2.imshow(evals_matrix, cmap='YlOrRd', aspect='auto')
    
    # Set ticks
    ax2.set_xticks(np.arange(len(problem_names)))
    ax2.set_yticks(np.arange(len(methods)))
    ax2.set_xticklabels(problem_names, fontsize=11)
    ax2.set_yticklabels(methods, fontsize=12, fontweight='bold')
    
    # Rotate x labels
    plt.setp(ax2.get_xticklabels(), rotation=0, ha="center")
    
    # Add colorbar
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Function Evaluations', fontsize=12, fontweight='bold')
    
    # Add text annotations
    for i in range(len(methods)):
        for j in range(len(problem_names)):
            text = ax2.text(j, i, f'{int(evals_matrix[i, j])}',
                           ha="center", va="center", color="black", fontsize=12, fontweight='bold')
    
    ax2.set_title(f'Computational Cost at h={h_value}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save heatmap
    heatmap_path = os.path.join(output_dir, 'comprehensive_heatmap_comparison.png')
    fig.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {heatmap_path}")
    plt.close(fig)
    
    return summary_data


# ============================================================================
# PART 7: FAILURE MODE ANALYSIS
# ============================================================================

def test_method_limits():
    """
    Comprehensive failure mode analysis
    Addresses Module 2 feedback about incomplete method comparison
    """
    print(f"\n{'='*120}")
    print("PART 7: FAILURE MODE ANALYSIS - When Do Methods Break Down?")
    print(f"{'='*120}")
    
    # ========================================================================
    # TEST 1: Stiff problem (Euler stability limit)
    # ========================================================================
    
    print(f"\n{'[TEST 1] STIFF PROBLEM: Euler Stability Boundary'}")
    print(f"{'-'*120}")
    print(f"Problem: y' = -50y, y(0) = 1")
    print(f"Stability condition for Euler: h < 2/|λ| = 2/50 = 0.04")
    print(f"Expected: Euler becomes unstable when h > 0.04\n")
    
    def f_stiff(t, y):
        return -50 * y
    
    def exact_stiff(t):
        return np.exp(-50 * t)
    
    test_h_stiff = [0.05, 0.045, 0.04, 0.035, 0.03, 0.02]
    
    print(f"{'h':<10} {'Euler y(1)':<18} {'RK4 y(1)':<18} {'Exact y(1)':<18} {'Euler Status':<20}")
    print(f"{'-'*120}")
    
    exact_val = exact_stiff(1.0)
    
    for h in test_h_stiff:
        try:
            t_euler, y_euler, _ = euler_method(f_stiff, (0, 1), 1.0, h)
            euler_val = y_euler[-1]
            euler_error = abs(euler_val - exact_val)
            
            # Check for oscillatory behavior or divergence
            if euler_error > 1.0:
                status = "UNSTABLE (diverged)"
            elif any(np.diff(np.sign(np.diff(y_euler)))):
                status = "UNSTABLE (oscillatory)"
            else:
                status = "STABLE"
            
        except:
            euler_val = np.nan
            status = "NUMERICAL FAILURE"
        
        try:
            t_rk4, y_rk4, _ = rk4_method(f_stiff, (0, 1), 1.0, h)
            rk4_val = y_rk4[-1]
        except:
            rk4_val = np.nan
        
        euler_str = f"{euler_val:.6e}" if not np.isnan(euler_val) else "OVERFLOW"
        rk4_str = f"{rk4_val:.6e}" if not np.isnan(rk4_val) else "OVERFLOW"
        
        print(f"{h:<10.4f} {euler_str:<18} {rk4_str:<18} {exact_val:<18.6e} {status:<20}")
    
    print(f"\n{'Key Finding:'} Euler method is STABILITY-LIMITED for stiff problems.")
    print(f"             RK4 has better stability properties but is not A-stable.")
    print(f"             For truly stiff problems, implicit methods (e.g., BDF) are needed.\n")
    
    # ========================================================================
    # TEST 2: Very smooth problem (when is RK4 overkill?)
    # ========================================================================
    
    print(f"\n{'[TEST 2] EFFICIENCY TRADE-OFF: When Is RK4 Overkill?'}")
    print(f"{'-'*120}")
    print(f"Problem: y' = t (very smooth), Target accuracy: 1e-4")
    print(f"Question: At what point does Euler become competitive?\n")
    
    def f_simple(t, y):
        return t
    
    def exact_simple(t):
        return 0.5 * t**2 + 1
    
    target_acc = 1e-4
    
    print(f"{'Method':<15} {'h':<10} {'Steps':<10} {'Evals/step':<12} {'Total Evals':<12} {'Error':<15} {'Meets Target?':<15}")
    print(f"{'-'*120}")
    
    # Find h for Euler to meet target
    h_euler_target = 0.1
    for h_test in [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]:
        _, y_euler, euler_evals = euler_method(f_simple, (0, 1), 1.0, h_test)
        error = abs(y_euler[-1] - exact_simple(1.0))
        meets = "YES" if error < target_acc else "NO"
        print(f"{'Euler':<15} {h_test:<10.5f} {int(1/h_test):<10} {1:<12} {euler_evals:<12} {error:<15.3e} {meets:<15}")
        if error < target_acc and h_euler_target == 0.1:
            h_euler_target = h_test
    
    print(f"{'-'*120}")
    
    # RK4 comparison
    for h_test in [0.1, 0.05]:
        _, y_rk4, rk4_evals = rk4_method(f_simple, (0, 1), 1.0, h_test)
        error = abs(y_rk4[-1] - exact_simple(1.0))
        meets = "YES" if error < target_acc else "NO"
        print(f"{'RK4':<15} {h_test:<10.5f} {int(1/h_test):<10} {4:<12} {rk4_evals:<12} {error:<15.3e} {meets:<15}")
    
    print(f"\n{'Analysis:'} For very smooth problems with modest accuracy requirements,")
    print(f"           Euler with small h can be competitive due to fewer function evaluations per step.")
    print(f"           However, for high accuracy requirements, RK4 is vastly superior.\n")
    
    # ========================================================================
    # TEST 3: Nearly singular problem
    # ========================================================================
    
    print(f"\n{'[TEST 3] NUMERICAL ROBUSTNESS: Handling Difficult Cases'}")
    print(f"{'-'*120}")
    print(f"Problem: y' = 1/y², y(0) = 0.1 (starts very close to singularity at y=0)")
    print(f"Challenge: Small y leads to large y', requiring careful step size control\n")
    
    def f_singular(t, y):
        if abs(y) < 1e-10:
            return 1e10  # Safeguard
        return 1 / y**2
    
    def exact_singular(t):
        return (3*t + 0.001)**(1/3)  # Adjusted initial condition
    
    test_h_singular = [0.1, 0.05, 0.01, 0.005, 0.001]
    
    print(f"{'h':<10} {'Euler Success?':<20} {'RK4 Success?':<20} {'Euler Error':<18} {'RK4 Error':<18}")
    print(f"{'-'*120}")
    
    for h in test_h_singular:
        # Euler
        try:
            _, y_euler, _ = euler_method(f_singular, (0, 1), 0.1, h)
            if np.any(np.isnan(y_euler)) or np.any(np.isinf(y_euler)):
                euler_success = "FAILED (NaN/Inf)"
                euler_error = np.nan
            else:
                euler_success = "SUCCESS"
                euler_error = abs(y_euler[-1] - exact_singular(1.0))
        except:
            euler_success = "FAILED (exception)"
            euler_error = np.nan
        
        # RK4
        try:
            _, y_rk4, _ = rk4_method(f_singular, (0, 1), 0.1, h)
            if np.any(np.isnan(y_rk4)) or np.any(np.isinf(y_rk4)):
                rk4_success = "FAILED (NaN/Inf)"
                rk4_error = np.nan
            else:
                rk4_success = "SUCCESS"
                rk4_error = abs(y_rk4[-1] - exact_singular(1.0))
        except:
            rk4_success = "FAILED (exception)"
            rk4_error = np.nan
        
        euler_err_str = f"{euler_error:.3e}" if not np.isnan(euler_error) else "N/A"
        rk4_err_str = f"{rk4_error:.3e}" if not np.isnan(rk4_error) else "N/A"
        
        print(f"{h:<10.4f} {euler_success:<20} {rk4_success:<20} {euler_err_str:<18} {rk4_err_str:<18}")
    
    print(f"\n{'Key Finding:'} Both methods require sufficiently small h near singularities.")
    print(f"             Adaptive step size control is crucial for such problems.")
    print(f"             Fixed-step methods can fail catastrophically without proper safeguards.\n")
    
    print(f"{'='*120}\n")


# ============================================================================
# PART 8: MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    # Initial conditions
    t_span = (0, 1)
    y0 = 1.0
    
    print("\n" + "="*120)
    print("MODULE 3 - R3.1: COMPREHENSIVE EULER VS RK4 METHOD COMPARISON")
    print("="*120)
    print("\nThis analysis addresses Module 2 feedback:")
    print("  ✓ Complete computational cost tracking (function evaluations)")
    print("  ✓ Richardson extrapolation for predictive analysis")
    print("  ✓ Unified summary tables (not redundant)")
    print("  ✓ Failure mode analysis")
    print("  ✓ Comprehensive discussion of results\n")
    
    # ========================================================================
    # Solve all three problems
    # ========================================================================
    
    print("\n" + "="*120)
    print("SOLVING PROBLEM (a): y' = t")
    print("="*120)
    results_a = solve_problem('(a)', "y' = t", f_a, exact_a, t_span, y0, 'problem_a')
    
    print("\n" + "="*120)
    print("SOLVING PROBLEM (b): y' = 2(t+1)y")
    print("="*120)
    results_b = solve_problem('(b)', "y' = 2(t+1)y", f_b, exact_b, t_span, y0, 'problem_b')
    
    print("\n" + "="*120)
    print("SOLVING PROBLEM (c): y' = 1/y²")
    print("="*120)
    results_c = solve_problem('(c)', "y' = 1/y²", f_c, exact_c, t_span, y0, 'problem_c')
    
    # ========================================================================
    # Create unified summary
    # ========================================================================
    
    create_comprehensive_summary(results_a, results_b, results_c)
    
    # ========================================================================
    # Create summary convergence comparison figure
    # ========================================================================
    
    print(f"\n{'='*120}")
    print("CREATING SUMMARY CONVERGENCE COMPARISON")
    print(f"{'='*120}")
    
    fig_summary, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig_summary.suptitle('Summary: Convergence Comparison for All Three Problems', 
                         fontsize=18, fontweight='bold')
    
    step_sizes = results_a['step_sizes']
    
    problems = [
        ('(a)', results_a, "y' = t"),
        ('(b)', results_b, "y' = 2(t+1)y"),
        ('(c)', results_c, "y' = 1/y²")
    ]
    
    for idx, (name, results, equation) in enumerate(problems):
        ax = axes[idx]
        
        euler_err = results['euler_errors']
        rk4_err = results['rk4_errors']
        
        ax.loglog(step_sizes, euler_err, 'ro-', linewidth=2.5, 
                  markersize=10, label='Euler', alpha=0.8, markeredgewidth=1.5, markeredgecolor='darkred')
        ax.loglog(step_sizes, rk4_err, 'bs-', linewidth=2.5, 
                  markersize=10, label='RK4', alpha=0.8, markeredgewidth=1.5, markeredgecolor='darkblue')
        
        # Reference lines
        h_ref = np.array(step_sizes)
        euler_ref = euler_err[0] * (h_ref / h_ref[0])
        rk4_ref = rk4_err[0] * (h_ref / h_ref[0])**4
        
        ax.loglog(h_ref, euler_ref, 'r--', linewidth=1.5, alpha=0.5, label='O(h)')
        ax.loglog(h_ref, rk4_ref, 'b--', linewidth=1.5, alpha=0.5, label='O(h⁴)')
        
        ax.set_xlabel('Step Size h', fontsize=13, fontweight='bold')
        ax.set_ylabel('Error at t = 1', fontsize=13, fontweight='bold')
        ax.set_title(f'Problem {name}: {equation}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, which='both', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save summary figure
    summary_path = os.path.join(output_dir, 'summary_all_problems.png')
    fig_summary.savefig(summary_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {summary_path}")
    plt.close(fig_summary)
    
    # ========================================================================
    # Run failure mode analysis
    # ========================================================================
    
    test_method_limits()
    
    # ========================================================================
    # Final summary
    # ========================================================================
    
    print("\n" + "="*120)
    print("ANALYSIS COMPLETE")
    print("="*120)
    print(f"\n📊 Total figures generated: 8")
    print(f"   - Problem (a): 2 figures (solution comparison + convergence)")
    print(f"   - Problem (b): 2 figures (solution comparison + convergence)")
    print(f"   - Problem (c): 2 figures (solution comparison + convergence)")
    print(f"   - Comprehensive heatmap comparison: 1 figure")
    print(f"   - Summary convergence plot: 1 figure")
    print(f"\n📁 All figures saved in: {output_dir}")
    print(f"\n✅ Key improvements over Module 2:")
    print(f"   1. Complete function evaluation tracking")
    print(f"   2. Richardson extrapolation with predictions")
    print(f"   3. Unified summary tables (not redundant)")
    print(f"   4. Comprehensive failure mode analysis")
    print(f"   5. Enhanced visualizations with annotations")
    print("\n" + "="*120)


# ============================================================================
# RUN MAIN PROGRAM
# ============================================================================

if __name__ == "__main__":
    main()