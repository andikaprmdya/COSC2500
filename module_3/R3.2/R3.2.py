import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fsolve
import os

# ============================================================================
# PART 0: CREATE OUTPUT DIRECTORY
# ============================================================================

output_dir = os.path.join('module_3', 'R3.2')
os.makedirs(output_dir, exist_ok=True)
print(f"\n[*] Output directory created: {output_dir}")
print(f"   All figures will be saved here.\n")

# ============================================================================
# PART 1: PROBLEM DEFINITIONS AND EXACT SOLUTIONS
# ============================================================================

# Problem 2.1(a)(i): y'' = y + (2/3)e^t, y(0)=0, y(1)=(1/3)e
def exact_2_1_a_i(t):
    """Exact solution: y = (1/3)te^t"""
    return (1/3) * t * np.exp(t)

# Problem 2.1(a)(ii): y'' = (2+4t^2)y, y(0)=1, y(1)=e
def exact_2_1_a_ii(t):
    """Exact solution: y = e^(t^2)"""
    return np.exp(t**2)

# Problem 2.1(b)(i): y'' = 3y - 2y', y(0)=e^3, y(1)=1
def exact_2_1_b_i(t):
    """Exact solution: y = e^(3-3t)"""
    return np.exp(3 - 3*t)


# ============================================================================
# PART 2: SHOOTING METHOD FOLLOWING THE HINT
# ============================================================================

# Following hint structure: Convert 2nd order ODE to system of 1st order ODEs
# Let y[0] = y, y[1] = y'
# Then: y[0]' = y[1], y[1]' = f(t, y[0], y[1])

def ode_system_2_1_a_i(y, t):
    """
    Convert y'' = y + (2/3)e^t to system of 1st order ODEs
    y[0] = y
    y[1] = y'
    Returns [y', y''] as numpy array
    """
    dydt = np.zeros(2)
    dydt[0] = y[1]
    dydt[1] = y[0] + (2/3)*np.exp(t)
    return dydt


def ode_system_2_1_a_ii(y, t):
    """
    Convert y'' = (2+4t^2)y to system of 1st order ODEs
    """
    dydt = np.zeros(2)
    dydt[0] = y[1]
    dydt[1] = (2 + 4*t**2) * y[0]
    return dydt


def ode_system_2_1_b_i(y, t):
    """
    Convert y'' = 3y - 2y' to system of 1st order ODEs
    """
    dydt = np.zeros(2)
    dydt[0] = y[1]
    dydt[1] = 3*y[0] - 2*y[1]
    return dydt


def shooting_method(ode_func, t_span, y0, yf, initial_guess=0.0):
    """
    Shooting method following the hint structure
    
    Parameters:
    -----------
    ode_func : function
        ODE system function (already converted to 1st order system)
    t_span : tuple
        (t_initial, t_final)
    y0 : float
        Initial boundary condition y(t0)
    yf : float
        Final boundary condition y(tf)
    initial_guess : float
        Initial guess for y'(0)
    
    Returns:
    --------
    dict with solution and statistics
    """
    
    def bc_mismatch(y_prime_0):
        """
        Following hint: Find the roots of this function
        This is the objective function for root-finding
        
        We guess the missing initial condition (y_prime_0),
        solve the IVP, and compare with final boundary condition.
        Want: y(tf) - yf = 0
        """
        # Solve the IVP with guessed initial slope
        t = np.linspace(t_span[0], t_span[1], 100)
        # Extract scalar from array if needed (fsolve passes array)
        y_prime_val = y_prime_0[0] if hasattr(y_prime_0, '__len__') else y_prime_0
        y_initial = np.array([y0, y_prime_val])  # [y(0), y'(0)]
        
        # Solve using odeint (similar to MATLAB's ode45)
        sol = odeint(ode_func, y_initial, t)
        
        # Compare with final BC
        # sol[-1, 0] is y(tf), sol[-1, 1] is y'(tf)
        final_error = sol[-1, 0] - yf
        
        return final_error
    
    # Use root-finder (fsolve, similar to MATLAB's fzero)
    print(f"   Finding optimal initial slope y'(0)...")
    print(f"   Initial guess: {initial_guess:.6f}")
    
    # Find root
    y_prime_0_optimal, info, ier, msg = fsolve(bc_mismatch, initial_guess, full_output=True)
    y_prime_0_optimal = y_prime_0_optimal[0]
    
    print(f"   Optimal y'(0): {y_prime_0_optimal:.10f}")
    print(f"   Final error: {abs(bc_mismatch(y_prime_0_optimal)):.2e}")
    print(f"   Convergence: {'SUCCESS' if ier == 1 else 'FAILED'}")
    print(f"   Function evaluations: {info['nfev']}")
    
    # Get final solution with optimal initial condition
    t = np.linspace(t_span[0], t_span[1], 1000)
    y_initial = np.array([y0, y_prime_0_optimal])
    sol = odeint(ode_func, y_initial, t)
    
    return {
        'success': ier == 1,
        't': t,
        'y': sol[:, 0],
        'y_prime': sol[:, 1],
        'y_prime_0': y_prime_0_optimal,
        'final_error': abs(bc_mismatch(y_prime_0_optimal)),
        'iterations': info['nfev']
    }


# ============================================================================
# PART 3: FINITE DIFFERENCE METHOD FOLLOWING THE HINT
# ============================================================================

def finite_difference_method(problem_name, t_span, y0, yf, N=10):
    """
    Finite difference method following the hint structure
    
    Following hint steps:
    1. Substitute finite difference approximations into ODE
    2. Generate linear system (boundary conditions as 1st and last rows)
    3. Solve and plot
    """
    
    print(f"\n   Setting up finite difference system...")
    print(f"   Grid points: {N}")
    
    # Create grid
    t = np.linspace(t_span[0], t_span[1], N)
    h = t[1] - t[0]
    print(f"   Step size h: {h:.6f}")
    
    # Pre-allocate memory for matrix and vector (following hint)
    A = np.zeros((N, N))
    c = np.zeros(N)
    
    # Boundary conditions (following hint: 1st and last rows)
    A[0, 0] = 1
    c[0] = y0  # First BC
    
    A[N-1, N-1] = 1
    c[N-1] = yf  # End BC
    
    # Make the matrix (following hint structure)
    print(f"   Building coefficient matrix...")
    
    for n in range(1, N-1):
        tn = t[n]
        
        if "2.1(a)(i)" in problem_name:
            # y'' = y + (2/3)e^t
            # Central difference: y'' ≈ (y_{n+1} - 2y_n + y_{n-1})/h²
            # Substituting: (y_{n+1} - 2y_n + y_{n-1})/h² = y_n + (2/3)e^{tn}
            # Rearranging: y_{n-1} - (2 + h²)y_n + y_{n+1} = (2/3)h²e^{tn}
            
            A[n, n-1] = 1
            A[n, n] = -(2 + h**2)
            A[n, n+1] = 1
            c[n] = (2/3) * h**2 * np.exp(tn)
            
        elif "2.1(a)(ii)" in problem_name:
            # y'' = (2+4t²)y
            # (y_{n+1} - 2y_n + y_{n-1})/h² = (2+4t_n²)y_n
            # y_{n-1} - (2 + h²(2+4t_n²))y_n + y_{n+1} = 0
            
            A[n, n-1] = 1
            A[n, n] = -(2 + h**2 * (2 + 4*tn**2))
            A[n, n+1] = 1
            c[n] = 0
            
        elif "2.1(b)(i)" in problem_name:
            # y'' = 3y - 2y'
            # Central diff for y'': (y_{n+1} - 2y_n + y_{n-1})/h²
            # Central diff for y': (y_{n+1} - y_{n-1})/(2h)
            # Substituting: (y_{n+1} - 2y_n + y_{n-1})/h² = 3y_n - 2(y_{n+1} - y_{n-1})/(2h)
            # Multiplying by h²: y_{n+1} - 2y_n + y_{n-1} = 3h²y_n - h(y_{n+1} - y_{n-1})
            # Rearranging: (1 - h)y_{n-1} + (-2 - 3h²)y_n + (1 + h)y_{n+1} = 0
            
            A[n, n-1] = 1 - h
            A[n, n] = -(2 + 3*h**2)
            A[n, n+1] = 1 + h
            c[n] = 0
    
  # Solve the linear system (following hint)
    print(f"   Solving linear system...")
    y = np.linalg.solve(A, c)
    
    # Calculate condition number (Module 2 feedback)
    cond_number = np.linalg.cond(A)
    print(f"   Condition number: {cond_number:.2e}")
    
    print(f"   System solved successfully!")
    
    return {
        'success': True,
        't': t,
        'y': y,
        'A': A,
        'c': c,
        'N': N,
        'h': h,
        'cond_number': cond_number  # NEW
    }




# ============================================================================
# PART 4: VISUALIZATION FUNCTION
# ============================================================================

def visualize_problem(problem_name, equation, t_span, y0, yf, 
                     ode_func, exact_func, shooting_guess, save_prefix):
    """
    Solve and visualize a BVP problem using both methods
    """
    
    print(f"\n{'='*80}")
    print(f"SOLVING PROBLEM: {problem_name}")
    print(f"{'='*80}")
    print(f"Equation: {equation}")
    print(f"Domain: [{t_span[0]}, {t_span[1]}]")
    print(f"Boundary conditions: y({t_span[0]}) = {y0:.6f}, y({t_span[1]}) = {yf:.6f}")
    
    # ========== SHOOTING METHOD ==========
    print(f"\n--- SHOOTING METHOD ---")
    shooting_result = shooting_method(ode_func, t_span, y0, yf, shooting_guess)
    
    # ========== FINITE DIFFERENCE METHOD ==========
    print(f"\n--- FINITE DIFFERENCE METHOD ---")
    fd_result = finite_difference_method(problem_name, t_span, y0, yf, N=20)
    
    # ========== ERROR ANALYSIS ==========
    print(f"\n--- ERROR ANALYSIS ---")
    
    # Shooting error
    y_exact_shooting = exact_func(shooting_result['t'])
    shooting_error = np.max(np.abs(shooting_result['y'] - y_exact_shooting))
    print(f"Shooting method max error: {shooting_error:.2e}")
    
    # Finite difference error
    y_exact_fd = exact_func(fd_result['t'])
    fd_error = np.max(np.abs(fd_result['y'] - y_exact_fd))
    print(f"Finite difference max error: {fd_error:.2e}")
    print(f"Error ratio (FD/Shooting): {fd_error/shooting_error:.2f}x")
    
    # ========== VISUALIZATION ==========
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Solution Comparison
    ax1 = fig.add_subplot(gs[0, :])
    
    t_dense = np.linspace(t_span[0], t_span[1], 1000)
    y_exact_dense = exact_func(t_dense)
    
    ax1.plot(t_dense, y_exact_dense, 'k-', linewidth=3, label='Exact Solution', alpha=0.8)
    ax1.plot(shooting_result['t'], shooting_result['y'], 'r--', linewidth=2, 
             label='Shooting Method', alpha=0.8)
    ax1.plot(fd_result['t'], fd_result['y'], 'bo-', markersize=6, linewidth=2,
             label=f'Finite Difference (N={fd_result["N"]})', alpha=0.7)
    ax1.plot([t_span[0], t_span[1]], [y0, yf], 'gs', markersize=15,
             label='Boundary Conditions', markeredgewidth=2, markeredgecolor='darkgreen')
    
    ax1.set_xlabel('t', fontsize=14, fontweight='bold')
    ax1.set_ylabel('y(t)', fontsize=14, fontweight='bold')
    ax1.set_title(f'{problem_name}: {equation}', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Error Distribution
    ax2 = fig.add_subplot(gs[1, 0])
    
    shooting_error_dist = np.abs(shooting_result['y'] - y_exact_shooting)
    ax2.semilogy(shooting_result['t'], shooting_error_dist, 'r-', linewidth=2, label='Shooting')
    
    fd_error_dist = np.abs(fd_result['y'] - y_exact_fd)
    ax2.semilogy(fd_result['t'], fd_error_dist, 'bo-', markersize=6, linewidth=2, label='Finite Diff')
    
    ax2.set_xlabel('t', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Absolute Error', fontsize=12, fontweight='bold')
    ax2.set_title('Error Distribution', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')
    
    # Plot 3: Finite Difference Matrix Structure
    ax3 = fig.add_subplot(gs[1, 1])
    
    A_sparse = np.abs(fd_result['A']) > 1e-10
    ax3.spy(A_sparse, markersize=5, color='blue')
    ax3.set_title('FD Matrix Sparsity Pattern', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Column', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Row', fontsize=12, fontweight='bold')
    
    # Plot 4: Matrix Structure Detail
    ax4 = fig.add_subplot(gs[1, 2])
    
    # Show actual values of a few rows
    im = ax4.imshow(fd_result['A'], cmap='RdBu', aspect='auto', interpolation='nearest')
    ax4.set_title('FD Matrix Values', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Column', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Row', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax4)
    
    # Plot 5: Method Statistics
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    stats_text = f"""
    COMPREHENSIVE METHOD COMPARISON
    {'='*70}
    
    SHOOTING METHOD:
    • Optimal Initial Slope y'(0): {shooting_result['y_prime_0']:.10f}
    • Function Evaluations: {shooting_result['iterations']}
    • Final Boundary Error: {shooting_result['final_error']:.2e}
    • Maximum Error vs Exact: {shooting_error:.2e}
    • Method: Root-finding + IVP solver (odeint)
    
    FINITE DIFFERENCE METHOD:
    • Grid Points N: {fd_result['N']}
    • Step Size h: {fd_result['h']:.6f}
    • Matrix Size: {fd_result['N']} × {fd_result['N']}
    • Matrix Bandwidth: 3 (tridiagonal structure)
    • Sparsity: {100 * (1 - np.count_nonzero(fd_result['A']) / fd_result['A'].size):.1f}% zeros
    • Maximum Error vs Exact: {fd_error:.2e}
    • Method: Direct linear system solve
    
    COMPARISON:
    • Error Ratio (FD/Shooting): {fd_error/shooting_error:.2f}x
    • Both methods produce accurate solutions
    • Shooting: Continuous solution, requires root-finding
    • Finite Diff: Discrete solution, simpler implementation for linear problems
    """
    
    ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes,
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle(f'Complete Analysis: {problem_name}', fontsize=18, fontweight='bold', y=0.995)
    
    # Save to module_3/R3.2 folder
    fig_path = os.path.join(output_dir, f'{save_prefix}_analysis.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\n[+] Saved: {fig_path}")
    plt.close()


# ============================================================================
# PART 5: CONVERGENCE STUDY
# ============================================================================

def convergence_study(problem_name, equation, t_span, y0, yf, exact_func, save_prefix):
    """
    Study convergence of finite difference method as grid is refined
    """
    
    print(f"\n{'='*80}")
    print(f"CONVERGENCE STUDY: {problem_name}")
    print(f"{'='*80}")
    
    N_values = [10, 20, 40, 80, 160]
    errors = []
    h_values = []
    
    for N in N_values:
        fd_result = finite_difference_method(problem_name, t_span, y0, yf, N=N)
        y_exact = exact_func(fd_result['t'])
        error = np.max(np.abs(fd_result['y'] - y_exact))
        
        errors.append(error)
        h_values.append(fd_result['h'])
        
        print(f"N = {N:4d}, h = {fd_result['h']:.6f}, Max Error = {error:.2e}")
    
    # Calculate convergence rates
    rates = []
    for i in range(len(errors) - 1):
        rate = np.log(errors[i] / errors[i+1]) / np.log(h_values[i] / h_values[i+1])
        rates.append(rate)
    
    avg_rate = np.mean(rates)
    print(f"\nAverage convergence rate: {avg_rate:.3f} (Expected: 2.0 for O(h^2))")

    log_h = np.log(np.array(h_values))
    log_err = np.log(np.array(errors))
    coeffs = np.polyfit(log_h, log_err, 1)
    p_fit = coeffs[0]
    C_fit = np.exp(coeffs[1])

    print(f"\nExtrapolation: error ~= {C_fit:.2e} * h^{p_fit:.3f}")
    print(f"Grid sizes for target errors:")
    for target in [1e-6, 1e-8, 1e-10]:
        h_need = (target / C_fit) ** (1/p_fit)
        N_need = int(np.ceil(1.0 / h_need))
        print(f"  Error < {target:.0e}: N ~= {N_need:,}")
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Log-log plot
    ax1.loglog(h_values, errors, 'bo-', linewidth=2, markersize=10, label='FD Method')
    
    # O(h²) reference line
    h_ref = np.array(h_values)
    error_ref = errors[0] * (h_ref / h_ref[0])**2
    ax1.loglog(h_ref, error_ref, 'r--', linewidth=2, alpha=0.6, label='O(h²) reference')
    
    h_fit = np.logspace(np.log10(h_values[-1]/2), np.log10(h_values[0]*2), 50)
    error_fit = C_fit * h_fit**p_fit
    ax1.loglog(h_fit, error_fit, 'g:', linewidth=2, label=f'Fit: {C_fit:.1e}×h^{p_fit:.2f}')

    ax1.set_xlabel('Step Size h', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Maximum Error', fontsize=14, fontweight='bold')
    ax1.set_title(f'Convergence Analysis: {problem_name}', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3, which='both')
    
    # Convergence rates
    ax2.plot(range(len(rates)), rates, 'go-', linewidth=2, markersize=10, label='Observed Rate')
    ax2.axhline(y=2.0, color='r', linestyle='--', linewidth=2, label='Expected: O(h²)')
    ax2.set_xlabel('Refinement Step', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Convergence Rate', fontsize=14, fontweight='bold')
    ax2.set_title('Observed Convergence Rate', fontsize=16, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 3])
    
    # Add average rate text
    ax2.text(0.5, 0.05, f'Average Rate: {avg_rate:.3f}', 
            transform=ax2.transAxes, fontsize=14, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    plt.tight_layout()
    
    # Save to module_3/R3.2 folder
    fig_path = os.path.join(output_dir, f'{save_prefix}_convergence.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"[+] Saved: {fig_path}")
    plt.close()


# ============================================================================
# PART 6: SOLVE ALL PROBLEMS
# ============================================================================

print("\n" + "="*80)
print("MODULE 3 - R3.2: BVP SOLUTION METHODS")
print("Following Programming Hints Structure")
print("="*80)

# Problem 2.1(a)(i)
visualize_problem(
    "Problem 2.1(a)(i)",
    "y'' = y + (2/3)e^t",
    (0, 1),
    0,
    (1/3)*np.e,
    ode_system_2_1_a_i,
    exact_2_1_a_i,
    0.0,
    "problem_2_1_a_i"
)

convergence_study(
    "Problem 2.1(a)(i)",
    "y'' = y + (2/3)e^t",
    (0, 1),
    0,
    (1/3)*np.e,
    exact_2_1_a_i,
    "problem_2_1_a_i"
)

# Problem 2.1(a)(ii)
visualize_problem(
    "Problem 2.1(a)(ii)",
    "y'' = (2+4t²)y",
    (0, 1),
    1,
    np.e,
    ode_system_2_1_a_ii,
    exact_2_1_a_ii,
    0.0,
    "problem_2_1_a_ii"
)

convergence_study(
    "Problem 2.1(a)(ii)",
    "y'' = (2+4t²)y",
    (0, 1),
    1,
    np.e,
    exact_2_1_a_ii,
    "problem_2_1_a_ii"
)

# Problem 2.1(b)(i)
visualize_problem(
    "Problem 2.1(b)(i)",
    "y'' = 3y - 2y'",
    (0, 1),
    np.exp(3),
    1,
    ode_system_2_1_b_i,
    exact_2_1_b_i,
    -3.0,
    "problem_2_1_b_i"
)

convergence_study(
    "Problem 2.1(b)(i)",
    "y'' = 3y - 2y'",
    (0, 1),
    np.exp(3),
    1,
    exact_2_1_b_i,
    "problem_2_1_b_i"
)

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\n[*] Total figures saved: 6")
print(f"   - 3 comprehensive analysis figures")
print(f"   - 3 convergence study figures")
print(f"\n[*] All figures saved in: {output_dir}")
print("\n" + "="*80)
