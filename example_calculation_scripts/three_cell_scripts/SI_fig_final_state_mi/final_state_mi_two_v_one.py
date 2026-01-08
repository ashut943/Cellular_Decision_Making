import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import time
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning, message='divide by zero encountered in log')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in log')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in scalar multiply')


def find_stationary_and_boundary_points(alpha_val=0.7):
    """
    Find stationary points and relevant boundary points for the given alpha value.
    """

    x, y = sp.symbols('x y')
    b_val = (1 - alpha_val) / 3

    f1_expr = (alpha_val - 2*y - x)**2 * y**3 * (1 - (x + 2*b_val + y)) - \
              (y + b_val)**2 * (alpha_val - 3*y - x)**3 * (x + 2*b_val + y)

    f2_expr = (alpha_val - 2*y - x) * x * (1 - (x + 2*b_val + y)) - \
              (x + b_val) * (alpha_val - 3*y - x) * (x + 2*b_val + y)

    f1 = sp.lambdify((x, y), f1_expr, 'numpy')
    f2 = sp.lambdify((x, y), f2_expr, 'numpy')

    # Find interior stationary points (intersections of f1=0 and f2=0)
    solutions = []
    n_samples = 6
    for i in range(n_samples):
        for j in range(n_samples):
            x0 = (i + 0.5) / n_samples * float(alpha_val)
            y0 = (j + 0.5) / n_samples * float(alpha_val) / 3
            try:
                sol = sp.nsolve([f1_expr, f2_expr], [x, y], [x0, y0], tol=1e-14, maxsteps=100)
                sol = [float(sol[0]), float(sol[1])]

                if 0 <= sol[0] <= float(alpha_val) and 0 <= sol[1] <= float(alpha_val) / 3:
                    if not any(np.hypot(sol[0] - s[0], sol[1] - s[1]) < 1e-8 for s in solutions):
                        solutions.append(sol)
            except (ValueError, RuntimeError, ZeroDivisionError):
                pass

    stationary_points = np.array(solutions)

    # Find boundary points on x-axis (relevant_points_c)
    if alpha_val >= 2/3:
        relevant_points_c = [[alpha_val, 0], [0, 0], 
                           [(1-5*b_val+np.sqrt(1-10*b_val+9*b_val**2))/4, 0]]
    else:
        relevant_points_c = [[alpha_val, 0], [0, 0]]

    # Find boundary points on y-axis (relevant_points_a)
    relevant_points_a = [[0, 0]]
    
    # Solve: y^3(1-(2b+y))(alpha-2y)^2 = (b+y)^2(alpha-3y)^3*(2b+y)
    x = sp.symbols('x')
    f_c_expr = (alpha_val - 2*x)**2 * x**3 * (1 - (2*b_val + x)) - \
               (x + b_val)**2 * (alpha_val - 3*x)**3 * (x + 2*b_val)

    f_c = sp.lambdify((x), f_c_expr, 'numpy')

    solutions = []
    n_samples = 6
    for i in range(n_samples):
        x0 = (i + 0.5) / n_samples * float(alpha_val)
        try:
            sol = sp.nsolve([f_c_expr], [x], [x0], tol=1e-14, maxsteps=100)
            sol = [float(sol[0])]

            if 0 <= sol[0] <= float(alpha_val):
                if not any(np.abs(sol[0] - s[0]) < 1e-8 for s in solutions):
                    solutions.append(sol)
        except (ValueError, RuntimeError, ZeroDivisionError):
            pass

    for sol in solutions:
        relevant_points_a.append([0, float(sol[0])])
    
    relevant_points_a = np.array(relevant_points_a)

    # Find boundary points on other boundary (relevant_points_d)
    relevant_points_d = []
    
    x = sp.symbols('x')
    f_d_expr = (alpha_val - 3*x + b_val)**2 * x**3 * (alpha_val-2*x+2*b_val) - \
               (x + b_val)**2 * (alpha_val - 3*x)**3 * (1-(alpha_val-2*x+2*b_val))

    f_d = sp.lambdify((x), f_d_expr, 'numpy')

    solutions = []
    n_samples = 6
    for i in range(n_samples):
        x0 = (i + 0.5) / n_samples * float(alpha_val)
        try:
            sol = sp.nsolve([f_d_expr], [x], [x0], tol=1e-14, maxsteps=100)
            sol = [float(sol[0])]

            if 0 <= sol[0] <= float(alpha_val):
                if not any(np.abs(sol[0] - s[0]) < 1e-8 for s in solutions):
                    solutions.append(sol)
        except (ValueError, RuntimeError, ZeroDivisionError):
            pass

    for sol in solutions:
        relevant_points_d.append([0, float(sol[0])])
    
    relevant_points_d = np.array(relevant_points_d)

    return stationary_points, relevant_points_a, relevant_points_c, relevant_points_d

def xlogx(x):
    return np.where((x <= 0.0) | np.isnan(x), 0.0, x * np.log(x))

def f_vec(v, E):
    a, c = v
    b  = (1-E)/3.0
    d  = E - 3*c - a
    d1 = E - 2*c - a
    p  = a + 2*b + c

    if (d < 0) or (d1 < 0) or (p < 0) or (1-p < 0) or (a+b < 0) or (b+c < 0):
        return np.nan

    return (
        xlogx(a)            + 3*xlogx(b)      + 3*xlogx(c)    + xlogx(d)
      - xlogx(a+b)          - 2*xlogx(b+c)    - xlogx(d1)
      - xlogx(p)            - xlogx(1-p)
    )

def find_min_max_values(E):
    """Find the minimum and maximum function values and their coordinates for given parameter E."""
    
    stationary_points, relevant_points_a, relevant_points_c, relevant_points_d = find_stationary_and_boundary_points(alpha_val=E)
    
    all_points = np.concatenate((stationary_points, relevant_points_a, relevant_points_c, relevant_points_d))
    
    function_values = {}
    for point in all_points:
        function_values[tuple(point)] = f_vec(point, E)
    
    valid_values = {k: v for k, v in function_values.items() if not np.isnan(v)}
    
    if not valid_values:
        return {
            'min_value': np.nan,
            'min_coords': None,
            'max_value': np.nan,
            'max_coords': None,
            'all_points': all_points,
            'function_values': function_values
        }
    
    min_value = min(valid_values.values())
    max_value = max(valid_values.values())
    
    min_coords = None
    max_coords = None
    
    for point, value in valid_values.items():
        if value == min_value:
            min_coords = list(point)
        if value == max_value:
            max_coords = list(point)
    
    return {
        'min_value': min_value,
        'min_coords': min_coords,
        'max_value': max_value,
        'max_coords': max_coords,
        'all_points': all_points,
        'function_values': function_values
    }

def process_single_E(E):
    """Helper function for parallel processing."""
    result = find_min_max_values(E)
    return E, result['min_value'], result['max_value']

def plot_min_max_curves():

    print("Starting computation...")
    start_time = time.time()
    
    E_coarse = np.linspace(0.0, 1.0, 100)
    h_error=[0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0]
    mi_array=[0.5901843248952974, 0.4680798072551431, 0.35860119544483854, 0.27453177805888695, 0.20761884228767935, 0.16270986604250048, 0.12267031452516289, 0.10003861290242169, 0.08095095803675872, 0.06489692897529442, 0.04911791012784311, 0.03415621244425354, 0.028600061962394452, 0.020949383452892323, 0.01476911804865283, 0.009867977002210176, 0.006092135930149012, 0.0033158006294736475, 0.0014343395301089912, 0.0003582239475322241, 0.0]
    
    min_values = []
    max_values = []
    
    try:
        with Pool(processes=min(cpu_count(), 10)) as pool: 
            results = pool.map(process_single_E, E_coarse)
            
        E_values = [r[0] for r in results]
        min_values = [r[1] for r in results]
        max_values = [r[2] for r in results]
        
    except Exception as e:
        print(f"Parallel processing failed, falling back to sequential: {e}")
        for E in E_coarse:
            result = find_min_max_values(E)
            min_values.append(result['min_value'])
            max_values.append(result['max_value'])
        E_values = E_coarse
    
    computation_time = time.time() - start_time
    print(f"Computation completed in {computation_time:.2f} seconds")
    print("Plotting...")
    
    min_values = np.array(min_values)
    max_values = np.array(max_values)
    E_values = np.array(E_values)
    
    plt.figure(figsize=(10, 6))
    
    valid_min_mask = ~np.isnan(min_values)
    valid_max_mask = ~np.isnan(max_values)
    
    if np.any(valid_min_mask):
        plt.plot(E_values[valid_min_mask], min_values[valid_min_mask], 'b-', linewidth=2, label='Min Value', alpha=0.8)
    if np.any(valid_max_mask):
        plt.plot(E_values[valid_max_mask], max_values[valid_max_mask], 'r-', linewidth=2, label='Max Value', alpha=0.8)
    plt.plot(h_error, mi_array, 'g-', linewidth=2, label='Pareto front values', alpha=0.8)
    plt.plot(h_error, mi_array, 'go', markersize=8, alpha=0.8)

    plt.title(r'Three cells: $I(X;(Y,Z))$', fontsize=16, fontweight='bold')
    plt.xlabel(r'$\epsilon$', fontsize=14)
    plt.ylabel('Mutual Information', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    
    plt.savefig("threecellmaxmin_one_v_two.png", dpi=300, bbox_inches='tight')
    plt.savefig("threecellmaxmin_one_v_two.pdf", bbox_inches='tight')
    plt.savefig("threecellmaxmin_one_v_two.svg", bbox_inches='tight')
    plt.show()
    
    print("Plots saved!")


if __name__ == "__main__":
    result = find_min_max_values(0.2)
    print(f"Results for E = 0.2:")
    print(f"Minimum value: {result['min_value']:.6f} at coordinates {result['min_coords']}")
    print(f"Maximum value: {result['max_value']:.6f} at coordinates {result['max_coords']}")
    
    plot_min_max_curves()
