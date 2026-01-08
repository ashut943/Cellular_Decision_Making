import os

os.environ['JAX_PLATFORM_NAME'] = 'cpu'
#need to force JAX to use cpu as on mac (to avoid errors)

import numpy as np
import h5py
import matplotlib.pyplot as plt


def xlogx(x):
    return np.where(x == 0.0, 0.0, x * np.log(x))

def check(E):
    b=(1-E)/3
    if E<=0.25:
        min_c=-xlogx(b)+xlogx(E)-4*b*np.log(2)-2*(1-2*b)*np.log(1-2*b)
        max_c=xlogx(E+b)+2*xlogx(b)-2*xlogx(E+2*b)-2*xlogx(1-(E+2*b))
        min_a=-xlogx(b)+xlogx(E)-4*b*np.log(2)-2*(1-2*b)*np.log(1-2*b)
        max_a=xlogx(b)+2*xlogx(b+(E/3))+xlogx(E/3)-2*xlogx(E/3+2*b)-2*xlogx(1-(E/3+2*b))

        min_overall=min(min_c,min_a)
        max_overall=max(max_c,max_a)
        return min_overall, max_overall
    else:
        min_c=0
        max_c_1=xlogx(0.5-b)+2*xlogx(b)+xlogx(E-0.5+2*b)+xlogx(2)
        max_c_2=xlogx(E+b)+2*xlogx(b)-2*xlogx(E+2*b)-2*xlogx(1-(E+2*b))
        max_c=np.max([max_c_1,max_c_2])
        min_a=0
        max_a=xlogx(b)+2*xlogx(b+(E/3))+xlogx(E/3)-2*xlogx(E/3+2*b)-2*xlogx(1-(E/3+2*b))
        min_overall=min(min_c,min_a)
        max_overall=max(max_c,max_a)
        return min_overall, max_overall


if __name__ == "__main__":
    # generate E values from 0 to 1 with more points for smoother curves
    E_values = np.linspace(0, 1, 5000)
    h_error=[0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0]

    mi_array=[0.1634992634538257, 0.13655127341100814, 0.11178688659911662, 0.0888529953527738, 0.07121591727289611, 0.05989364261047447, 0.04699612044008683, 0.04004936722478525, 0.03350801021328531, 0.027611583949070484, 0.021375293070980184, 0.015162119872116908, 0.012940732957458456, 0.009637162252450149, 0.006896871731404519, 0.004671787638512559, 0.0029207952790796243, 0.0016083541983449856, 0.0007033070415314868, 0.00017742993895361603, 0.0]
    mi_array=np.array(mi_array)
    print(mi_array.shape)
    print(mi_array)

    # calculate min and max values for each E
    min_values = []
    max_values = []
    
    for E in E_values:
        min_val, max_val = check(E)
        min_values.append(min_val)
        max_values.append(max_val)
    
    # Create the plot with matplotlib
    plt.figure(figsize=(10, 6))
    
    # Plot min and max curves
    plt.plot(E_values, min_values, 'b-', linewidth=2, label='Min Value', alpha=0.8)
    plt.plot(E_values, max_values, 'r-', linewidth=2, label='Max Value', alpha=0.8)
    plt.plot(h_error, mi_array, 'g-', linewidth=2, label='Pareto front values', alpha=0.8)
    plt.plot(h_error, mi_array, 'go', markersize=8, alpha=0.8)

    plt.title(r'Three cells: $I(X;Y)$', fontsize=16, fontweight='bold')
    plt.xlabel(r'$\epsilon$', fontsize=14)
    plt.ylabel('Mutual Information', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    
    # Save and show the plot
    plt.savefig("check_function_plot.png", dpi=300, bbox_inches='tight') 
    plt.savefig("check_function_plot.pdf", bbox_inches='tight')
    plt.savefig("check_function_plot.svg", bbox_inches='tight')
    plt.show()
    print("Plots saved!")

