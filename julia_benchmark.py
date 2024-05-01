
import time
from multiprocessing import Pool, TimeoutError
import numpy as np
from julia_par import compute_julia_in_parallel
import matplotlib.pyplot as plt
from julia_curve import c_from_group

# section 2.2
def run_benchmark(problem_sizes, patch_size, nprocs, num_experiments, xmin, xmax, ymin, ymax ,c):
    results = {}
        
    for size in problem_sizes:
        results[size]= {}
        for n in nprocs:
            results[size][n] = {}
            runtimes = []
            for i in range(num_experiments):
                stime = time.perf_counter()
                compute_julia_in_parallel(size, xmin, xmax, ymin, ymax, patch_size, n, c)
                rtime = time.perf_counter() - stime
                runtimes.append(rtime)
            results[size][n]["mean_runtime"] = np.mean(runtimes)

    for size in problem_sizes:
        for n in nprocs:
            results[size][n]["speedup"] = results[size][1]["mean_runtime"] / results[size][n]["mean_runtime"]
            results[size][n]["parallel_efficiency"] = results[size][n]["speedup"] / n

    print(results)

    nprocs = list(results[155].keys())
    mean_runtime_155, mean_runtime_1100 = [results[155][n]['mean_runtime'] for n in nprocs], [results[1100][n]['mean_runtime'] for n in nprocs]
    speedup_155, speedup_1100  = [results[155][n]['speedup'] for n in nprocs], [results[1100][n]['speedup'] for n in nprocs]
    parallel_eff_155, parallel_eff_1100 = [results[155][n]['parallel_efficiency'] for n in nprocs], [results[1100][n]['parallel_efficiency'] for n in nprocs]

    def plot(x_values, y_values_155, y_values_1100, title, x_label, y_label):
        plt.figure(figsize=(10, 6))
        plt.bar([n - 0.22 for n in x_values], y_values_155, width=0.4, label='Size 155')
        plt.bar([n + 0.22 for n in x_values], y_values_1100, width=0.4, label='Size 1100')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.xticks(nprocs)
        plt.legend()
        plt.grid(True)
        plt.show()

    plot(nprocs, mean_runtime_155, mean_runtime_1100, 'Mean Runtime', 'Number of Processors', 'Mean Runtime (s)')
    plot(nprocs, speedup_155, speedup_1100, 'Speedup', 'Number of Processors', 'Speedup')
    plot(nprocs, parallel_eff_155, parallel_eff_1100, 'Parallel Efficiency', 'Number of Processors', 'Parallel Efficiency')

if __name__ == '__main__':

    problem_sizes = [155, 1100]
    patch_size = 26
    nprocs = [1, 2, 4, 8, 16, 24, 32]
    num_experiments = 3
    xmin, xmax, ymin, ymax = -1.5, 1.5, -1.5, 1.5
    
    c = complex(-0.2, -0.65) 
    run_benchmark(problem_sizes, patch_size, nprocs, num_experiments, xmin, xmax, ymin, ymax ,c)
    
    print("**********************************")
    
    c = c_from_group(2, 4) 

    run_benchmark(problem_sizes, patch_size, nprocs, num_experiments, xmin, xmax, ymin, ymax ,c)
    
    