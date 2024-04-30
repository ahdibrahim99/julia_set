
import time
from multiprocessing import Pool, TimeoutError
import numpy as np
from julia_par import compute_julia_in_parallel

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

    return results

if __name__ == '__main__':

    problem_sizes = [155, 1100]
    patch_size = 26
    nprocs = [1, 2, 4, 8, 16, 24, 32]
    num_experiments = 3
    xmin, xmax, ymin, ymax = -1.5, 1.5, -1.5, 1.5
    c = complex(-0.2, -0.65) 

    results = run_benchmark(problem_sizes, patch_size, nprocs, num_experiments, xmin, xmax, ymin, ymax ,c)
    print(results)