import numpy as np
from src.usePydl.predictor import default_predictor,gaussian_1D_predictor,uniform_predictor
import time
import matplotlib.pyplot as plt
from pathlib import Path


MAX_TIME = 500

def criteria_test(test, predictor):
    if test == "DEPTH": return depth_test(predictor)
    elif test == "N_SAMPLES": return samples_test(predictor)
    elif test == "N_FEATURES": return feature_test(predictor)

PREDICTORS ={
    "DEFAULT": default_predictor.DefaultPredictor,
    "GAUSSIAN_1D": gaussian_1D_predictor.GaussianPredictor,
    "UNIFORM": uniform_predictor.UNiPredictor
}
TEST_FUN = None


def depth_test(predictor):
    duration = 0
    depth = 1
    while duration < MAX_TIME or depth < 100:
        bools, nums = generate_random_samples(n=1000, features=1000)
        start = time.perf_counter()
        p = predictor(bools, nums, max_depth=depth)
        duration = time.perf_counter() - start
        print(f'loop done, depth: {depth} = {duration}s')
        depth += 1
    return

def samples_test(predictor):
    duration = 0
    n_smaples = 1000
    while duration < MAX_TIME or n_smaples < 10000:
        bools, nums = generate_random_samples(n=n_smaples, features=1000)
        start = time.perf_counter()
        p = predictor(bools, nums, max_depth=100)
        duration = time.perf_counter() - start
        print(f'loop done, samples: {n_smaples} = {duration}s')
        n_smaples += 500
    return

def feature_test(predictor):
    duration = 0
    n_features = 1000
    while duration < MAX_TIME or n_features < 10000:
        bools, nums = generate_random_samples(n=1000, features=n_features)
        start = time.perf_counter()
        p = predictor(bools, nums, max_depth=100)
        duration = time.perf_counter() - start
        print(f'loop done, features: {n_features} = {duration}s')
        n_features += 500
    return


def generate_random_samples(n, features):
    bools= np.random.choice([True, False], size=(n, features)).astype(int)
    nums = np.random.rand(n, features)
    print(bools.shape)
    return bools, nums



def benchmark_samples():
    samples = 100
    increment = 100
    max_samples = 10000
    sizes = []
    times = []


    for number in range(samples,max_samples,increment):
        sizes.append(number)
        start = time.perf_counter()
        #predictor = UNiPredictor(bool_mat[:number,:100],num_mat[:number,:100], max_depth=3,min_sup=1)
        duration = time.perf_counter() - start
        times.append(duration)

    save_results(sizes, times)




def save_results(plot_vals, times, dir='title'):
    results_dir = Path(dir) #save in dir
    results_dir.mkdir(exist_ok=True)

    if not plot_vals:
        print("No successful runs. No plot generated.")
        return

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 1, 1)# 1-row, 1-column grid, first graph
    plt.plot(plot_vals, times, marker='o')#Plot a line with markers
    plt.xlabel('Number of samples')
    plt.ylabel('Time (seconds)')
    plt.title(f'Title')
    plt.grid(True)

    plt.tight_layout() #layout so labels/titles don't overlap

    plot_path = results_dir / f'benchmark_results.png'
    plt.savefig(plot_path, dpi=150)

    plt.show()
    print(f"Plot saved.")




def run_benchmark():
    return

if __name__ == '__main__':
    criteria_test("N_FEATURES", PREDICTORS["GAUSSIAN_1D"])