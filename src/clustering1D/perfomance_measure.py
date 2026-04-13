"""
checking how well the clustering scales when you set a max depth of for exemple 100
(so use it for future binning)
"""

import time
import tracemalloc
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


class ClusterNode:
    __slots__ = ('points', 'min_val', 'max_val', 'left', 'right', 'depth')
    def __init__(self, points, depth=0):
        self.points = points
        self.min_val = points[0]
        self.max_val = points[-1]
        self.left = None
        self.right = None
        self.depth = depth

class DivisiveHierarchicalCluster:
    def __init__(self, max_depth=None, min_cluster_size=1):
        self.max_depth = max_depth
        self.min_cluster_size = min_cluster_size
        self._root = None
        self._original_data = None

    def fit(self, data):
        if not data:
            self._original_data = []
            self._root = None
            return
        self._original_data = data.copy()
        sorted_points = sorted(data)
        self._root = ClusterNode(sorted_points, depth=0)
        self._split_recursive(self._root)

    def _split_recursive(self, node):
        if self.max_depth is not None and node.depth >= self.max_depth:
            return
        if len(node.points) < 2 * self.min_cluster_size:
            return
        pts = node.points
        max_gap = -1.0
        split_idx = -1
        for i in range(len(pts)-1):
            gap = pts[i+1] - pts[i]
            if gap > max_gap:
                max_gap = gap
                split_idx = i
        if split_idx == -1:
            return
        left_pts = pts[:split_idx+1]
        right_pts = pts[split_idx+1:]
        if len(left_pts) < self.min_cluster_size or len(right_pts) < self.min_cluster_size:
            return
        node.left = ClusterNode(left_pts, node.depth+1)
        node.right = ClusterNode(right_pts, node.depth+1)
        self._split_recursive(node.left)
        self._split_recursive(node.right)

    def get_clusters(self):
        leaves = []
        self._collect_leaves(self._root, leaves)
        return [leaf.points.copy() for leaf in leaves]

    def _collect_leaves(self, node, leaves):
        if node is None:
            return
        if node.left is None and node.right is None:
            leaves.append(node)
        else:
            self._collect_leaves(node.left, leaves)
            self._collect_leaves(node.right, leaves)

# ------------------------------------------------------------
# Benchmarking utilities
# ------------------------------------------------------------
def generate_dataset(size: int, seed: int = 42) -> list:
    random.seed(seed)
    return [random.uniform(0, 1000) for _ in range(size)]

def measure_performance(sizes: list, max_depth: int = 100, repeats: int = 3) -> dict:
    results = {'sizes': [], 'times': [], 'memories': [], 'n_clusters': []}
    for n in sizes:
        print(f"Testing n = {n} ...")
        times = []
        memories = []
        n_clusters_list = []
        for rep in range(repeats):
            data = generate_dataset(n, seed=rep)
            tracemalloc.start()
            start = time.perf_counter()
            clusterer = DivisiveHierarchicalCluster(max_depth=max_depth, min_cluster_size=1)
            clusterer.fit(data)
            end = time.perf_counter()
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            times.append(end - start)
            memories.append(peak / (1024 * 1024))
            n_clusters_list.append(len(clusterer.get_clusters()))
        results['sizes'].append(n)
        results['times'].append(np.mean(times))
        results['memories'].append(np.mean(memories))
        results['n_clusters'].append(np.mean(n_clusters_list))
        print(f"  Time: {results['times'][-1]:.3f}s, Memory: {results['memories'][-1]:.1f} MB, "
              f"Clusters: {results['n_clusters'][-1]:.0f}")
    return results

def plot_results(results: dict, output_dir: str = "divisive_perf_plots"):
    Path(output_dir).mkdir(exist_ok=True)
    sizes = results['sizes']
    times = results['times']
    memories = results['memories']
    clusters = results['n_clusters']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # Time
    axes[0].plot(sizes, times, 'bo-', linewidth=2)
    axes[0].set_xlabel('Number of points (n)')
    axes[0].set_ylabel('Time (seconds)')
    axes[0].set_title('Time Complexity (max_depth=100)')
    axes[0].grid(True, alpha=0.3)
    # Memory
    axes[1].plot(sizes, memories, 'ro-', linewidth=2)
    axes[1].set_xlabel('Number of points (n)')
    axes[1].set_ylabel('Memory (MB)')
    axes[1].set_title('Memory Usage')
    axes[1].grid(True, alpha=0.3)
    # Number of clusters (saturates at 2^max_depth)
    axes[2].plot(sizes, clusters, 'go-', linewidth=2)
    axes[2].set_xlabel('Number of points (n)')
    axes[2].set_ylabel('Number of clusters')
    axes[2].set_title(f'Clusters formed (max_depth={results.get("max_depth",100)})')
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'divisive_performance.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Also save raw data
    df = pd.DataFrame(results)
    df.to_csv('divisive_performance.csv', index=False)
    print(f"\nPlots saved to {output_dir}/, data saved to divisive_performance.csv")

def main():
    print("=" * 60)
    print("Benchmarking Divisive Hierarchical Clustering (max_depth=100)")
    print("=" * 60)
    # Sizes to test – adjust based on your machine (divisive is very fast)
    sizes = [100, 500, 1000, 2000, 5000, 10000, 20000, 50000]
    results = measure_performance(sizes, max_depth=100, repeats=3)
    results['max_depth'] = 100
    plot_results(results)
    print("\n✅ Benchmark completed.")

if __name__ == "__main__":
    main()