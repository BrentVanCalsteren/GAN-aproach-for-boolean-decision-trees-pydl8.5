from typing import List, Optional, Any, Callable, Union

import numpy as np


class Node:
    __slots__ = ('points', 'min_val', 'max_val', 'next_node','split_index', 'depth')

    def __init__(self, points: List[Any], depth: int = 0):
        self.points = sorted(points)  # Keep sorted for consistent splitting
        self.min_val = self.points[0] if self.points else None
        self.max_val = self.points[-1] if self.points else None
        self.next_node = None
        self.split_index = []
        self.depth = depth

    def get_clusters_bonds(self):
        print('getting clusters')
        clusters = []
        pref_index = 0
        print(f'split index: {self.split_index}')
        for index in self.split_index:
            cluster = [self.points[pref_index], self.points[index-1]]
            clusters.append(cluster)
            pref_index = index
        clusters.append([self.points[pref_index], self.points[len(self.points) - 1]])
        return clusters


def get_last_node_clusters(node: Node,depth: int =-1) -> List[List[Any]]:
    if node is None:
        return []
    temp = node
    while temp.next_node is not None or depth > 0:
        temp = temp.next_node
        depth -= 1
    return temp.get_clusters_bonds()


class DivisiveCluster:
    def __init__(self,max_depth: Optional[int] = None,min_cluster_size: int = 1,distance_func: Optional[Callable[[Any, Any], float]] = None):
        self.max_depth = max_depth
        self.min_cluster_size = min_cluster_size
        self.distance_func = distance_func
        self._root = None
        self._original_data = None


    def fit(self, data) -> None:
        if data is None or len(data) == 0:
            self._original_data = []
            self._root = None
            return

        self._original_data = data.copy()
        sorted_points = sorted(data)
        self._root = Node(sorted_points, depth=1)
        self.split_recursive(self._root)

    def split_recursive(self, node: Node) -> None:
        #stop cond check
        if self.max_depth is not None and node.depth >= self.max_depth:
            return

        pts = node.points
        new_split_indx = node.split_index.copy()
        split = -1
        max_gap = 0
        for i in range(len(pts) - 1):
            if i+1 not in node.split_index:
                gap = self._get_gap(pts[i], pts[i + 1])
                if gap > max_gap:
                    max_gap = gap
                    split = i+1

        if split == -1:
            return

        new_split_indx.append(split)
        node.next_node = Node(node.points, node.depth + 1)
        node.next_node.split_index = sorted(new_split_indx)
        self.split_recursive(node.next_node)


    def _get_gap(self, a: Any, b: Any) -> float:
        if self.distance_func is not None:
            return self.distance_func(a, b)

        #num gap
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return float(b - a)

        #str gap
        if isinstance(a, str) and isinstance(b, str):
            return 1.0 if a != b else 0.0

        raise TypeError(
            f"Cant calc gap {type(a).__name__} and {type(b).__name__}"
        )


    def get_clusters(self) -> List[List[float]]:
        if self._root is None:
            return []
        return get_last_node_clusters(self._root)

    def get_clusters_at_depth(self, depth: int) -> List[List[Any]]:
        if self._root is None:
            return []
        return get_last_node_clusters(self._root, depth=depth)
