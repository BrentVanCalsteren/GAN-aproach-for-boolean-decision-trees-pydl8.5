import re
from typing import Dict, List, Optional
import numpy as np
from usePydl.predictors.helpers.interval import Intervals, Interval

def parse_values(value_str) -> List[float]:
    #handles all possible values
    if isinstance(value_str, (int, float, np.integer, np.floating)): return [float(value_str)]
    if value_str is None: return []
    txt = str(value_str)
    #robust matches to find all +/-1(.0)(E+/-1) (so even sintific not)
    matches = re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?",txt)
    return [float(x) for x in matches]


def _parse_single_value(value_str) -> Optional[float]:
    if value_str is None: return None

    for token in reversed(value_str.split("_")):
        try: return float(token)
        except Exception: pass

    vals = parse_values(value_str)
    if vals: return float(vals[-1])
    return None


def _feature_ids_for_split(split_val, feat_id=0) -> List[int]:
    #is function for handing splits that contain multiple split features
    if isinstance(split_val, str) and "lincomp" in split_val:
        parsed = None #not implemented
        if parsed is not None:
            f1, f2, _, _, _ = parsed
            return list(dict.fromkeys([int(f1), int(f2)]))
    return [feat_id]


def _parse_interval_bounds(value_str):
    #get the interval bounds out of a split
    if value_str is None:
        return None

    txt = str(value_str)
    vals = []
    #multiple values
    if "interval_" in txt:
        tail = txt.split("interval_", 1)[1]
        for token in tail.split("_"):
            if token == "": continue
            try: vals.append(float(token))
            except Exception: pass
        # Fallback to regex parsing.
        if len(vals) < 2: vals = parse_values(tail)
    #single values
    else: vals = parse_values(txt)

    if len(vals) < 2: return None
    lo, hi = sorted(vals[:2])
    return float(lo), float(hi)

def add_left_interval(value_str, intervals: Intervals):
    #path take L turn == true to expression
    if value_str is None: return
    if "lincomp" in value_str: #not implemented yet
        raise ValueError('Not implemented')

    if "bigger_eq" in value_str:
        val = _parse_single_value(value_str)
        if val is not None: intervals.add_interval(val, intervals.max_val, "closed")
        return

    if "smaller_eq" in value_str:
        val = _parse_single_value(value_str)
        if val is not None: intervals.add_interval(intervals.min_val, val, "closed")
        return

    if "even" in value_str:
        val = _parse_single_value(value_str)
        if val is not None: intervals.add_interval(val, val, "closed")
        return

    if "interval" in value_str:
        bounds = _parse_interval_bounds(value_str)
        if bounds is not None:
            lo, hi = bounds
            intervals.add_interval(lo, hi, "closed")
        return

    #fallback
    val = _parse_single_value(value_str)
    if val is not None: intervals.add_interval(intervals.min_val, val, "closed")


def add_right_interval(value_str, intervals: Intervals):
    # path take R turn == false to expression
    if value_str is None: return

    if "lincomp" in value_str: raise ValueError('Not implemented')

    if "bigger_eq" in value_str:
        val = _parse_single_value(value_str)
        if val is not None: intervals.add_interval(intervals.min_val, val, "half-closed")
        return

    if "smaller_eq" in value_str:
        val = _parse_single_value(value_str)
        if val is not None: intervals.add_interval(val, intervals.max_val, "half-open")
        return

    if "even" in value_str:
        val = _parse_single_value(value_str)
        if val is not None:
            intervals.add_constraint_union([
                Interval(intervals.min_val, val, "half-closed"),
                Interval(val, intervals.max_val, "half-open")])
        return

    if "interval" in value_str:
        bounds = _parse_interval_bounds(value_str)
        if bounds is not None:
            lo, hi = bounds
            intervals.add_constraint_union([
                Interval(intervals.min_val, lo, "half-closed"),
                Interval(hi, intervals.max_val, "half-open")])
        return
    #fallb
    val = _parse_single_value(value_str)
    if val is not None: intervals.add_interval(val, intervals.max_val, "half-open")

def add_intervals_for_feat(path, start_interval: Intervals) -> Intervals:
    if path is None: return start_interval
    intervals = start_interval

    try: split_vals, directions = path
    except Exception: return intervals

    for split_val, direction in zip(split_vals, directions):
        if direction == "R": add_right_interval(split_val, intervals)
        else: add_left_interval(split_val, intervals)

        if intervals.is_empty: break
    return intervals

def get_constraints_from_path_dict(path_dict):
    if path_dict is None:
        print('no path given')
        return []

    constraints = []
    seen = set()

    for fid, pair in path_dict.items():
        if pair is None: continue
        try: split_vals, directions = pair
        except Exception: continue

        for split_val, direction in zip(split_vals, directions):
            feat_ids = _feature_ids_for_split(split_val, fid)
            key = (tuple(feat_ids), split_val, direction)
            if key in seen: continue

            seen.add(key)
            constraints.append((feat_ids, split_val, direction))

    return constraints


def apply_path_constraint(feat_ids, split_val, direction, intervals_by_feat):
    left_branch = (direction != "R")

    if "lincomp" in split_val: raise ValueError('Not implemented')
    for fid in feat_ids:
        fid = int(fid)
        if fid not in intervals_by_feat: continue
        if left_branch: add_left_interval(split_val, intervals_by_feat[fid])
        else: add_right_interval(split_val, intervals_by_feat[fid])


def calc_intervals_of_path(path, feat_history, max_passes: int = 10):
    n_features = len(feat_history.feature_info_list)

    intervals_by_feat = {f_id: Intervals(f_id, feat_history) for f_id in range(n_features)}

    if path is None: return intervals_by_feat

    if isinstance(path, dict): #handles every possible form you can pass a path
        if "constraints" in path: constraints = list(path["constraints"])
        elif "path" in path: constraints = get_constraints_from_path_dict(path["path"])
        else: constraints = get_constraints_from_path_dict(path)
    else: constraints = get_constraints_from_path_dict(path)

    def snapshot(): #f_id: str(interval)
        return {
            f_id: [repr(inter) for inter in intervals.interval_list]
            for f_id, intervals in intervals_by_feat.items()
        }

    prev = snapshot()
    for _ in range(max_passes):
        for feat_ids, split_val, direction in constraints:
            apply_path_constraint(feat_ids=feat_ids,split_val=split_val,direction=direction,
                                  intervals_by_feat=intervals_by_feat)

        #a feature domain empty -> path impossible
        if any(inters.is_empty for inters in intervals_by_feat.values()):
            for inters in intervals_by_feat.values(): inters.interval_list = []
            break

        cur = snapshot()
        if cur == prev: break
        prev = cur

    return intervals_by_feat


# ----------------------------------------------------------------------
# Tree
# ----------------------------------------------------------------------

class Tree:

    def __init__(self, tree):
        self.tree: Dict = tree

    def get_leaf_id_for_sample(self, sample):
        current_path = {}

        def add_path_constraint(f_idx, split_val, direction):
            if f_idx not in current_path: current_path[f_idx] = ([], [])
            current_path[f_idx][0].append(split_val)
            current_path[f_idx][1].append(direction)

        def traverse(node):
            if "value" in node:
                leaf_id = (
                    node["value"].get("leaf_id", 0)
                    if isinstance(node["value"], dict) else 0)
                error = float(node.get("error", 0.0))
                rel_prob = (
                    float(node["value"].get("rel_prob", 1.0))
                    if isinstance(node["value"], dict) else 1.0)

                return leaf_id, current_path, error, rel_prob

            feat_id = node.get("feat", 0)
            split_val = node.get("split_val", "")

            if isinstance(split_val, str) and "lincomp" in split_val:
                raise ValueError('Not implemented')

            else:
                try: val = sample[feat_id]
                except Exception:
                    val = sample[0] if hasattr(sample, "__len__") and len(sample) > 0 else 0.0

                if "bigger_eq" in split_val:
                    v = _parse_single_value(split_val)
                    try: cond = bool(val >= v) if v is not None else True
                    except Exception: cond = True

                elif "smaller_eq" in split_val:
                    v = _parse_single_value(split_val)
                    try: cond = bool(val <= v) if v is not None else True
                    except Exception: cond = True

                elif "even" in split_val:
                    v = _parse_single_value(split_val)
                    if v is None: cond = True
                    else:
                        try: cond = bool(np.isclose(float(val), float(v)))
                        except Exception: cond = bool(val == v)

                elif "interval" in split_val:
                    bounds = _parse_interval_bounds(split_val)
                    if bounds is None: cond = True
                    else:
                        try: cond = bool(bounds[0] <= float(val) <= bounds[1])
                        except Exception: cond = False

                else: #fallb
                    vals = parse_values(split_val)
                    if not vals: cond = True
                    else:
                        try: cond = bool(val <= vals[0])
                        except Exception: cond = True

                direction = "L" if cond else "R"
                add_path_constraint(feat_id, split_val, direction)

            if cond: next_node = node.get("left", node.get("right"))
            else: next_node = node.get("right", node.get("left"))

            if next_node is None: return 0, current_path, float(node.get("error", 0.0)), 1.0

            return traverse(next_node)

        return traverse(self.tree)

    def avg_features_used_each_path(self, n_features=None):
        if n_features is None:
            print("can't tell, don't know how many feat there are, n_feat is none")
            return []

        paths = self.get_all_paths()
        n_paths = len(paths)

        split_counts_sum = np.zeros(n_features, dtype=float)

        for path_obj in paths:
            p_dict = path_obj.get("path", {})
            for feat_id, (splits_vals, directions) in p_dict.items():
                feat_idx = int(feat_id)
                if feat_idx < n_features:
                    split_counts_sum[feat_idx] += len(splits_vals)

        if n_paths == 0:
            return split_counts_sum

        avg_counts = split_counts_sum / float(n_paths)
        return avg_counts

    def tree_label_distr_each_leaf(self, samples, labels):
        labels_arr = np.asarray(labels).flatten()
        samples_arr = np.asarray(samples)

        if samples_arr.ndim == 1:
            samples_arr = samples_arr.reshape(1, -1)

        if samples_arr.size == 0:
            return 0.0, {}

        leaf_label_map = {}

        for sample, label in zip(samples_arr, labels_arr):
            leaf_id, _, _, _ = self.get_leaf_id_for_sample(sample)

            if leaf_id not in leaf_label_map:
                leaf_label_map[leaf_id] = []

            leaf_label_map[leaf_id].append(label)

        leaf_purity_info = {}
        total_purity_sum = 0.0
        n_leaves = len(leaf_label_map)

        for l_id, l_labels in leaf_label_map.items():
            l_labels = np.array(l_labels)
            total_count = len(l_labels)

            uniques, counts = np.unique(l_labels, return_counts=True)
            label_probs = dict(zip(uniques, counts / total_count))

            dominant_prob = float(np.max(counts) / total_count)
            total_purity_sum += dominant_prob

            leaf_purity_info[l_id] = {
                "count": total_count,
                "label_probs": label_probs,
                "purity": dominant_prob,
                "num_unique_labels": len(uniques)
            }

        avg_purity = total_purity_sum / n_leaves if n_leaves > 0 else 0.0
        print(f"leaf purtoy: {avg_purity} ")

        return avg_purity, leaf_purity_info

    def get_depth(self):
        def recurse(node, depth):
            if "value" in node:return depth

            left_depth = depth
            right_depth = depth

            if "left" in node: left_depth = recurse(node["left"], depth + 1)

            if "right" in node: right_depth = recurse(node["right"], depth + 1)

            return max(left_depth, right_depth)

        return recurse(self.tree, 0)

    def get_leafs(self):
        leaves = []

        def recurse(node):
            if "value" in node: leaves.append(node)
            else:
                if "left" in node: recurse(node["left"])
                if "right" in node: recurse(node["right"])

        recurse(self.tree)
        return leaves

    def remap_tree(self, feature_history):
        def recurse(node):
            if "value" in node: return

            feat_id, split_val = feature_history.get_feat_split_result(node["feat"])
            node["feat"] = feat_id
            node["split_val"] = split_val

            if "left" in node: recurse(node["left"])
            if "right" in node: recurse(node["right"])

        recurse(self.tree)

    def extend_tree(self, subtree, leaf_id):
        def merge(node):
            if "value" in node:
                l_id = node["value"]["leaf_id"]
                if leaf_id == l_id:
                    print("MERGED!")
                    return subtree
                return node

            if "left" in node:
                node["left"] = merge(node["left"])
            if "right" in node:
                node["right"] = merge(node["right"])

            return node

        self.tree = merge(self.tree)

    def get_intervals_each_path(self, feat_history):
        paths = self.get_all_paths()
        interval_path_dic = {}

        for i, path_obj in enumerate(paths):
            interval_path_dic[i] = calc_intervals_of_path(path_obj, feat_history)

        return interval_path_dic

    def remove_sample_ids_from_leafs(self):
        def recurse(node):
            if "value" in node:
                if isinstance(node["value"], dict):
                    node["value"].pop("sample_ids", None)
            else:
                if "left" in node: recurse(node["left"])
                if "right" in node: recurse(node["right"])

        if self.tree is not None: recurse(self.tree)

    def get_all_paths(self):
        all_paths = []

        def copy_path(p):
            return {k: (v[0].copy(), v[1].copy()) for k, v in p.items()}

        def add_constraint(current_path, current_constraints, feat_ids, split_val, direction):
            feat_ids = list(dict.fromkeys(int(f) for f in feat_ids))

            new_path = copy_path(current_path)

            for f_id in feat_ids:
                if f_id not in new_path: new_path[f_id] = ([], [])
                new_path[f_id][0].append(split_val)
                new_path[f_id][1].append(direction)

            new_constraints = current_constraints + ((tuple(feat_ids), split_val, direction),)

            return new_path, new_constraints

        def path_finder(node, current_path, current_constraints):
            if "value" in node:
                value = node.get("value", {})

                all_paths.append({
                    "path": current_path,
                    "constraints": list(current_constraints),
                    "error": node.get("error", 0),
                    "sample_ids": value.get("sample_ids", []) if isinstance(value, dict) else [],
                    "rel_prob": value.get("rel_prob", 0.0) if isinstance(value, dict) else 0.0,
                    "leaf_id": value.get("leaf_id", 0) if isinstance(value, dict) else 0
                })
                return

            split_val = node.get("split_val", "")
            feat_ids = _feature_ids_for_split(split_val, node.get("feat", 0))

            if "left" in node:
                left_path, left_constraints = add_constraint(current_path,current_constraints,
                    feat_ids,split_val,"L")
                path_finder(node["left"], left_path, left_constraints)

            if "right" in node:
                right_path, right_constraints = add_constraint(
                    current_path,
                    current_constraints,
                    feat_ids,
                    split_val,
                    "R"
                )
                path_finder(node["right"], right_path, right_constraints)

        path_finder(self.tree, {}, ())

        if len(all_paths) != len(self.get_leafs()):
            print("LEN PATHS AND LEAVES SHOULD BE THE SAME")

        return all_paths


def remap_tree(tree, feature_history):
    def recurse(node):
        if "value" in node: return

        feat_id, split_val = feature_history.get_feat_split_result(node["feat"])
        node["feat"] = feat_id
        node["split_val"] = split_val

        if "left" in node: recurse(node["left"])
        if "right" in node: recurse(node["right"])

    recurse(tree)
    return tree


def extend_tree(tree, hist_tree, leaf_id):
    def merge(node):
        if "value" in node:
            l_id = node["value"]["leaf_id"]
            if leaf_id == l_id:
                print("MERGED!")
                return hist_tree
            return node

        if "left" in node: node["left"] = merge(node["left"])
        if "right" in node: node["right"] = merge(node["right"])

        return node

    return merge(tree)