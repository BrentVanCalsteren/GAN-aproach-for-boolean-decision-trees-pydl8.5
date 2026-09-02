from __future__ import annotations
from typing import List, Optional, Sequence, Tuple, Union
import CONFIG
import numpy as np


ClosureInput = Union[str, Tuple[bool, bool]]

class Intervals:
    def __init__(self, feat_id, min_v=None,max_v=None,):
        self.feat_id = feat_id
        self.min_val = min_v
        if min_v is None:
            self.min_val = float(CONFIG.GLOBAL_CHUNK_INFO.processed_feat_min[feat_id])
        self.max_val = max_v
        if max_v is None:
            self.max_val = float(CONFIG.GLOBAL_CHUNK_INFO.processed_feat_max[feat_id])

        self.interval_list: List[Interval] = []
        self.generate_start_interval()

    def generate_start_interval(self):
        self.interval_list = [Interval(self.min_val, self.max_val, "closed")]

    @property
    def is_empty(self) -> bool:
        return len(self.interval_list) == 0

    def add_interval(self, startpoint: float, endpoint: float, closureType: str = "closed"):
        add_constraint_union(self.interval_list,[Interval(startpoint, endpoint, closureType)])

    def get_domain_intervals(self) -> List[Interval]:
        #returns the interval object copy list
        return [iv.copy() for iv in self.interval_list]

    def get_domains(self) -> List[List[float]]:
        #makes intervals closed witout epsilon
        return [[iv.startpoint, iv.endpoint] for iv in self.interval_list]

    def get_epsilon_domains(self, delta: float = 1e-6) -> List[List[float]]:
        #makes all intervals closed
        out = []
        for inter in self.interval_list:
            ep_inter = inter.epsilon_closed(delta)
            if ep_inter is not None: out.append([ep_inter.startpoint, ep_inter.endpoint])
        return out

def add_constraint_union(interval_list, constraint_intervals: Sequence[Interval]):
    #will intersect a list of union intervals with already present intervals
    constraints = [iv for iv in constraint_intervals if iv.isvalid_interval]
    #no valid constraints
    if not constraints:
        return []

    new_list: List[Interval] = []
    for current in interval_list:
        for constraint in constraints:
            inter = current.intersect(constraint)
            if inter is not None:
                new_list.append(inter)
    return merge_intervals(new_list)

class Interval:
    #interval types: 'closed' true, true 'open' false, false 'half-open' f, t  'half-closed' t, f
    def __init__(self,startpoint: float,endpoint: float,closure: ClosureInput = "closed"):
        self.startpoint = float(startpoint)
        self.endpoint = float(endpoint)

        if isinstance(closure, str): self.left_closed, self.right_closed = self._parse_closure(closure)
        else: self.left_closed, self.right_closed = bool(closure[0]), bool(closure[1])
        self.isvalid_interval = self._validate()

    def __str__(self) -> str:
        return f"({self.startpoint},{self.endpoint})"

    def __repr__(self) -> str:
        if not self.isvalid_interval:
            return "EmptyInterval"

        left = "_[" if self.left_closed else "_("
        right = "]_" if self.right_closed else ")_"
        return f"{left}{self.startpoint}, {self.endpoint}{right}"

    def contains_value(self, value: float):
        if value < self.startpoint or value > self.endpoint: return False
        return True

    def return_interval_as_list(self):
        return np.array([self.startpoint, self.endpoint])

    @staticmethod
    def _parse_closure(closureType: str) -> Tuple[bool, bool]:
        c = str(closureType).lower().strip()
        if c == "closed": return True, True
        if c == "open": return False, False
        if c == "half-open": return False, True
        if c == "half-closed": return True, False
        return True, True

    def _validate(self) -> bool:
        if np.isnan(self.startpoint) or np.isnan(self.endpoint): return False
        if self.startpoint > self.endpoint: return False
        if self.startpoint == self.endpoint: return self.left_closed and self.right_closed
        return True

    def copy(self) -> "Interval":
        return Interval(self.startpoint,self.endpoint,(self.left_closed, self.right_closed))

    def make_closed(self, closureType: str) -> bool:
        self.left_closed, self.right_closed = self._parse_closure(closureType)
        self.isvalid_interval = self._validate()
        return self.isvalid_interval

    def intersect(self, other: "Interval") -> Optional["Interval"]:
        if not (self.isvalid_interval and other.isvalid_interval): return None

        if self.startpoint > other.startpoint:
            start = self.startpoint
            left_closed = self.left_closed
        elif self.startpoint < other.startpoint:
            start = other.startpoint
            left_closed = other.left_closed
        else:
            start = self.startpoint
            left_closed = self.left_closed and other.left_closed

        if self.endpoint < other.endpoint:
            end = self.endpoint
            right_closed = self.right_closed
        elif self.endpoint > other.endpoint:
            end = other.endpoint
            right_closed = other.right_closed
        else:
            end = self.endpoint
            right_closed = self.right_closed and other.right_closed

        out = Interval(start, end, (left_closed, right_closed))
        return out if out.isvalid_interval else None

    def epsilon_closed(self, delta: float = 1e-6) -> Optional["Interval"]:
        #make all intervals closed by introducing a small epsilon
        if not self.isvalid_interval: return None

        s = self.startpoint + (delta if not self.left_closed else 0.0)
        e = self.endpoint - (delta if not self.right_closed else 0.0)

        if s > e: return None
        return Interval(s, e, "closed")


def merge_intervals(intervals: Sequence[Interval]) -> List[Interval]:
    #merge intervals
    valid = [iv.copy() for iv in intervals if iv.isvalid_interval]
    if not valid: return []
    #sort left closed bound
    valid.sort(key=lambda iv: (iv.startpoint, not iv.left_closed, iv.endpoint))
    merged: List[Interval] = [valid[0].copy()]

    for iv in valid[1:]:
        last = merged[-1]
        #gap.
        if last.endpoint < iv.startpoint:
            merged.append(iv.copy())
            continue
        #touching but on both open sides.
        if last.endpoint == iv.startpoint and not (last.right_closed or iv.left_closed):
            merged.append(iv.copy())
            continue
        #overlap
        if iv.endpoint > last.endpoint:
            new_end = iv.endpoint
            new_right_closed = iv.right_closed
        elif iv.endpoint < last.endpoint:
            new_end = last.endpoint
            new_right_closed = last.right_closed
        else:
            new_end = last.endpoint
            new_right_closed = last.right_closed or iv.right_closed

        new_left_closed = last.left_closed
        if last.startpoint == iv.startpoint:
            new_left_closed = last.left_closed or iv.left_closed

        merged[-1] = Interval(last.startpoint,new_end,(new_left_closed, new_right_closed))
    return merged