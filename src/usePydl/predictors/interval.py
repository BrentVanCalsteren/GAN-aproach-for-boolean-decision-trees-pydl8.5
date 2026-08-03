from typing import List

class Intervals:
    def __init__(self,feat_id, chunkinfo):
        self.min_val = chunkinfo.processed_feat_min[feat_id]
        self.max_val = chunkinfo.processed_feat_max[feat_id]
        self.interval_list: List[Interval] = []
        self.generate_start_interval()

    def generate_start_interval(self):
        self.interval_list.append(Interval(self.min_val, self.max_val, 'closed'))

    def add_interval(self, startpoint, endpoint, closureType):
        interval = Interval(startpoint, endpoint, closureType)
        if interval.isvalid_interval:
            self.interval_list.append(interval)

    def get_complete_domain(self):
        interval = [self.min_val, self.max_val]
        for inter in self.interval_list:
            if not inter.isvalid_interval:
                continue
            start = max(interval[0], inter.startpoint)
            end = min(interval[1], inter.endpoint)
            if start <= end:
                interval = [start, end]
        if interval[0] >= interval[1]:
            interval[0] = max(self.min_val, interval[0] - 1e-6)
            interval[1] = min(self.max_val, interval[1] + 1e-6)
        return interval


class Interval:
    def __init__(self, startpoint, endpoint, closure: str):
        self.startpoint = startpoint
        self.endpoint = endpoint
        self.isvalid_interval = self.make_closed(closure)

    def make_closed(self, closureType):
        delta = 1e-6
        if closureType == 'closed':
            return True
        elif closureType == 'open':
            if self.startpoint == self.endpoint:
                return False
            self.startpoint+= delta
            self.endpoint -= delta
        elif closureType == 'half-open':
            if self.startpoint == self.endpoint:
                return False
            self.startpoint+= delta
        elif closureType == 'half-closed':
            if self.startpoint == self.endpoint:
                return False
            self.endpoint -= delta
        return True

    def get_boundry_points(self):
        if self.startpoint == self.endpoint:
            return [self.endpoint]
        return [self.startpoint, self.endpoint]