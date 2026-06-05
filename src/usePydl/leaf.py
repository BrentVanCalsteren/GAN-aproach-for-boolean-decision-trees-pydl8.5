import numpy as np
from src.samplers.load_samplers import get_sampler_class

#default leaf value -> return the samplers
class ReturnIDSandPROB:
    def __init__(self, size):
        self.total_size = size

    def __call__(self, tids):
        sample_list = list(tids)
        return {"sample_ids": sample_list,
                "rel_prob": len(sample_list) / self.total_size}


#other leaf val functions: NOT USED RIGHT NOW
def empty_val():
    def value(_):
        return 1
    return value
#-------------------------------------------------

#helpers for looking at leaf data
def get_leafs(tree):
    leaves = []
    def recurse(node):
        if "value" in node:
            leaves.append(node)
        else:
            recurse(node["left"])
            recurse(node["right"])
    recurse(tree)
    return leaves


