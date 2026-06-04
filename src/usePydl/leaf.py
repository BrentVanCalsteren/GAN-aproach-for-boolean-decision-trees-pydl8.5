import numpy as np
from src.samplers.load_samplers import get_sampler_class

#default leaf value -> return the samplers
def just_return_sample_ids(total_samples):
    def value(tids):
        sample_list = list(tids)
        return {"sample_ids": sample_list, "rel_prob": len(sample_list) / total_samples}
    return value


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


