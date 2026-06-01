from src.samplers.uniform import *
from src.samplers.multinomial import *
from src.samplers.single_gaussian import *

def create_sampler(sample_type: str):
    if sample_type == 'multinomial':
        return MultinomialSampler()
    elif sample_type == 'uniform':
        return UniformSampler()
    elif sample_type == 'single_gaussian':
        return SingleGaussian1DSampler()
    else:
        print('Unknown sampler type')