from bernoulli import *
from uniform import *
from multinomial import *
from single_gaussian import *

def create_sampler(sample_type: str):
    if sample_type == 'multinomial':
        return Multinomial_sampler()
    elif sample_type.lower() == 'bernoulli':
        return  Bernoulli_sampler()
    elif sample_type.lower() == 'uniform':
        return Uniform_sampler()
    elif sample_type.lower() == 'single_gaussian':
        return SingleGaussian1D_sampler()
    else:
        print('Unknown sampler type')