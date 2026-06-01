from src.samplers.uniform import *
from src.samplers.multinomial import *
from src.samplers.single_gaussian import *
from src.samplers.multivariate_gaussian import *

def get_sampler_class(sample_type: str):
    if sample_type == 'multinomial':
        return MultinomialSampler
    elif sample_type == 'uniform':
        return UniformSampler
    elif sample_type == 'single_gaussian':
        return SingleGaussian1DSampler
    elif sample_type == 'multi_gaussian':
        return MultivariateGaussianSampler
    else:
        print('Unknown sampler type:', sample_type)
        return None

def create_sampler(sample_type: str):
    cls = get_sampler_class(sample_type)
    if cls:
        return cls()
    return None