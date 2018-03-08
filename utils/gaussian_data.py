import numpy as np

def sample_z(num):
    return np.random.uniform(-1.0, 1.0, size=(num, 100))


def multivariate_normal_sampler(mean, cov):
    return lambda num : np.random.multivariate_normal(mean, cov, size=(num,100))

