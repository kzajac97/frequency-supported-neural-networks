from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
from scipy import stats


class AbstractNoiseGenerator(ABC):
    """
    Interface for NoiseGenerator used by NoisyTorchTimeSeriesCsvDataLoader

    The object is callable and takes sequence of values as input and returns noised sequence as output to allow using
    any type of noise over the dataset, it can be added, multiplied or replace the sequence
    """

    @abstractmethod
    def __call__(self, values: Sequence) -> Sequence:
        """Return noised values sequence"""
        ...


class AdditiveGaussianNoiseGenerator(AbstractNoiseGenerator):
    """
    Implements additive gaussian noise generation adding random noise with given mean and standard deviation
    """

    def __init__(self, mean: float, stddev: float):
        self.mean = mean
        self.stddev = stddev

    def __call__(self, values: np.array) -> np.array:
        """Adds Gaussian noise to input values"""
        noise = stats.norm.rvs(loc=self.mean, scale=self.stddev, size=values.size).reshape(values.shape)
        return values + noise


class AdditiveDistributionNoiseGenerator(AbstractNoiseGenerator):
    """
    Implements additive gaussian noise generation adding random noise with given mean and standard deviation
    """

    def __init__(self, distribution: stats.rv_continuous, mean: float, stddev: float):
        """
        :param distribution: instance of scipy rv_continuous used to generate random values
        :param mean: mean of the distribution
        :param stddev: standard deviation of the nois distribution
        """
        self.distribution = distribution
        self.mean = mean
        self.stddev = stddev

    def __call__(self, values: np.array) -> np.array:
        """Adds Gaussian noise to input values"""
        noise = self.distribution.rvs(loc=self.mean, scale=self.stddev, size=values.size).reshape(values.shape)
        return values + noise
