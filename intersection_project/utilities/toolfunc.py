#!/usr/bin/env python
import numpy as np
from math import exp

__author__ = 'qzq'


class ToolFunc(object):

    @staticmethod
    def ou(x, mu, theta, sigma):
        return theta * (mu - x) + sigma * np.random.randn(1)

    @staticmethod
    def sigmoid(x, b, c=0):
        return 1 / (1 + exp(- b * x + c))
