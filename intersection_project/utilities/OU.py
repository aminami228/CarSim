#!/usr/bin/env python
import numpy as np


class OU(object):

    @staticmethod
    def ou(x, mu, theta, sigma):
        return theta * (mu - x) + sigma * np.random.randn(1)
