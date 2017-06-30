from collections import namedtuple
import matplotlib.pyplot as plt
from random import *


class Vehicle(object):
    _points = []
    _hull_points = []

    def __init__(self):
        self._points = []
        self._hull_points = []