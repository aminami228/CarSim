#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import math
#import csv
import matplotlib.pyplot as plt
import time

#fig = plt.figure()
#ax = fig.add_subplot

x_points = [0, 1, 2, 3, 4, 5]
y_points = [0, 1, 2, 3, 4, 5]
p, = plt.plot([], [],'bo')
plt.show()

for i in range(5):
    x = x_points[i]
    y = y_points[i]
    plt.plot(x, y, 'bo')
    print(x)
#    p.set_xdata(np.append(p.get_xdata(), x))
#    p.set_xdata(np.append(p.get_ydata(), y))
#    plt.draw()
    plt.draw()
    time.sleep(1)
    #plt.pause(0.5)
    #plt.draw()

