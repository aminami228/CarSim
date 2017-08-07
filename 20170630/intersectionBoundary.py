from collections import namedtuple
import matplotlib.pyplot as plt
from random import *
import numpy as np

Point = namedtuple('Points','x y')


class Intersection(object):
    _points = []
    _hull_points = []

    def __init__(self):
        self._points = []
        self._hull_points = []

    def add(self, point):
        self._points.append(point)

    def _get_orientation(self, origin, p1, p2):
        '''
        Returns the orientation of the Point p1 with regards to Point p2 using origin.
        Negative if p1 is clockwise of p2.
        :param p1:
        :param p2:
        :return: integer
        '''
        difference = (
            ((p2.x - origin.x) * (p1.y - origin.y))
            - ((p1.x - origin.x) * (p2.y - origin.y))
        )

        return difference

    def compute_hull(self):
        '''
        Computes the points that make up the convex hull.
        :return:
        '''
        points = self._points

        # get leftmost point
        start = points[0]
        min_x = start.x
        for p in points[1:]:
            if p.x < min_x:
                min_x = p.x
                start = p

        point = start
        self._hull_points.append(start)

        far_point = None
        while far_point is not start:

            # get the first point (initial max) to use to compare with others
            p1 = None
            for p in points:
                if p is point:
                    continue
                else:
                    p1 = p
                    break

            far_point = p1

            for p2 in points:
                # ensure we aren't comparing to self or pivot point
                if p2 is point or p2 is p1:
                    continue
                else:
                    direction = self._get_orientation(point, far_point, p2)
                    if direction > 0:
                        far_point = p2

            self._hull_points.append(far_point)
            point = far_point

    def get_hull_points(self):
        # if self._points and not self._hull_points:
        self.compute_hull()

        return self._hull_points

    def display(self):
        # all points
        x = [p.x for p in self._points]
        y = [p.y for p in self._points]
        plt.plot(x, y, 'b.', linestyle='None')
        plt.draw()

        # hull points
        # hx = [p.x for p in self._hull_points]
        # hy = [p.y for p in self._hull_points]
        hx = [p.x for p in self._points]
        hy = [p.y for p in self._points]
        # plt.plot(hx, hy, 'b')

        plt.title('Intersection')
        plt.axis('equal')
        plt.draw()
        return hx, hy


def drawIntersection():
    ch1 = Intersection()
    for _ in range(500):
        ch1.add(Point(uniform(-64, -4), -4 + uniform(-0.5, 0.5)))
        ch1.add(Point(-4 + uniform(-0.5, 0.5), uniform(-44, -4)))
    # print("hull1:", ch1.get_hull_points())
    ch1.get_hull_points()
    hx1, hy1 = ch1.display()
    hull1 = np.vstack(([hx1],[hy1])).T

    ch2 = Intersection()
    for _ in range(500):
        ch2.add(Point(uniform(-64, -4), 4 + uniform(-0.5, 0.5)))
        ch2.add(Point(-4 + uniform(-0.5, 0.5), uniform(4, 44)))
    # print("hull2:", ch2.get_hull_points())
    ch2.get_hull_points()
    hx2, hy2 = ch2.display()
    hull2 = np.vstack(([hx2], [hy2])).T

    ch3 = Intersection()
    for _ in range(500):
        ch3.add(Point(4 + uniform(-0.5, 0.5), uniform(4, 44)))
        ch3.add(Point(uniform(4, 44), 4 + uniform(-0.5, 0.5)))
        ch3.add(Point(44 + uniform(-0.5, 0.5), uniform(4, 44)))
    # print("hull3:", ch3.get_hull_points())
    ch3.get_hull_points()
    hx3, hy3 = ch3.display()
    hull3 = np.vstack(([hx3], [hy3])).T

    ch4 = Intersection()
    for _ in range(500):
        ch4.add(Point(4 + uniform(-0.5, 0.5), uniform(-44, -4)))
        ch4.add(Point(uniform(4, 44), -4 + uniform(-0.5, 0.5)))
        ch4.add(Point(44 + uniform(-0.5, 0.5), uniform(-44, -4)))
    # print("hull4:", ch4.get_hull_points())
    ch4.get_hull_points()
    hx4, hy4 = ch4.display()
    hull4 = np.vstack(([hx4], [hy4])).T

    ch5 = Intersection()
    for _ in range(500):
        ch5.add(Point(52 + uniform(-0.5, 0.5), uniform(-44, -4)))
        ch5.add(Point(uniform(52, 92), -4 + uniform(-0.5, 0.5)))
        ch5.add(Point(92 + uniform(-0.5, 0.5), uniform(-44, -4)))
    # print("hull5:", ch5.get_hull_points())
    ch5.get_hull_points()
    hx5, hy5 = ch5.display()
    hull5 = np.vstack(([hx5], [hy5])).T

    ch6 = Intersection()
    for _ in range(500):
        ch6.add(Point(52 + uniform(-0.5, 0.5), uniform(4, 44)))
        ch6.add(Point(uniform(52, 92), 4 + uniform(-0.5, 0.5)))
        ch6.add(Point(92 + uniform(-0.5, 0.5), uniform(4, 44)))
    # print("hull6:", ch6.get_hull_points())
    ch6.get_hull_points()
    hx6, hy6 = ch6.display()
    hull6 = np.vstack(([hx6], [hy6])).T

    ch7 = Intersection()
    for _ in range(500):
        ch7.add(Point(100 + uniform(-0.5, 0.5), uniform(4, 44)))
        ch7.add(Point(uniform(100, 160), 4 + uniform(-0.5, 0.5)))
    # print("hull7:", ch7.get_hull_points())
    ch7.get_hull_points()
    hx7, hy7 = ch7.display()
    hull7 = np.vstack(([hx7], [hy7])).T

    ch8 = Intersection()
    for _ in range(500):
        ch8.add(Point(100 + uniform(-0.5, 0.5), uniform(-44, -4)))
        ch8.add(Point(uniform(100, 160), -4 + uniform(-0.5, 0.5)))
    # print("hull4:", ch4.get_hull_points())
    ch8.get_hull_points()
    hx8, hy8 = ch8.display()
    hull8 = np.vstack(([hx8], [hy8])).T


    plt.plot(range(-64,160), np.zeros(224), 'r.')

    plt.show()
    return hull1, hull2, hull3, hull4, hull5, hull6, hull7, hull8

if __name__ == '__main__':
   hull = drawIntersection()