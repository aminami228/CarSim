#!/usr/bin/env python
import matplotlib.pyplot as plt
from matplotlib import text
import numpy as np
from random import randint
import random
import math


class InterSim(object):
    Tau = 1. / 30
    Speed_limit = 12        # m/s
    Scenary = randint(0, 2)
    Inter_Ori = 0.
    Stop_Line = - 5. - random.random()
    Pass_Point = 20.
    Inter_Low = - 4.
    Inter_Up = 4.
    Inter_Left = - 4.
    Inter_Right = 4.
    Vehicle_NO = 3
    Cft_Accel = 3.     # m/s**2
    Visual = False

    def __init__(self):
        self.sim = None
        self.av_pos = dict()
        self.av_pos['y'] = - random.random() * 50. - 100.
        self.av_pos['x'] = 2.
        self.av_pos['vx'] = 0.
        self.av_pos['vy'] = random.random() * self.Speed_limit + 5
        self.hv_poses = []
        for i in range(self.Vehicle_NO):
            hv_pos = dict()
            hv_pos['y'] = self.av_pos['y'] + random.random() * 100. + 20.
            hv_pos['x'] = 2.
            hv_pos['vx'] = 0.
            hv_pos['vy'] = self.Speed_limit - random.random()
            self.hv_poses.append(hv_pos)
        self.target_dis = None
        self.target_v = None
        self.state = None
        self.state_dim = None

    def draw_scenary(self, av, hvs):
        if self.Visual:
            plt.figure(1)
            plt.plot(0, self.Inter_Ori, 'g.', markersize=10)
            plt.plot(list(range(4)), list([self.Stop_Line] * 4), 'g')
            plt.plot(list(xrange(-20, 20)), list([self.Inter_Low] * 40), 'r')
            plt.plot(list(xrange(-20, 20)), list([self.Inter_Up] * 40), 'r')
            plt.plot(list([self.Inter_Left] * 400), list(xrange(-200, 200)), 'r')
            plt.plot(list([self.Inter_Right] * 400), list(xrange(-200, 200)), 'r')
            plt.plot(self.Inter_Right / 2., self.Pass_Point, 'g.', markersize=10)
            plt.plot(av['x'], av['y'], 'r.', markersize=15)
            plt.text(av['x'], av['y'], str(av['vy']))
            for hv in hvs:
                plt.plot(hv['x'], hv['y'], 'c.', markersize=15)
                plt.text(hv['x'], hv['y'], str(hv['vy']))
            plt.show()
            plt.pause(0.2)
            plt.clf()

    def get_state(self, a=0):
        self.update_vehicle(a)
        front_dis = [hv_pos['y'] - self.av_pos['y'] for hv_pos in self.hv_poses]
        sl_dis = self.Stop_Line - self.av_pos['y']
        dis_pool = [sl_dis] + front_dis
        self.target_dis = min(dis_pool)
        self.target_v = 0. if dis_pool.index(min(dis_pool)) == 0 else self.hv_poses['vy']
        self.state = np.array([self.av_pos['vy'], self.target_v, self.target_dis], ndim=2)
        self.state_dim = np.shape(self.state)[1]
        print 'Accel: ', a, ', V_av = ', self.av_pos['vy'], ', Distance Pool: ', dis_pool

    def update_vehicle(self, a=0):
        for hv_pos in self.hv_poses:
            hv_a = - 0.5 * (hv_pos['vy'] ** 2) / (self.Stop_Line - hv_pos['y']) if hv_pos['y'] < self.Stop_Line - 1 \
                else self.Cft_Accel
            hv_pos['vy'] += hv_a * self.Tau
            hv_pos['vy'] = min(max(0.1, hv_pos['vy']), self.Speed_limit)
            hv_pos['y'] += hv_pos['vy'] * self.Tau + 0.5 * hv_a * (self.Tau ** 2)
        old_av_vel = self.av_pos['vy']
        self.av_pos['vy'] += a * self.Tau
        self.av_pos['y'] += old_av_vel * self.Tau + 0.5 * a * (self.Tau ** 2)
        if self.Visual:
            self.draw_scenary(self.av_pos, self.hv_poses)


if __name__ == '__main__':
    sim = InterSim()
    plt.ion()
    while True:
        sim.update_vehicle()
        sim.get_state()
