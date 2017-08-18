import matplotlib.pyplot as plt
import numpy as np
from random import randint, random
from utilities.toolfunc import ToolFunc
import logging
import utilities.log_color

__author__ = 'qzq'


class InterSim(object):
    Tau = 1. / 30
    Speed_limit = 12        # m/s
    Scenary = randint(0, 2)
    Inter_Ori = 0.
    Stop_Line = - 5. - random()
    Pass_Point = 20.
    Inter_Low = - 4.
    Inter_Up = 4.
    Inter_Left = - 4.
    Inter_Right = 4.
    Vehicle_NO = 1
    Lane_Left = 0.
    Lane_Right = 4.
    Cft_Accel = 3.     # m/s**2
    Full_Accel = 5.
    Full_Brake = 5.

    tools = ToolFunc()

    def __init__(self, visual=False):
        self.Visual = visual
        self.av_pos = dict()
        self.av_pos['y'] = - random() * 50. - 100.
        self.Start_Pos = self.av_pos['y']
        self.av_pos['x'] = 2.
        self.av_pos['vx'] = 0.
        self.av_pos['vy'] = self.Speed_limit + random() * 2. - 1.
        self.av_pos['heading'] = 0
        self.av_pos['accel'] = 0
        self.av_pos['steer'] = 0
        self.av_size = [4, 2]
        self.hv_poses = []
        for i in range(self.Vehicle_NO):
            hv_pos = dict()
            hv_pos['y'] = self.av_pos['y'] + random() * 50. + 20.
            hv_pos['x'] = 2.
            hv_pos['vx'] = 0.
            hv_pos['vy'] = self.Speed_limit - random()
            self.hv_poses.append(hv_pos)
        self.target_dis = None
        self.target_v = None
        self.state = None
        self.state_dim = None

        self.state_av = []
        self.state_fv = []
        self.state_road = []

    def draw_scenary(self, av, hvs, r):
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
            plt.text(av['x'], av['y'], 'a: ' + str(av['accel']) + ', v: ' + str(av['vy']) + ', reward: ' + str(r) +
                     '\n f_dis: ' + str(self.state_fv[1]) + ', sl_dis: ' + str(self.Stop_Line - av['y']))
            for hv in hvs:
                plt.plot(hv['x'], hv['y'], 'c.', markersize=15)
                plt.text(hv['x'], hv['y'], str(hv['vy']))
            plt.show()
            plt.pause(0.1)
            plt.clf()

    def get_state(self):
        self.state_av = [self.av_pos['vy'], self.av_pos['heading'], self.av_pos['accel'], self.av_pos['steer']]
        fv_dis_list = [hv_pos['y'] - self.av_pos['y'] for hv_pos in self.hv_poses]
        fv_index = np.argmin(fv_dis_list)
        fv_pos = self.hv_poses[fv_index]
        self.state_fv = [fv_pos['vy'], fv_pos['y'] - self.av_pos['y']]
        sl_dis = self.Stop_Line - self.av_pos['y']
        ll = self.av_pos['x'] - self.av_size[1] / 2 - self.Lane_Left
        lr = self.Lane_Right - (self.av_pos['x'] + self.av_size[1] / 2)
        start_pos = self.Start_Pos
        self.state_road = [sl_dis, ll, lr, start_pos]
        self.state = np.array(self.state_av + self.state_fv + self.state_road, ndmin=2)
        self.state_dim = self.state.shape[1]
        return self.state

    def update_vehicle(self, r, a=0., b=0., st=0.):
        accel = self.Full_Accel * a - self.Full_Brake * b
        for hv_pos in self.hv_poses:
            hv_a = - 0.5 * (hv_pos['vy'] ** 2) / (self.Stop_Line - hv_pos['y']) if hv_pos['y'] < self.Stop_Line - 1 \
                else self.Cft_Accel
            hv_pos['vy'] += hv_a * self.Tau
            hv_pos['vy'] = min(max(0.1, hv_pos['vy']), self.Speed_limit)
            hv_pos['y'] += hv_pos['vy'] * self.Tau + 0.5 * hv_a * (self.Tau ** 2)
        old_av_vel = self.av_pos['vy']
        self.av_pos['vy'] += accel * self.Tau
        self.av_pos['vy'] = max(0.0, self.av_pos['vy'])
        self.av_pos['y'] += old_av_vel * self.Tau + 0.5 * accel * (self.Tau ** 2)
        self.av_pos['heading'] += st
        self.av_pos['accel'] = accel
        self.av_pos['steer'] = st
        if self.Visual:
            self.draw_scenary(self.av_pos, self.hv_poses, r)


if __name__ == '__main__':
    sim = InterSim()
    plt.ion()
    while sim.av_pos['y'] <= sim.Pass_Point:
        sim.get_state()
        # sim.update_vehicle()
