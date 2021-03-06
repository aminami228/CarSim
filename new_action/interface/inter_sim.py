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
    FV_NO = 3
    LV_NO = 4
    RV_NO = 5
    Lane_Left = 0.
    Lane_Right = 4.
    Cft_Accel = 3.     # m/s**2

    tools = ToolFunc()

    def __init__(self, visual=False):
        self.Visual = visual
        self.av_pos = dict()
        self.av_pos['y'] = - random() * 50. - 100.
        self.Start_Pos = self.av_pos['y']
        self.av_pos['x'] = 2.
        self.av_pos['vx'] = 0.
        self.av_pos['vy'] = self.Speed_limit - random() * 5.
        self.ini_speed = self.av_pos['vy']
        self.av_pos['heading'] = 0
        self.av_pos['accel'] = 0
        self.av_pos['steer'] = 0
        self.av_size = [4, 2]
        self.fv_poses = []
        for i in range(self.FV_NO):
            fv_pos = dict()
            fv_pos['y'] = self.av_pos['y'] + random() * 30. + 20.
            fv_pos['x'] = 2.
            fv_pos['vx'] = 0.
            fv_pos['vy'] = self.Speed_limit - random()
            self.fv_poses.append(fv_pos)
        self.target_dis = None
        self.target_v = None
        self.state = None
        self.state_dim = None

        self.state_av = []
        self.state_fv = []
        self.state_road = []

    def draw_scenary(self, av, fvs, r):
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
            for fv in fvs:
                plt.plot(fv['x'], fv['y'], 'c.', markersize=15)
                plt.text(fv['x'], fv['y'], str(fv['vy']))
            plt.show()
            plt.pause(0.1)
            plt.clf()

    def get_state(self):
        self.state_av = [self.av_pos['vy'], self.av_pos['heading'], self.av_pos['accel'], self.av_pos['steer']]
        fv_dis_list = [fv_pos['y'] - self.av_pos['y'] for fv_pos in self.fv_poses]
        fv_index = np.argmin(fv_dis_list)
        fv_pos = self.fv_poses[fv_index]
        self.state_fv = [fv_pos['vy'], fv_pos['y'] - self.av_pos['y'] - 4.]
        sl_dis = self.Stop_Line - self.av_pos['y']
        ll = self.av_pos['x'] - self.av_size[1] / 2 - self.Lane_Left
        lr = self.Lane_Right - (self.av_pos['x'] + self.av_size[1] / 2)
        start_pos = self.Start_Pos
        self.state_road = [sl_dis, ll, lr, start_pos]
        self.state = np.array(self.state_av + self.state_fv + self.state_road, ndmin=2)
        self.state_dim = self.state.shape[1]
        return self.state

    def update_vehicle(self, r, a=0, st=0):
        accel = self.Cft_Accel * a
        for fv_pos in self.fv_poses:
            fv_a = - 0.5 * (fv_pos['vy'] ** 2) / (self.Stop_Line - fv_pos['y']) if fv_pos['y'] < self.Stop_Line - 1 \
                else self.Cft_Accel
            fv_pos['vy'] += fv_a * self.Tau
            fv_pos['vy'] = min(max(0.1, fv_pos['vy']), self.Speed_limit)
            fv_pos['y'] += fv_pos['vy'] * self.Tau + 0.5 * fv_a * (self.Tau ** 2)
        old_av_vel = self.av_pos['vy']
        self.av_pos['vy'] += accel * self.Tau
        self.av_pos['vy'] = max(0.0, self.av_pos['vy'])
        self.av_pos['y'] += old_av_vel * self.Tau + 0.5 * accel * (self.Tau ** 2)
        self.av_pos['heading'] += st
        self.av_pos['accel'] = accel
        self.av_pos['steer'] = st
        if self.Visual:
            self.draw_scenary(self.av_pos, self.fv_poses, r)


if __name__ == '__main__':
    sim = InterSim()
    plt.ion()
    while sim.av_pos['y'] <= sim.Pass_Point:
        sim.get_state()
        # sim.update_vehicle()

