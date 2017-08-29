import matplotlib.pyplot as plt
import numpy as np
from random import randint, random, sample, uniform
from utilities.toolfunc import ToolFunc
import time
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
    FV_NO = 2
    LV_NO = 5
    RV_NO = 8
    Lane_Left = 0.
    Lane_Right = 4.
    Cft_Accel = 3.     # m/s**2

    history_len = 50
    tools = ToolFunc()

    def __init__(self, visual=False):
        self.Visual = visual
        self.av_pos = dict()
        self.av_pos['y'] = self.Stop_Line - 5. + random()
        self.Start_Pos = self.av_pos['y']
        self.av_pos['x'] = 2.
        self.av_pos['vx'] = 0.
        self.av_pos['vy'] = 5. - random()
        self.av_pos['heading'] = 0
        self.av_pos['accel'] = 0
        self.av_pos['steer'] = 0
        self.av_size = [4, 2]

        # self.fv_poses = []
        # fv_locs = 15. * np.array(sample(xrange(1, 4), self.FV_NO)) + random()
        # for y in fv_locs:
        #     fv_pos = dict()
        #     fv_pos['y'] = self.av_pos['y'] + y
        #     fv_pos['x'] = 2.
        #     fv_pos['v'] = self.Speed_limit - random()
        #     fv_pos['a'] = 0.
        #     self.fv_poses.append(fv_pos)

        self.lv_poses = []
        lv_locs = 15. * np.array(sample(xrange(-6, 4), self.LV_NO)) + random()
        for x in lv_locs:
            lv_pos = dict()
            lv_pos['y'] = self.Inter_Low + 2.
            lv_pos['x'] = x
            lv_pos['v'] = self.Speed_limit - 5. * random()
            lv_pos['a'] = uniform(-1., 1.)
            lv_pos['dir'] = 'R'
            self.lv_poses.append(lv_pos)
        self.rv_poses = []
        rv_locs = 15. * np.array(sample(xrange(-3, 7), self.RV_NO)) + random()
        for x in rv_locs:
            rv_pos = dict()
            rv_pos['y'] = self.Inter_Up - 2.
            rv_pos['x'] = x
            rv_pos['v'] = self.Speed_limit - 5. * random()
            rv_pos['a'] = uniform(-1., 1.)
            rv_pos['dir'] = 'L'
            self.rv_poses.append(rv_pos)
        self.hv_poses = self.lv_poses + self.rv_poses
        self.state_hv_mem = []
        self.target_dis = None
        self.target_v = None
        self.state = None
        self.state_dim = None

        self.state_av = []
        self.state_fv = []
        self.state_hv = []
        self.state_road = []

    def draw_scenary(self, av, hvs, r):
        if self.Visual:
            plt.figure(1)
            plt.plot(0, self.Inter_Ori, 'g.', markersize=10)
            plt.plot(list(range(4)), list([self.Stop_Line] * 4), 'g')
            plt.plot(list(xrange(-150, 150)), list([self.Inter_Low] * 300), 'r')
            plt.plot(list(xrange(-150, 150)), list([self.Inter_Up] * 300), 'r')
            plt.plot(list([self.Inter_Left] * 400), list(xrange(-200, 200)), 'r')
            plt.plot(list([self.Inter_Right] * 400), list(xrange(-200, 200)), 'r')
            plt.plot(self.Inter_Right / 2., self.Pass_Point, 'g.', markersize=10)
            plt.plot([av['x'], av['x']], [av['y'] - 2., av['y'] + 2.], 'r.-', markersize=10)
            plt.text(av['x'], av['y'], 'a: ' + str(av['accel']) + ', v: ' + str(av['vy']) + ', reward: ' + str(r) +
                     '\n f_dis: ' + str(self.state_fv[1]) + ', sl_dis: ' + str(self.Stop_Line - av['y']))
            for v in hvs:
                plt.plot([v['x'] - 2., v['x'] + 2.], [v['y'], v['y']], 'c.-', markersize=10)
                # plt.text(v['x'], v['y'], str(v['v']))
            plt.axis([-100, 100, -20, 50])
            plt.show()
            plt.pause(0.0001)
            plt.clf()

    def get_state(self, his_):
        self.state_av = [self.av_pos['vy'], self.av_pos['heading'], self.av_pos['accel'], self.av_pos['steer']]

        # fv_dis_list = [fv_pos['y'] - self.av_pos['y'] for fv_pos in self.fv_poses]
        # fv_index = np.argmin(fv_dis_list)
        # fv_pos = self.fv_poses[fv_index]
        # self.state_fv = [fv_pos['v'], fv_pos['y'] - self.av_pos['y'] - 4.]

        sl_dis = self.Stop_Line - self.av_pos['y']
        int_center_dis = self.Inter_Ori - self.av_pos['y']
        int_upper_dis = self.Inter_Up - self.av_pos['y']
        pass_dis = self.Pass_Point - self.av_pos['y']
        ll = self.av_pos['x'] - self.av_size[1] / 2 - self.Lane_Left
        lr = self.Lane_Right - (self.av_pos['x'] + self.av_size[1] / 2)
        start_pos = self.Start_Pos
        self.state_road = [sl_dis, int_center_dis, int_upper_dis, pass_dis, ll, lr, start_pos]

        lv_dis_list = [self.av_pos['x'] - lv_pos['x'] - 3. for lv_pos in self.lv_poses]
        l_min_dis = np.inf
        lv_pos = self.lv_poses[0]
        for i, dis in enumerate(lv_dis_list):
            if dis <= l_min_dis and (dis >= 0.):
                lv_pos = self.lv_poses[i]
                l_min_dis = dis
        if np.isinf(l_min_dis):
            l_min_dis = -1.
            lv_pos['v'] = -1.
        rv_dis_list = [rv_pos['x'] - self.av_pos['x'] - 3. for rv_pos in self.rv_poses]
        r_min_dis = np.inf
        rv_pos = self.rv_poses[0]
        for i, dis in enumerate(rv_dis_list):
            if dis <= r_min_dis and (dis >= 0.):
                rv_pos = self.rv_poses[i]
                r_min_dis = dis
        if np.isinf(r_min_dis):
            r_min_dis = -1.
        if not his_:
            his_ = [[lv_pos['v'], l_min_dis, rv_pos['v'], r_min_dis]]
        else:
            his_.append([lv_pos['v'], l_min_dis, rv_pos['v'], r_min_dis])
        if len(his_) > self.history_len:
            his_ = his_[-self.history_len:]
        self.state_hv_mem = np.array(his_, ndmin=2)
        self.state_hv_mem = np.expand_dims(self.state_hv_mem, axis=0)

        self.state = np.array(self.state_av + self.state_road + self.state_hv, ndmin=2)
        return self.state, self.state_hv_mem, his_

    def update_vehicle(self, r, a=0, st=0):
        a = self.Cft_Accel * a

        # for fv_pos in self.fv_poses:
        #     if fv_pos['y'] < self.Stop_Line - 1:
        #         fv_pos['a'] = - 0.5 * (fv_pos['v'] ** 2) / (self.Stop_Line - fv_pos['y'])
        #     elif fv_pos['y'] >= self.Stop_Line - 1 and (fv_pos['v'] <= (self.Speed_limit - 2.)):
        #         fv_pos['a'] = 1.
        #     else:
        #         fv_pos['a'] += uniform(-1., 1.) * self.Tau
        #     fv_pos['v'] += fv_pos['a'] * self.Tau
        #     fv_pos['v'] = min(max(0.1, fv_pos['v']), self.Speed_limit)
        #     fv_pos['y'] += fv_pos['v'] * self.Tau + 0.5 * fv_pos['a'] * (self.Tau ** 2)

        for hv_pos in self.hv_poses:
            jerk = uniform(-1., 1.)
            hv_pos['a'] += jerk * self.Tau
            hv_pos['v'] += hv_pos['a'] * self.Tau
            hv_pos['v'] = min(max(0.1, hv_pos['v']), self.Speed_limit)
            hv_pos['x'] = hv_pos['x'] + (hv_pos['v'] * self.Tau + 0.5 * hv_pos['a'] * (self.Tau ** 2)) if \
                hv_pos['dir'] == 'R' else hv_pos['x'] - (hv_pos['v'] * self.Tau + 0.5 * hv_pos['a'] * (self.Tau ** 2))
        old_av_vel = self.av_pos['vy']
        self.av_pos['vy'] += a * self.Tau
        self.av_pos['vy'] = max(0.0, self.av_pos['vy'])
        self.av_pos['y'] += old_av_vel * self.Tau + 0.5 * a * (self.Tau ** 2)
        self.av_pos['heading'] += st
        self.av_pos['accel'] = a
        self.av_pos['steer'] = st
        if self.Visual:
            self.draw_scenary(self.av_pos, self.hv_poses, r)


if __name__ == '__main__':
    sim = InterSim(True)
    plt.ion()
    drawtime = time.time()
    while sim.av_pos['y'] <= sim.Pass_Point:
        sim.get_state()
        sim.update_vehicle(0)
        print('Time: ' + str(time.time() - drawtime))
        drawtime = time.time()
