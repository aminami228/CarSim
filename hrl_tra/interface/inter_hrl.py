import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as Rectangle
import matplotlib.image as mpimg
import numpy as np
from random import randint, random, sample, uniform
import time
import logging
import utilities.log_color

__author__ = 'qzq'

Lidar_NO = 30.
Resolution = 2. * np.pi / Lidar_NO
Lidar_len_Max = 50.
Lidar_len_step = 0.1
Visibility = 80.
Focus_No = 10


class InterSim(object):
    Tau = 1. / 30.
    Speed_limit = 12        # m/s
    Inter_Ori = {'x': 0., 'y': 0.}
    Stop_Line = - 7. - random()
    Pass_Point = 12.
    Inter_Low = - 4.
    Inter_Up = 4.
    Inter_Left = - 4.
    Inter_Right = 4.
    FV_NO = 1
    LV_NO = 8
    RV_NO = 8
    Lane_Left = 0.
    Lane_Right = 4.
    Cft_Accel = 3.     # m/s**2

    history_len = 50

    def __init__(self, visual=False):
        self.Visual = visual
        self.av_pos = dict()
        self.av_pos['y'] = self.Stop_Line
        self.Start_Pos = self.av_pos['y']
        self.av_pos['x'] = 2. + random() - 0.5
        self.av_pos['vx'] = 0.
        self.av_pos['vy'] = 0.      # self.Speed_limit - random() * 5.
        self.av_pos['heading'] = 0.
        self.av_pos['accel'] = 0.
        self.av_pos['steer'] = 0.
        self.av_pos['l'] = 4. + 0.4 * random() - 0.2
        self.av_pos['w'] = 2. + 0.2 * random() - 0.1

        self.fig = None
        self.ax = None

        self.fv_poses = []
        for i in range(self.FV_NO):
            fv_pos = dict()
            fv_pos['y'] = self.av_pos['y'] + random() * 30. + 15.
            fv_pos['x'] = 2.
            fv_pos['v'] = self.Speed_limit - random()
            fv_pos['a'] = 0.
            fv_pos['l'] = 4. + 2. * random()
            fv_pos['w'] = 2. + random() - 0.5
            self.fv_poses.append(fv_pos)

        # self.sce = GenScen()
        self.lv_poses, self.rv_poses = [], []
        self.hv_poses = self.lv_poses + self.rv_poses

        self.state_dim = None
        self.state = None
        self.state_av = []
        self.state_fv = []
        self.state_hv = []
        self.state_road = []

    def draw_scenary(self, av, hvs, fvs, a, r=0):
        if self.Visual:
            self.fig = plt.figure(1)
            self.ax = self.fig.add_subplot(1, 1, 1)
            plt.axis([-100, 100, -110, 110])
            self.ax.fill_between(np.arange(-104, self.Inter_Left, 0.5), self.Inter_Low,
                                 np.arange(self.Inter_Low, -104, -0.5), facecolor='black')
            self.ax.fill_between(np.arange(-104, self.Inter_Left, 0.5), self.Inter_Up,
                                 np.arange(self.Inter_Up, 104, 0.5), facecolor='black')
            self.ax.fill_between(np.arange(self.Inter_Right, 104, 0.5), self.Inter_Low,
                                 np.arange(-104, self.Inter_Low, 0.5), facecolor='black')
            self.ax.fill_between(np.arange(self.Inter_Right, 104, 0.5), self.Inter_Up,
                                 np.arange(104, self.Inter_Up, -0.5), facecolor='black')
            self.ax.add_patch(Rectangle((av['x'] - av['w'] / 2., av['y'] - av['l']), av['w'], av['l'], color='red'))
            for v in hvs:
                if v['dir'] == 'L':
                    self.ax.add_patch(Rectangle((v['x'] - v['l'], v['y'] - v['w'] / 2.), v['l'], v['w'], color='green'))
                if v['dir'] == 'R':
                    self.ax.add_patch(Rectangle((v['x'], v['y'] - v['w'] / 2.), v['l'], v['w'], color='green'))
            for v in fvs:
                self.ax.add_patch(Rectangle((v['x'] - v['w'] / 2., v['y']), v['w'], v['l'], color='green'))
            # plt.axis([-100, 100, -110, 110])
            self.ax.plot(list(xrange(-104, 104)), list([(self.Inter_Up + self.Inter_Low) / 2.] * 208), 'y--')
            self.ax.plot(list([(self.Inter_Right + self.Inter_Left) / 2.] * 208), list(xrange(-104, 104)), 'y--')
            self.ax.plot(list([-Visibility] * 10), list(xrange(-5, 5)), 'r')
            self.ax.plot(list([Visibility] * 10), list(xrange(-5, 5)), 'r')
            self.ax.plot(list(xrange(0, 5)), list([self.Stop_Line] * 5), 'r')
            plt.text(av['x'], av['y'], 'a: {0:.0f}'.format(a) + ', v: {0:.2f}'.format(av['vy']) +
                     '\n reward: {0:.2f}'.format(r) + ', center_dis: {0:.2f}'.format(self.Inter_Ori['y'] - av['y'])
                     + ', fv_dis: {0:.2f}'.format(self.fv_poses[0]['y'] - self.av_pos['y']),
                     color='red')
            plt.show()
            plt.pause(0.1)
            plt.clf()

    def get_state(self):
        # 0 - 3
        self.state_av = [self.av_pos['vy'], self.av_pos['accel'], self.av_pos['l'], self.av_pos['w']]

        sl_dis = self.Stop_Line - self.av_pos['y']
        int_lower_dis = self.Inter_Low - self.av_pos['y']
        int_center_y = self.Inter_Ori['y'] - self.av_pos['y']
        int_center_x = self.Inter_Ori['x'] - self.av_pos['x']
        int_upper_dis = self.Inter_Up - self.av_pos['y']
        pass_dis = self.Pass_Point - self.av_pos['y']
        ll = self.av_pos['x'] - self.Inter_Left
        lr = self.Inter_Right - self.av_pos['x']
        start_pos = self.Start_Pos
        # 4 - 12
        self.state_road = [sl_dis, int_lower_dis, int_center_y, int_center_x, int_upper_dis, pass_dis, start_pos, ll, lr]

        # 13 14
        fv_dis_list = [fv_pos['y'] - self.av_pos['y'] for fv_pos in self.fv_poses]
        fv_index = np.argmin(fv_dis_list)
        fv_pos = self.fv_poses[fv_index]
        if fv_pos['y'] - self.av_pos['y'] > Visibility:
            self.state_fv = [self.Speed_limit, Visibility]
        else:
            self.state_fv = [fv_pos['v'], fv_pos['y'] - self.av_pos['y']]

        # 15 - 34, 35 - 54
        lv_cand = []
        rv_cand = []
        for v1, v2 in zip(self.lv_poses, self.rv_poses):
            if self.Inter_Ori['x'] - Visibility < v1['x'] < self.Inter_Right:
                lv_cand.append(v1)
            if self.Inter_Ori['x'] < v2['x'] < self.Inter_Right + Visibility:
                rv_cand.append(v1)
        # dis_ = - lr + self.av_pos['w'] / 2.
        dis_ = - 3.
        veh_no = Focus_No
        crash_l, crash_r = [], []
        while veh_no >= 0:
            if not lv_cand:
                crash_l = [dis_, dis_ / 20.] * veh_no + crash_l
                break
            elif self.av_pos['x'] - self.av_pos['w'] / 2. - lv_cand[0]['x'] > lv_cand[0]['l']:
                c_dis = self.av_pos['x'] - self.av_pos['w'] / 2. - lv_cand[0]['x'] - lv_cand[0]['l']
                crash_l += [c_dis, c_dis / lv_cand[0]['v']]
            elif lv_cand[0]['x'] > (self.av_pos['x'] + self.av_pos['w'] / 2.):
                c_dis = self.av_pos['x'] + self.av_pos['w'] / 2. - lv_cand[0]['x']
                crash_l += [c_dis, c_dis / lv_cand[0]['v']]
            else:
                crash_l += [0., 0.]
            lv_cand.pop(0)
            veh_no -= 1
        veh_no = Focus_No
        while veh_no >= 0:
            if not rv_cand:
                crash_r = [dis_, dis_ / 20.] * veh_no + crash_r
                break
            elif rv_cand[0]['x'] - (self.av_pos['x'] + self.av_pos['w'] / 2.) > rv_cand[0]['l']:
                c_dis = rv_cand[0]['x'] - (self.av_pos['x'] + self.av_pos['w'] / 2.) - rv_cand[0]['l']
                crash_r += [c_dis, c_dis / rv_cand[0]['v']]
            elif rv_cand[0]['x'] < (self.av_pos['x'] - self.av_pos['w'] / 2.):
                c_dis = rv_cand[0]['x'] - (self.av_pos['x'] - self.av_pos['w'] / 2.)
                crash_r += [c_dis, c_dis / rv_cand[0]['v']]
            else:
                crash_r += [0., 0.]
            rv_cand.pop(0)
            veh_no -= 1

        self.state_hv = crash_l + crash_r
        self.state = np.array(self.state_av + self.state_road + self.state_fv + self.state_hv, ndmin=2)
        return self.state

    def update_vehicle(self, a=0, r=0, st=0):
        accel = self.Cft_Accel * a
        for fv_pos in self.fv_poses:
            if fv_pos['y'] < self.Stop_Line - 1:
                fv_pos['a'] = - 0.5 * (fv_pos['v'] ** 2) / (self.Stop_Line - fv_pos['y'])
            elif fv_pos['y'] >= self.Stop_Line - 1 and (fv_pos['v'] <= (self.Speed_limit - 2.)):
                fv_pos['a'] = 1.
            else:
                fv_pos['a'] += uniform(-1., 1.) * self.Tau
            fv_pos['v'] += fv_pos['a'] * self.Tau
            fv_pos['v'] = min(max(0.1, fv_pos['v']), self.Speed_limit)
            fv_pos['y'] += fv_pos['v'] * self.Tau + 0.5 * fv_pos['a'] * (self.Tau ** 2)

        if self.av_pos['y'] >= self.Stop_Line and (not self.hv_poses):
            lv_locs = np.array(sample(xrange(-15, 2), self.LV_NO))
            lv_locs = 10. * np.array(sorted(lv_locs, reverse=True)) + 2. * random() - 1.
            rv_locs = np.array(sample(xrange(0, 17), self.RV_NO))
            rv_locs = 10. * np.array(sorted(rv_locs)) + 2. * random() - 1.
            for x1, x2 in zip(lv_locs, rv_locs):
                lv_pos = dict()
                rv_pos = dict()
                lv_pos['y'] = (self.Inter_Ori['y'] + self.Inter_Low) / 2.
                rv_pos['y'] = (self.Inter_Ori['y'] + self.Inter_Up) / 2.
                lv_pos['x'] = x1
                rv_pos['x'] = x2
                lv_pos['v'] = self.Speed_limit - random()
                rv_pos['v'] = self.Speed_limit - random()
                lv_pos['a'] = 0.
                lv_pos['a'] = 0.
                lv_pos['l'] = 4. + 2. * random()
                rv_pos['l'] = 4. + 2. * random()
                lv_pos['w'] = 2. + random() - 0.5
                rv_pos['w'] = 2. + random() - 0.5
                lv_pos['dir'] = 'R'
                rv_pos['dir'] = 'L'
                self.lv_poses.append(lv_pos)
                self.rv_poses.append(rv_pos)
            self.hv_poses = self.lv_poses + self.rv_poses

        for hv_pos in self.hv_poses:
            hv_pos['v'] = min(max(0.1, hv_pos['v']), self.Speed_limit)
            hv_pos['x'] = hv_pos['x'] + hv_pos['v'] * self.Tau if hv_pos['dir'] == 'R' \
                else hv_pos['x'] - hv_pos['v'] * self.Tau
        old_av_vel = self.av_pos['vy']
        self.av_pos['vy'] += accel * self.Tau
        self.av_pos['vy'] = max(0.0, self.av_pos['vy'])
        s = old_av_vel * self.Tau + 0.5 * accel * (self.Tau ** 2)
        s = max(0.0, s)
        self.av_pos['y'] += s
        self.av_pos['heading'] += st
        self.av_pos['accel'] = accel
        self.av_pos['steer'] = st
        if self.Visual:
            self.draw_scenary(self.av_pos, self.hv_poses, self.fv_poses, accel, r)


if __name__ == '__main__':
    sim = InterSim(True)
    plt.ion()
    drawtime = time.time()
    while sim.av_pos['y'] <= sim.Pass_Point:
        sim.draw_scenary(sim.av_pos, sim.hv_poses, sim.fv_poses)
        sim.get_state()
        sim.update_vehicle(0)
        print('Time: ' + str(time.time() - drawtime))
        drawtime = time.time()
