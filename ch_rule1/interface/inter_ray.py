import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as Rectangle
import matplotlib.image as mpimg
import numpy as np
from random import randint, random, sample, uniform
from utilities.toolfunc import ToolFunc
import time
from math import ceil, cos, sin, pi, floor
import pickle
import logging
import utilities.log_color

__author__ = 'qzq'

Lidar_NO = 30.
Resolution = 2. * np.pi / Lidar_NO
Lidar_len_Max = 50.
Lidar_len_step = 0.1


class InterSim(object):
    Tau = 1. / 10.
    Speed_limit = 12        # m/s
    Scenary = randint(0, 2)
    Inter_Ori = [0., 0.]
    Stop_Line = - 7. - random()
    Pass_Point = 12.
    Inter_Low = - 4.
    Inter_Up = 4.
    Inter_Left = - 4.
    Inter_Right = 4.
    FV_NO = 2
    LV_NO = 8
    RV_NO = 0
    Lane_Left = 0.
    Lane_Right = 4.
    Cft_Accel = 3.     # m/s**2

    history_len = 50
    tools = ToolFunc()

    def __init__(self, visual=False):
        self.Visual = visual
        self.av_pos = dict()
        self.av_pos['y'] = self.Stop_Line - 0.5 + random()
        self.Start_Pos = self.av_pos['y']
        self.av_pos['x'] = 2.
        self.av_pos['vx'] = 0.
        self.av_pos['vy'] = 0.5
        self.av_pos['heading'] = 0.
        self.av_pos['accel'] = 0
        self.av_pos['steer'] = 0
        self.av_size = [4, 2]

        self.map = None
        # self.fig = plt.figure(1)
        self.fig = None
        self.ax = None

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
        lv_locs = 10. * np.array(sample(xrange(-9, 2), self.LV_NO)) + 2. * random() - 1.
        for x in lv_locs:
            lv_pos = dict()
            lv_pos['y'] = self.Inter_Low + 2.
            lv_pos['x'] = x
            lv_pos['v'] = self.Speed_limit - 5. * random()
            lv_pos['a'] = 0.
            lv_pos['l'] = 4. + random()
            lv_pos['dir'] = 'R'
            self.lv_poses.append(lv_pos)
        self.rv_poses = []
        rv_locs = 10. * np.array(sample(xrange(-1, 13), self.RV_NO)) + random()
        for x in rv_locs:
            rv_pos = dict()
            rv_pos['y'] = self.Inter_Up - 2.
            rv_pos['x'] = x
            rv_pos['v'] = self.Speed_limit - 10. * random()
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
        self.state_vm = []

    def draw_scenary(self, av, hvs, r=0):
        # black #000000 rgb(0,0,0)
        # green	#008000	rgb(0,128,0)
        # yellow #FFFF00 rgb(255,255,0)
        # red #FF0000 rgb(255,0,0)
        # coral #FF7F50 rgb(255,127,80)

        self.fig = plt.figure(1)
        self.ax = self.fig.add_subplot(1, 1, 1)
        plt.axis([-100, 100, -80, 80])
        self.ax.fill_between(np.arange(-104, self.Inter_Left, 0.5), self.Inter_Low,
                             np.arange(self.Inter_Low, -104, -0.5), facecolor='black')
        self.ax.fill_between(np.arange(-104, self.Inter_Left, 0.5), self.Inter_Up,
                             np.arange(self.Inter_Up, 104, 0.5), facecolor='black')
        self.ax.fill_between(np.arange(self.Inter_Right, 104, 0.5), self.Inter_Low,
                             np.arange(-104, self.Inter_Low, 0.5), facecolor='black')
        self.ax.fill_between(np.arange(self.Inter_Right, 104, 0.5), self.Inter_Up,
                             np.arange(104, self.Inter_Up, -0.5), facecolor='black')
        self.ax.add_patch(Rectangle((av['x'] - 1., av['y'] - 4.), 2, 4, color='red'))
        for v in hvs:
            if v['dir'] == 'L':
                self.ax.add_patch(Rectangle((v['x'] - 4., v['y'] - 1.), 4, 2, color='green', fill=True))
            if v['dir'] == 'R':
                self.ax.add_patch(Rectangle((v['x'], v['y'] - 1.), 4, 2, color='green', fill=True))
        # self.fig.savefig('map.png', dpi=self.fig.dpi)
        # self.map = mpimg.imread('map.png')
        # self.state_vm = self.visibilty_map(av)
        if self.Visual:
            plt.axis([-100, 100, -80, 80])
            self.ax.plot(list(xrange(-104, 104)),
                         list([(self.Inter_Up - self.Inter_Low) / 2. + self.Inter_Low] * 208), 'y--')
            self.ax.plot(list([(self.Inter_Right - self.Inter_Left) / 2. + self.Inter_Left] * 208),
                         list(xrange(-104, 104)), 'y--')
            plt.text(av['x'], av['y'], 'a: {0:.2f}'.format(av['accel']) + ', v: {0:.2f}'.format(av['vy']) +
                     '\n reward: {0:.2f}'.format(r) + ', center_dis: {0:.2f}'.format(self.Inter_Ori[1] - av['y']),
                     color='red')
            plt.show()
            plt.pause(0.001)
            plt.clf()
        else:
            plt.clf()

    def get_pixle_color(self, x, y):
        xy_pixels = self.ax.transData.transform(np.vstack([x, y]).T)
        xpix, ypix = xy_pixels.T
        width, height = self.fig.canvas.get_width_height()
        ypix = height - ypix
        return self.map[int(round(ypix)), int(round(xpix))]

    def visibilty_map(self, av_pos):
        dis_vm = []
        poses = [[av_pos['x'] - 1., av_pos['y']],
                 [av_pos['x'] + 1., av_pos['y']]]
        angle_range = [[7./6. * pi, 4./15. * pi],
                       [2./3. * pi, -7./30. * pi]]
        # poses = [[av_pos['x'] - 1., av_pos['y'] - 4.],
        #        [av_pos['x'] - 1., av_pos['y'] - 2.],
        #        [av_pos['x'] - 1., av_pos['y']],
        #        [av_pos['x'], av_pos['y']],
        #        [av_pos['x'] + 1., av_pos['y']],
        #        [av_pos['x'] + 1., av_pos['y'] - 2.],
        #        [av_pos['x'] + 1., av_pos['y'] - 4.]]
        # angle_range = [[7. / 6. * pi, 5. / 6. * pi],
        #                [7. / 6. * pi, 5. / 6. * pi],
        #                [pi, 1. / 2. * pi],
        #                [2. / 3. * pi, 1. / 3. * pi],
        #                [1. / 2. * pi, 0.],
        #                [1. / 6. * pi, - 1. / 6. * pi],
        #                [1. / 6. * pi, - 1. / 6. * pi]]
        for pos, angle in zip(poses, angle_range):
            for theta in np.arange(angle[0], angle[1], - Resolution):
                l = 0.
                while True:
                    if self.get_pixle_color(pos[0] + l * cos(theta), pos[1] + l * sin(theta))[0] > 0.5 \
                            and l < Lidar_len_Max:
                        l += Lidar_len_step
                    else:
                        dis_vm.append(l)
                        if self.Visual:
                            self.ax.plot([pos[0], pos[0] + l * cos(theta)], [pos[1], pos[1] + l * sin(theta)],
                                         color='fuchsia')
                            self.ax.plot(pos[0] + l * cos(theta), pos[1] + l * sin(theta), 'r*', markersize=4)
                        break
        return dis_vm

    def get_his_state(self, his_):
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

    def get_state_all(self):
        self.state_av = [self.av_pos['vy'], self.av_pos['accel']]

        # fv_dis_list = [fv_pos['y'] - self.av_pos['y'] for fv_pos in self.fv_poses]
        # fv_index = np.argmin(fv_dis_list)
        # fv_pos = self.fv_poses[fv_index]
        # self.state_fv = [fv_pos['v'], fv_pos['y'] - self.av_pos['y'] - 4.]

        int_lower_dis = self.Inter_Low - self.av_pos['y']
        int_center_y = self.Inter_Ori[1] - self.av_pos['y']
        int_center_x = self.Inter_Ori[0] - self.av_pos['x']
        int_upper_dis = self.Inter_Up - self.av_pos['y']
        pass_dis = self.Pass_Point - self.av_pos['y']
        ll = self.av_pos['x'] - self.Inter_Left
        lr = self.Inter_Right - self.av_pos['x']
        start_pos = self.Start_Pos
        self.state_road = [int_lower_dis, int_center_y, int_center_x, int_upper_dis, pass_dis, start_pos, ll, lr]

        lv_dis_list = [self.av_pos['x'] - lv_pos['x'] for lv_pos in self.lv_poses]
        l_min_dis = np.inf
        lv_pos = self.lv_poses[0]
        # l_y, r_y = -6., -6.
        for i, dis in enumerate(lv_dis_list):
            if dis <= l_min_dis and (dis > -1.):
                lv_pos = self.lv_poses[i]
                l_min_dis = dis
                # l_y = lv_pos['y'] - self.av_pos['y'] - 1.
        if np.isinf(l_min_dis):
            l_min_dis = self.av_pos['x'] - self.Inter_Right
            lv_pos['v'] = self.Speed_limit
        rv_dis_list = [rv_pos['x'] - self.av_pos['x'] for rv_pos in self.rv_poses]
        r_min_dis = np.inf
        rv_pos = self.rv_poses[0]
        for i, dis in enumerate(rv_dis_list):
            if dis <= r_min_dis and (dis > -1.):
                rv_pos = self.rv_poses[i]
                r_min_dis = dis
                # r_y = rv_pos['y'] - self.av_pos['y'] - 1.
        if np.isinf(r_min_dis):
            r_min_dis = self.Inter_Left - self.av_pos['x']
            rv_pos['v'] = self.Speed_limit
        self.state_hv = [lv_pos['v'], l_min_dis, rv_pos['v'], r_min_dis]
        self.state = np.array(self.state_av + self.state_road + self.state_hv, ndmin=2)
        return self.state

    def get_state(self):
        self.state_av = [self.av_pos['vy'], self.av_pos['accel']]

        int_lower_dis = self.Inter_Low - self.av_pos['y']
        int_center_y = self.Inter_Ori[1] - self.av_pos['y']
        int_center_x = self.Inter_Ori[0] - self.av_pos['x']
        int_upper_dis = self.Inter_Up - self.av_pos['y']
        pass_dis = self.Pass_Point - self.av_pos['y']
        ll = self.av_pos['x'] - self.Inter_Left
        lr = self.Inter_Right - self.av_pos['x']
        start_pos = self.Start_Pos
        self.state_road = [int_lower_dis, int_center_y, int_center_x, int_upper_dis, pass_dis, start_pos, ll, lr]

        lv_dis_list = [max(-2., self.av_pos['x'] - lv_pos['x']) for lv_pos in self.lv_poses]
        lv_v_list = [pos['v'] for pos in self.lv_poses]
        self.state_hv = lv_v_list + lv_dis_list
        self.state = np.array(self.state_av + self.state_road + self.state_hv, ndmin=2)
        return self.state

    def update_vehicle(self, a=0, r=0, st=0):
        accel = self.Cft_Accel * a

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
            # jerk = uniform(-1., 1.)
            jerk = 0.
            hv_pos['a'] += jerk * self.Tau
            hv_pos['v'] += hv_pos['a'] * self.Tau
            hv_pos['v'] = min(max(0.1, hv_pos['v']), self.Speed_limit)
            hv_pos['x'] = hv_pos['x'] + max(0., (hv_pos['v'] * self.Tau + 0.5 * hv_pos['a'] * (self.Tau ** 2))) \
                if hv_pos['dir'] == 'R' \
                else hv_pos['x'] - max(0., (hv_pos['v'] * self.Tau + 0.5 * hv_pos['a'] * (self.Tau ** 2)))
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
            self.draw_scenary(self.av_pos, self.hv_poses, r)


if __name__ == '__main__':
    sim = InterSim(True)
    plt.ion()
    drawtime = time.time()
    while sim.av_pos['y'] <= sim.Pass_Point:
        sim.draw_scenary(sim.av_pos, sim.hv_poses)
        sim.get_state()
        sim.update_vehicle(0)
        print('Time: ' + str(time.time() - drawtime))
        drawtime = time.time()
