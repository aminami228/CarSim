from random import randint, random, sample, uniform
import numpy as np

__author__ = 'qzq'


class GenScen(object):

    def __int__(self):
        self.lv_poses = []
        self.rv_poses = []
        self.cond = None

        self.Inter_Ori = {'x': 0., 'y': 0.}
        self.Stop_Line = - 7. - random()
        self.Pass_Point = 12.
        self.Inter_Low = - 4.
        self.Inter_Up = 4.
        self.Inter_Left = - 4.
        self.Inter_Right = 4.
        self.Lane_Left = 0.
        self.Lane_Right = 4.
        self.Speed_limit = 12.  # m/s

    def gen_front(self):
        # self.fv_poses = []
        # fv_locs = 15. * np.array(sample(xrange(1, 4), self.FV_NO)) + random()
        # for y in fv_locs:
        #     fv_pos = dict()
        #     fv_pos['y'] = self.av_pos['y'] + y
        #     fv_pos['x'] = 2.
        #     fv_pos['v'] = self.Speed_limit - random()
        #     fv_pos['a'] = 0.
        #     self.fv_poses.append(fv_pos)
        pass

    def gen_left(self, l_no):
        lv_locs = np.array(sample(xrange(-50, 2), l_no))

        # if self.v_no <= 4:
        #     theta = randint(self.v_no * 2 + 2, (self.v_no + 1) * 4)
        #
        #     self.cond = 'V_NO: ' + str(self.v_no) + ', Interval: ' + str(theta)
        #     lv_locs = np.array(sample(xrange(-theta, 2), self.v_no))

        # if lam > gamma:
        #     pass
        # else:
        #     self.cond = 'tight'
        #     lv_locs = np.array(sample(xrange(-8, 2), self.LV_NO))

        lv_locs = 10. * np.array(sorted(lv_locs, reverse=True)) + 2. * random() - 1.
        for x in lv_locs:
            lv_pos = dict()
            lv_pos['y'] = (self.Inter_Ori['y'] + self.Inter_Low) / 2.
            lv_pos['x'] = x
            lv_pos['v'] = self.Speed_limit - random()
            lv_pos['a'] = 0.
            lv_pos['l'] = 4. + 2. * random()
            lv_pos['w'] = 2. + random() - 0.5
            lv_pos['dir'] = 'R'
            self.lv_poses.append(lv_pos)

    def gen_right(self, rv=0):
        # rv_locs = 10. * np.array(sample(xrange(-1, 13), self.RV_NO)) + random()
        # for x in rv_locs:
        #     rv_pos = dict()
        #     rv_pos['y'] = self.Inter_Up - 2.
        #     rv_pos['x'] = x
        #     rv_pos['v'] = self.Speed_limit - 10. * random()
        #     rv_pos['a'] = uniform(-1., 1.)
        #     rv_pos['dir'] = 'L'
        #     self.rv_poses.append(rv_pos)
        return self.rv_poses

    def get_vehicles(self, lv=0, rv=0):
        return self.gen_left(lv), self.gen_right(rv)
