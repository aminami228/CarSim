from random import randint, random
from utilities.toolfunc import ToolFunc
import logging
import utilities.log_color

__author__ = 'qzq'


class Reward(object):
    Tau = 1. / 30
    Speed_limit = 12  # m/s
    Scenary = randint(0, 2)
    Inter_Ori = 0.
    Stop_Line = 142.733
    Pass_Point = 168
    Inter_Low = 147.355
    Inter_Up = 157.931
    Inter_Left = 11.545
    Inter_Right = 19.571
    Vehicle_NO = 1
    Lane_Left = 14.935
    Lane_Right = 17.683
    Cft_Accel = 3.  # m/s**2
    Visual = False
    collision = False
    Start_Pos = - 42.

    def __init__(self):
        self.tools = ToolFunc()
        self.state = None

    def get_reward(self, state, a=0, st=0):
        self.state = state
        r_smooth = self.reward_smooth(a, st)
        r_clerance, collision = self.reward_clear()
        r_stop = self.reward_stop(a)
        r_speedlimit = self.reward_speedlimit()
        # r_v = 0.1 * self.av_pos['vy'] - 0.2 if self.av_pos['vy'] <= self.Speed_limit \
        #     else (- 0.6 * self.av_pos['vy'] + 8.4) - 0.2
        # r_v = max(- 0.2, r_v)
        r_time = - 1.
        r_crash = - 500. if collision == 1 else 0.
        r_dis = self.reward_dis()
        r_finish = self.reward_finish()
        r_notmove, not_move = self.reward_notmove(a)
        r = r_dis + r_time + r_speedlimit + r_clerance + r_smooth + r_stop + r_finish + r_crash + r_notmove
        return r, collision, not_move

    def reward_notmove(self, a):
        not_move = 0
        f = 0.
        accel = a
        if (self.state[0] <= 0.01) and (self.state[2] < 0) and (-1. <= accel < 0.):
            f = 100. * (self.tools.sigmoid(accel, 4.) - 0.5) if accel <= 0. else 0.
        if (self.state[0] <= 0.01) and (self.state[2] < 0) and (accel <= - 1.):
            not_move = 1
            f = - 500.
        return f, not_move

    def reward_smooth(self, a, st):
        #############################################################################
        # jerk = abs(a - self.state[2]) / self.Tau - 1.
        # f1 = 2. * (- self.tools.sigmoid(jerk, 10) + 0.5) if jerk >= 0. else 0.
        # # yaw = (st - self.av_pos['steer']) / self.Tau
        # # f2 = - 2 * abs(self.tools.sigmoid(yaw, 2) - 0.5)
        # f2 = 0.
        #############################################################################
        jerk = abs(a - self.state[2]) / self.Tau
        f1 = - 0.1 * (jerk - 1.) if jerk >= 1. else 0.
        f2 = 0.
        return f1 + f2

    def reward_clear(self):
        #############################################################################
        # f_clear = self.state[5] - 10.
        # t_clear = self.state[5] / (self.state[0] - self.state[4]) - 3. if self.state[0] - self.state[4] >= 0.1 else 20.
        # ff = min(0., 5. * (self.tools.sigmoid(f_clear, 0.5) - 0.5))
        # ft = min(0., 2. * (self.tools.sigmoid(t_clear, 6.) - 0.5))
        # l_clear = self.state[7]
        # # fl = self.tools.sigmoid(abs(l_clear), 6) - 0.95
        # fl = 0.
        # r_clear = self.state[8]
        # # fr = self.tools.sigmoid(abs(r_clear), 6) - 0.95
        # fr = 0.
        # collision = (f_clear <= 0.1) or (r_clear <= 0.1) or (l_clear <= 0.1)
        #############################################################################
        f_clear = self.state[5]
        if self.state[0] - self.state[4] >= 0.1:
            t_clear = self.state[5] / (self.state[0] - self.state[4])
        else:
            t_clear = 20.
        ff = - 1. * (10. - f_clear) if f_clear <= 10. else 0.
        ft = - 5. * (1.5 - f_clear) if t_clear <= 1.5 else 0.
        fl = 0.
        fr = 0.
        collision = (f_clear <= 0.1)
        return ff + ft + fl + fr,  collision

    def reward_stop(self, a):
        # th_1 = 2. * self.Cft_Accel
        # th_2 = 2.
        # mid_point = (th_1 + th_2) / 2.
        # a_mean = - self.av_pos['vy'] ** 2. / self.state_road[0]
        # score1 = - a_mean - mid_point
        # f1 = 0.1 * (self.tools.sigmoid(score1, - 2) - 0.5)
        # score2 = a - a_mean / 2.
        # f2 = 2. * (self.tools.sigmoid(score2, - 4) - 0.5) if score2 >= 0 else 0
        # # 0.2 * (self.tools.sigmoid(score2, - 4) - 0.5)
        # a_stop = - self.state[0] ** 2. / (2. * self.state[6])
        # delta = a - a_stop
        # f2 = 2. * (self.tools.sigmoid(delta, - 4) - 0.5) if delta >= 0 else 0
        f = 0.
        if self.state[6] <= 5 and (self.state[0] >= self.state[6]):
            f = - 5. * (self.state[0] - self.state[6])
        return f

    def reward_speedlimit(self):
        #############################################################################
        # th_1 = self.Speed_limit
        # th_2 = th_1 + 2.
        # mid_point = (th_1 + th_2) / 2
        # x = self.state[0] - mid_point
        # fx = min(0., 100. * (self.tools.sigmoid(x, - 3) - 0.5))
        #############################################################################
        fx = - 10. * (self.state[0] - self.Speed_limit) if self.state[0] >= self.Speed_limit else 0.
        return fx

    def reward_dis(self):
        dis = (self.state[0]) * self.Tau / (self.Stop_Line - self.state[9]) * 5000.
        return dis

    def reward_finish(self):
        if self.state[6] <= 2.0 and (self.state[0] <= 2.):
            return 1000.
        else:
            return 0.
