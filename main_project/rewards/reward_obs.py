from random import randint, random
from utilities.toolfunc import ToolFunc
import logging
import utilities.log_color

__author__ = 'qzq'


class ObsReward(object):
    Tau = 1. / 30
    Speed_limit = 12.  # m/s
    Scenary = randint(0, 2)
    Inter_Ori = 0.
    Stop_Line = - 5. - random()
    Pass_Point = 20.
    Inter_Low = - 4.
    Inter_Up = 4.
    Inter_Left = - 4.
    Inter_Right = 4.
    Lane_Left = 0.
    Lane_Right = 4.
    Cft_Accel = 3.  # m/s**2
    Full_Accel = 5.
    Full_Brake = 5.

    def __init__(self):
        self.tools = ToolFunc()
        self.state = None
        self.state_rec_his = None

    def get_reward(self, state, state_his, a=0., b=0., st=0.):
        self.state = state
        self.state_rec_his = state_his[-1]
        r_smooth = self.reward_smooth(a, st)
        r_clerance, collision = self.reward_clear()
        # r_stop = self.reward_stop(a)
        r_speedlimit = self.reward_speedlimit()
        r_time = - 1.
        r_crash = - 500. if collision == 1 else 0.
        r_dis = self.reward_dis()
        r_finish = self.reward_finish()
        r_notmove, not_move = self.reward_notmove(a)
        r = r_dis + r_time + r_speedlimit + r_clerance + r_smooth + r_finish + r_crash + r_notmove
        return r, collision, not_move

    def reward_notmove(self, a):
        not_move = 0
        f = 0.
        accel = self.Cft_Accel * a
        # fp = []
        # self.state = [0., 0., -2.]
        # for accel in np.linspace(-5., 5., 100):
        if self.state[7] > 1.0:
            if self.state_rec_his[1] < 0 and (self.state_rec_his[3] < 0):
                if (self.state[0] <= 0.01) and (self.state[2] < 0) and (-1. <= accel < 0.):
                    f = 100. * (self.tools.sigmoid(accel, 4.) - 0.5) if accel <= 0. else 0.
                if (self.state[0] <= 0.01) and (self.state[2] < 0) and (accel <= - 1.):
                    not_move = 1
                    f = - 500.
            # fp.append(f)
        # plt.plot(np.linspace(-5., 5., 100), fp, 'r.')
        # plt.xlabel('acceleration m/s')
        # plt.ylabel('reward')
        # plt.show()
        return f, not_move

    def reward_smooth(self, a, st):
        #############################################################################
        # jerk = abs(a - self.state[2]) / self.Tau
        # x1 = abs(jerk) - 1.
        # f1 = 4. * (- self.tools.sigmoid(x1, 3.) + 0.5) if x1 >= 0. else 0.
        # # yaw = (st - self.av_pos['steer']) / self.Tau
        # # f2 = - 2 * abs(self.tools.sigmoid(yaw, 2) - 0.5)
        # f2 = 0.
        #############################################################################
        accel = self.Cft_Accel * a
        jerk = abs(accel - self.state[2]) / self.Tau
        # for jerk in np.linspace(-5., 5., 100):
        f1 = - 0.1 * (jerk - 1.) if jerk >= 1. else 0.
        f2 = 0.
        # fp.append(f1 + f2)
        # plt.plot(np.linspace(-5., 5., 100), fp, 'r.')
        # plt.xlabel('|jerk| m/s')
        # plt.ylabel('reward')
        # plt.show()
        return f1 + f2

    def reward_cft(self, a, b):
        #############################################################################
        accel = self.Full_Accel * a - self.Full_Brake * b
        x = abs(accel) - self.Cft_Accel
        f = 4. * (- self.tools.sigmoid(x, 2.) + 0.5) if x >= 0. else 0.
        #############################################################################
        # f1 = - (abs(a) - self.Cft_Accel) if abs(a) >= self.Cft_Accel else 0.
        # f2 = - (abs(b) - self.Cft_Accel) if abs(b) >= self.Cft_Accel else 0.
        return f

    def reward_clear(self):
        #############################################################################
        # crash_dis_l = self.state[-2] - 5.
        # crash_dis_r = self.state[-1] - 5.
        # crash_time_l = crash_dis_l / self.state_his[0] if self.state_his[0] > 0.1 else 100.
        # crash_time_r = crash_dis_r / self.state_his[1] if self.state_his[1] > 0.1 else 100.
        # x1 = crash_dis_l - 10.
        # x2 = crash_time_l - 3.
        # x3 = crash_time_r - 10.
        # x4 = crash_time_r - 3.
        # f1 = 4. * (self.tools.sigmoid(x1, 0.3) - 0.5) if x1 <= 0. else 0.
        # f2 = 50. * (self.tools.sigmoid(x2, 1.) - 0.5) if x2 <= 0. else 0.
        # f3 = 4. * (self.tools.sigmoid(x3, 0.3) - 0.5) if x3 <= 0. else 0.
        # f4 = 50. * (self.tools.sigmoid(x4, 1.) - 0.5) if x4 <= 0. else 0.
        # # crash_left = self.state[7]
        # # f3 = 0.
        # # crash_right = self.state[8]
        # # f4 = 0.
        # collision = (crash_dis_l <= 0.1 or (crash_dis_r <= 0.1))
        #############################################################################
        f_clear_l = self.state_rec_his[1] - 5. if self.state[5] >= - 2 else 10000.
        f_clear_r = self.state_rec_his[3] - 5. if self.state[6] >= - 2 else 10000.
        t_clear_l = f_clear_l / self.state_rec_his[0] if self.state_rec_his[0] > 0.1 else 20.
        t_clear_r = f_clear_r / self.state_rec_his[2] if self.state_rec_his[2] > 0.1 else 20.
        t_clear_l = min(t_clear_l, 20.)
        t_clear_r = min(t_clear_r, 20.)
        # fp = []
        # for t_clear in np.linspace(0., 10., 100):
        ff1 = - 1. * (10. - f_clear_l) if f_clear_l <= 10. else 0.
        ff2 = - 1. * (10. - f_clear_r) if f_clear_r <= 10. else 0.
        ft1 = - 10. * (1.5 - t_clear_l) if t_clear_l <= 1.5 else 0.
        ft2 = - 10. * (1.5 - t_clear_r) if t_clear_r <= 1.5 else 0.
        # fp.append(ft)
        # plt.plot(np.linspace(0., 10., 100), fp, 'r')
        # plt.xlabel('Crash time to front vehilce s')
        # plt.ylabel('reward')
        # plt.show()
        fl = 0.
        fr = 0.
        collision = (f_clear_l <= 0.1 or (f_clear_r <= 0.1))
        return ff1 + ff2 + ft1 + ft2 + fl + fr,  collision

    def reward_stop(self):
        #############################################################################
        x = self.state[0] / self.state[6] - 1.
        f = 100. * (- self.tools.sigmoid(x, 2.5) + 0.5) if x >= 0. else 0.
        f = f if self.state[6] <= 5. else 0.
        #############################################################################
        # f = 0.
        # if self.state[6] <= 5 and (self.state[0] >= self.state[6]):
        #     f = - 10. * (self.state[0] - self.state[6])
        return f

    def reward_speedlimit(self):
        #############################################################################
        # x = self.state[0] - self.Speed_limit
        # f = 100. * (- self.tools.sigmoid(x, 1.5) + 0.5) if x >= 0. else 0.
        # if x >= 2.:
        #     f -= 500
        #############################################################################
        fx = - 30. * (self.state[0] - self.Speed_limit) if self.state[0] >= self.Speed_limit else 0.
        if self.state[0] >= (self.Speed_limit + 2.):
            fx = - 500
        return fx

    def reward_dis(self):
        dis = (self.state[0]) * self.Tau / (self.Pass_Point - self.state[10]) * 5000.
        return dis

    def reward_finish(self):
        if self.state[7] <= 1.0:
            return 1000.
        else:
            return 0.

