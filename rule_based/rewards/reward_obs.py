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
    Stop_Line = - 7. - random()
    Pass_Point = 12.
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

    def get_reward(self, state, a=0., b=0., st=0.):
        self.state = state
        r_smooth = self.reward_smooth(a, st)
        r_clerance, collision_l, collision_r = self.reward_clear()
        # r_stop = self.reward_stop(a)
        r_speedlimit = self.reward_speedlimit()
        r_time = - 0.1
        r_crash = - 500. if (collision_l == 1 or (collision_r == 1)) else 0.
        r_dis = self.reward_dis()
        r_finish = self.reward_finish()
        r_notmove, not_move = self.reward_notmove(a)
        r = r_dis + r_time + r_speedlimit + r_clerance + r_smooth + r_finish + r_crash + r_notmove
        return r, collision_l, collision_r, not_move

    def reward_notmove(self, a):
        not_move = 0
        f = 0.
        accel = self.Cft_Accel * a
        # fp = []
        # self.state = [0., 0., -2.]
        # for accel in np.linspace(-5., 5., 100):
        if self.state[5] > 4 and (self.state[10] < 0. or (self.state[10] > 40.)) \
                or (self.state[5] > 0 and (self.state[13] < 0. or (self.state[13] > 40.))):
            if (self.state[0] <= 0.01) and (self.state[2] < 0) and (-1. <= accel < 0.):
                f = 100 * accel
                # f = 100. * (self.tools.sigmoid(accel, 4.) - 0.5) if accel <= 0. else 0.
            if (self.state[0] <= 0.01) and (self.state[2] < 0) and (accel <= - 1.):
                not_move = 1
                f = 100 * accel
                # f = 100. * (self.tools.sigmoid(accel, 4.) - 0.5) if accel <= 0. else 0.
                f -= 500.
            # fp.append(f)
        # plt.plot(np.linspace(-5., 5., 100), fp, 'r.')
        # plt.xlabel('acceleration m/s')
        # plt.ylabel('reward')
        # plt.show()
        return f, not_move

    def reward_smooth(self, a, st):
        #############################################################################
        # jerk = abs(a - self.state[2]) / self.Tau - 1.
        # f1 = 2. * (- self.tools.sigmoid(jerk, 10) + 0.5) if jerk >= 0. else 0.
        # # yaw = (st - self.av_pos['steer']) / self.Tau
        # # f2 = - 2 * abs(self.tools.sigmoid(yaw, 2) - 0.5)
        # f2 = 0.
        #############################################################################
        a = self.Cft_Accel * a
        # fp = []
        jerk = abs(a - self.state[2]) / self.Tau
        # for jerk in np.linspace(-5., 5., 100):
        f1 = - 0.01 * (jerk - 1.) if jerk >= 1. else 0.
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
        f_clear_l, f_clear_r = 100., 100.
        ft1, ft2 = 0., 0.
        if self.state[10] >= 0.:
            f_clear_l = self.state[10] - 5.
            t_clear_l = f_clear_l / self.state[9]
            if f_clear_l <= 20. or t_clear_l <= 1.5:
                ft1 = - 20. * self.state[0] if (self.state[4] <= 0.) else 0.

        if self.state[13] >= 0.:
            f_clear_r = self.state[13] - 5.
            t_clear_r = f_clear_r / self.state[12]
            if f_clear_r <= 30. or t_clear_r <= 2.0:
                ft2 = - 20. * self.state[0] if (self.state[4] <= 0.) else 0.

        fl = 0.
        fr = 0.
        collision_l = (f_clear_l <= 0.) if (-4. <= self.state[11] <= 0.) else 0
        collision_r = (f_clear_r <= 0.) if (-4. <= self.state[14] <= 0.) else 0
        return ft1 + ft2 + fl + fr,  collision_l, collision_r

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
        fx = - 80. * (self.state[0] - self.Speed_limit) if self.state[0] >= self.Speed_limit else 0.
        if self.state[0] >= (self.Speed_limit + 2.):
            fx -= 500
        return fx

    def reward_dis(self):
        dis = (self.state[0]) * self.Tau / (self.Pass_Point - self.state[8]) * 2000.
        return dis

    def reward_finish(self):
        if self.state[7] <= 0.:
            return 1000.
        else:
            return 0.

