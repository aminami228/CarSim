from random import randint, random
from utilities.toolfunc import ToolFunc
import math
import numpy as np
import logging
import utilities.log_color

__author__ = 'qzq'

Safe_dis = 30.
Safe_time = 3.
Tho_dis = 10.
Tho_time = 1.5
Visibility = 50.
Focus_No = 10


class ObsReward(object):
    Tau = 1. / 10.
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
    LV_NO = 8

    def __init__(self):
        self.tools = ToolFunc()
        self.state = None
        self.state_rec_his = None
        self.lv_v_list = None
        self.lv_dis_list = None

    def get_reward(self, state, a=0., b=0., st=0.):
        self.state = state
        r_smooth, jerk = self.reward_smooth(a, st)
        r_clerance, collision_l, collision_r = self.reward_clear()
        # r_stop = self.reward_stop(a)
        r_speedlimit = self.reward_speedlimit()
        r_time = - 1.
        r_crash = - 500. if (collision_l == 1 or (collision_r == 1)) else 0.
        r_dis = self.reward_dis()
        r_finish = self.reward_finish()
        r_notmove, not_move = self.reward_notmove(a)
        r = r_dis + r_time + r_speedlimit + r_clerance + r_smooth + r_finish + r_crash + r_notmove
        return r, collision_l, collision_r, not_move, jerk

    def reward_old_notmove(self, a):
        not_move = 0
        f = 0.
        accel = self.Cft_Accel * a
        crash_time_l = [dis / v if dis > -2. else 20. for dis, v in zip(self.lv_dis_list, self.lv_v_list)]
        crash_time_l = np.array(crash_time_l)
        pass_crash_l = math.sqrt((self.state[2] + 8.) * 2.) if (self.state[2] >= -8.) else 0.
        lv_dis_list = np.array(self.lv_dis_list)
        lv_dis_list = lv_dis_list[lv_dis_list > -2.]
        if len(self.lv_dis_list[self.lv_dis_list > -2.]) == 0 \
                or (len(crash_time_l[crash_time_l <= pass_crash_l]) == 0
                    and (len(lv_dis_list[lv_dis_list <= 30.]) == 0)):
            if (self.state[0] <= 0.01) and (self.state[1] < 0) and (-0.5 <= accel < 0.):
                f = 100 * accel
                # f = 100. * (self.tools.sigmoid(accel, 4.) - 0.5) if accel <= 0. else 0.
            if (self.state[0] <= 0.01) and (self.state[1] < 0) and (accel < - 0.5):
                not_move = 1
                f = 100 * accel
                # f = 100. * (self.tools.sigmoid(accel, 4.) - 0.5) if accel <= 0. else 0.
                f -= 500.
        return f, not_move

    def reward_notmove(self, a):
        not_move = 0
        f = 0.
        accel = self.Cft_Accel * a
        if self.state[-2] <= -3.:     # or (self.state[12] > Safe_dis and (self.state[13] > Safe_time))\
                # or (self.state[5] < -self.state[2]):
            if (self.state[0] <= 0.01) and (accel <= 0):
                f = 50 * (accel - self.Cft_Accel)
                f -= 500.
                not_move = 1
        ###
        if self.state[-22] <= -3. and (self.state[-2] <= -3.)\
                and (self.state[0] == 0.) and (self.state[1] < 0):
            if - 0.5 < accel < 0.:
                f = 50 * (accel - self.Cft_Accel)
            elif accel <= - 0.5:
                f = 50 * (accel - self.Cft_Accel)
                f -= 500.
                not_move = 1
        return f, not_move

    def reward_smooth(self, a, st):
        #############################################################################
        # jerk = abs(a - self.state[2]) / self.Tau - 1.
        # f1 = 2. * (- self.tools.sigmoid(jerk, 10) + 0.5) if jerk >= 0. else 0.
        # # yaw = (st - self.av_pos['steer']) / self.Tau
        # # f2 = - 2 * abs(self.tools.sigmoid(yaw, 2) - 0.5)
        # f2 = 0.
        #############################################################################
        accel = self.Cft_Accel * a
        # fp = []
        jerk = abs(accel - self.state[1]) / self.Tau

        # for jerk in np.linspace(-5., 5., 100):
        f1 = - 0.5 * (jerk - 15.) if jerk >= 10. else 0.
        f2 = 0.
        # fp.append(f1 + f2)
        # plt.plot(np.linspace(-5., 5., 100), fp, 'r.')
        # plt.xlabel('|jerk| m/s')
        # plt.ylabel('reward')
        # plt.show()
        return f1 + f2, jerk

    def reward_cft(self, a, b):
        #############################################################################
        accel = self.Full_Accel * a - self.Full_Brake * b
        x = abs(accel) - self.Cft_Accel
        f = 4. * (- self.tools.sigmoid(x, 2.) + 0.5) if x >= 0. else 0.
        #############################################################################
        # f1 = - (abs(a) - self.Cft_Accel) if abs(a) >= self.Cft_Accel else 0.
        # f2 = - (abs(b) - self.Cft_Accel) if abs(b) >= self.Cft_Accel else 0.
        return f

    def reward_clear_old(self):
        r1, r2 = 0., 0.
        collision_l, collision_r = 0, 0
        if self.state[11] <= -2. or (self.state[11] >= 20.):
            l_dis = 0.
        elif -2. < self.state[11] <= -1.:
            l_dis = - self.state[11] - 2.
        elif -1. < self.state[11] <= 5.:
            l_dis = -1.
        else:
            l_dis = 1. / 15. * self.state[11] - 4. / 3.
        if self.state[13] <= -2. or (self.state[13] >= 20.):
            r_dis = 0.
        elif -2. < self.state[13] <= -1.:
            r_dis = - self.state[13] - 2.
        elif -1. < self.state[13] <= 5.:
            r_dis = -1.
        else:
            r_dis = 1. / 15. * self.state[13] - 4. / 3.
        if self.state[2] < 0. and (self.state[5] > 0.):
            r1 = l_dis * 50.
        elif self.state[3] < 0. and (self.state[5] > -4.):
            r2 = r_dis * 50.

        if -1. < self.state[11] <= 5. and (self.state[2] < 0. and (self.state[3] > -4.)):
            collision_l = 1
        if -1. < self.state[13] <= 5. and (self.state[3] < 0. and (self.state[5] > -4.)):
            collision_r = 1
        # f_clear_l, f_clear_r = 100., 100.
        # ft1, ft2 = 0., 0.
        # if self.state[10] >= 0.:
        #     f_clear_l = self.state[10] - 5.
        #     t_clear_l = f_clear_l / self.state[9]
        #     if f_clear_l <= 20. or t_clear_l <= 1.5:
        #         ft1 = - 20. * self.state[0] if (self.state[4] <= 0.) else 0.
        # if self.state[13] >= 0.:
        #     f_clear_r = self.state[13] - 5.
        #     t_clear_r = f_clear_r / self.state[12]
        #     if f_clear_r <= 30. or t_clear_r <= 2.0:
        #         ft2 = - 20. * self.state[0] if (self.state[4] <= 0.) else 0.
        # fl = 0.
        # fr = 0.
        # collision_l = (f_clear_l <= 0.) if (-4. <= self.state[11] <= 0.) else 0
        # collision_r = (f_clear_r <= 0.) if (-4. <= self.state[14] <= 0.) else 0
        return r1 + r2,  collision_l, collision_r

    def reward_clear(self):
        r1, r2 = 0., 0.
        t1, t2 = 0., 0.
        collision_l, collision_r = 0, 0
        crash = self.state[12:]
        crash_dis = crash[::2]
        crash_time = crash[1::2]
        for dis, t in zip(crash_dis, crash_time):
            if 0. <= dis <= Tho_dis:
                l_dis = 1. / Tho_dis * dis - 1.
            elif dis > Tho_dis or (dis < - 2.5):
                l_dis = 0.
            else:
                l_dis = 0.5 * dis - 1.

            if 0. <= t <= Tho_time:
                l_t = 1. / Tho_time * t - 1.
            elif t > Tho_time or (t < 0.):
                l_t = 0.
            else:
                l_t = - t - 1.

            if self.state[4] < 0.5 and (self.state[5] > - self.state[2]):
                r1 += l_dis * 10.
                t1 += l_t * 50.
            if l_dis <= - 1. and (self.state[4] < - 0.5 and (self.state[5] > - self.state[2])):
                collision_l += 1
        return r1 + r2 + t1,  collision_l, collision_r

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
        dis = (self.state[0]) * self.Tau / (self.Pass_Point - self.state[9]) * 1000.
        return dis

    def reward_finish(self):
        if self.state[8] <= - self.state[2]:
            return 500.
        else:
            return 0.
