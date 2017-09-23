from random import randint, random
from utilities.toolfunc import ToolFunc
import logging
import utilities.log_color

__author__ = 'qzq'

Safe_dis = 30.
Safe_time = 3.
Tho_dis = 10.
Tho_time = 2.
Visibility = 50.
Focus_No = 10


class HrlReward(object):
    Tau = 1. / 30.
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
        r_smooth, jerk = self.reward_smooth(a)
        r_clerance, collision_l, collision_r, collision_f = self.reward_clear()
        r_stop, not_stop = self.reward_stop()
        r_stop = 0.
        not_stop = 0
        r_speedlimit = self.reward_speedlimit()
        r_time = - 1.
        r_crash = - 500. if (collision_l == 1 or (collision_r == 1) or (collision_f == 1)) else 0.
        r_dis = self.reward_dis()
        r_finish = self.reward_finish()
        r_notmove, not_move = self.reward_notmove(a)
        r = r_dis + r_time + r_speedlimit + r_clerance + r_smooth + r_finish + r_crash + r_notmove + r_stop
        return r, collision_l, collision_r, collision_f, not_move, not_stop, jerk

    def reward_notmove(self, a):
        not_move = 0
        f = 0.
        accel = self.Cft_Accel * a
        if self.state[4] > 0.5 and (self.state[0] < 0.001) and (accel < 0) and (self.state[1] < 0)\
                and (self.state[14] > Safe_dis):
            f = 50. * accel
            f -= 500.
            not_move = 1
            return f, not_move

        if self.state[4] < 0.5 and (self.state[33] <= -3.) and (self.state[53] <= -3.)\
                and (self.state[0] == 0.) and (self.state[1] < 0):
            if 0.5 < accel < 0.:
                f = 50 * (accel - self.Cft_Accel)
            elif accel <= 0.5:
                f = 50 * (accel - self.Cft_Accel)
                f -= 500.
                not_move = 1
        return f, not_move

    def reward_smooth(self, a):
        accel = self.Cft_Accel * a
        jerk = abs(accel - self.state[1]) / self.Tau
        f1 = - 0.1 * (jerk - 15.) if jerk >= 15. else 0.
        f2 = 0.
        return f1 + f2, jerk

    def reward_clear(self):
        r1, r2 = 0., 0.
        t_1, t_2 = 0., 0.
        collision_l, collision_r, collision_f = 0, 0, 0

        # if self.state[4] > - 0.5:
        #     crash_f = self.state[13:15]
        #     f_dis = 1. / Tho_dis * crash_f[1] - 1. if (0. <= crash_f[1] <= Tho_dis) else 0.
        #     tt = crash_f[1] / (self.state[0] - crash_f[0]) if (self.state[0] - crash_f[0] > 0) else 20.
        #     f_t = 1. / Tho_time * tt - 1. if (0. <= tt <= Tho_time) else 0.
        #     if crash_f[1] <= 0.5:
        #         collision_f += 1
        #     r1 += f_dis * 10.
        #     t1 += f_t * 50.
        #     return r1 + r2 + t1 + t2, collision_l, collision_r, collision_f

        crash_l, crash_r = self.state[15:35], self.state[35:]
        crash_l_dis, crash_r_dis = crash_l[::2], crash_r[::2]
        crash_l_time, crash_r_time = crash_l[1::2], crash_r[1::2]
        for dis1, t1, dis2, t2 in zip(crash_l_dis, crash_l_time, crash_r_dis, crash_r_time):
            if 0. <= dis1 <= Tho_dis:
                l_dis = 1. / Tho_dis * dis1 - 1.
            elif dis1 > Tho_dis or (dis1 < - 2.5):
                l_dis = 0.
            else:
                l_dis = 0.5 * dis1 - 1
            if 0. <= dis2 <= Tho_dis:
                r_dis = 1. / Tho_dis * dis2 - 1.
            elif dis2 > Tho_dis or (dis2 < - 2.5):
                r_dis = 0.
            else:
                r_dis = 0.5 * dis2 - 1
            if 0. <= t1 <= Tho_time:
                l_t = 1. / Tho_time * t1 - 1.
            elif t1 > Tho_time or (t1 < 0.):
                l_t = 0.
            else:
                l_t = - t1 - 1
            if 0. <= t2 <= Tho_time:
                r_t = 1. / Tho_time * t2 - 1.
            elif t2 > Tho_time or (t2 < 0.):
                r_t = 0.
            else:
                r_t = - t2 - 1
            if self.state[5] < 0. and (self.state[6] > - self.state[2]):
                r2 += l_dis * 10.
                t_2 += l_t * 50.
            if self.state[6] < 0. and (self.state[8] > - self.state[2]):
                r2 += r_dis * 10.
                t_2 += r_t * 50.
            if l_dis <= - 1. and (self.state[5] < 0. and (self.state[6] > - self.state[2])):
                collision_l += 1
            if r_dis <= - 1. and (self.state[6] < 0. and (self.state[8] > - self.state[2])):
                collision_r += 1
        return r1 + r2 + t_1 + t_2,  collision_l, collision_r, collision_f

    def reward_stop(self):
        f = - 200. * self.state[0] if (self.state[4] < 0.0) else 0.
        not_stop = 1 if (self.state[4] < - 0.5 and (self.state[0] > 0.1)) else 0
        return f, not_stop

    def reward_speedlimit(self):
        fx = - 80. * (self.state[0] - self.Speed_limit) if self.state[0] >= self.Speed_limit else 0.
        if self.state[0] >= (self.Speed_limit + 2.):
            fx -= 500
        return fx

    def reward_dis(self):
        dis = (self.state[0]) * self.Tau / (self.Pass_Point - self.state[10]) * 1000.
        return dis

    def reward_finish(self):
        if self.state[9] <= - self.state[2]:
            return 1000.
        else:
            return 0.
