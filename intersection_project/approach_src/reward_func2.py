from random import randint, random
from utilities.toolfunc import ToolFunc
import logging
import utilities.log_color

__author__ = 'qzq'


class Reward(object):
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
    Vehicle_NO = 1
    Lane_Left = 0.
    Lane_Right = 4.
    Cft_Accel = 3.  # m/s**2
    Full_Accel = 5.
    Full_Brake = 5.

    def __init__(self):
        self.tools = ToolFunc()
        self.state = None

    def get_reward(self, state, a=0., b=0., st=0.):
        self.state = state
        accel = self.Full_Accel * a - self.Full_Brake * b
        r_smooth = self.reward_smooth(accel, st)
        r_clerance, collision = self.reward_clear()
        r_stop = self.reward_stop()
        r_speedlimit = self.reward_speedlimit()
        # r_v = 0.1 * self.av_pos['vy'] - 0.2 if self.av_pos['vy'] <= self.Speed_limit \
        #     else (- 0.6 * self.av_pos['vy'] + 8.4) - 0.2
        # r_v = max(- 0.2, r_v)
        r_time = - 1.
        r_crash = - 1000. if collision == 1 else 0.
        r_dis = self.reward_dis()
        r_finish = self.reward_finish()
        r_cft = self.reward_cft(a, b)
        r_notmove, not_move = self.reward_notmove(a, b)
        # logging.error('r_smooth: ' + str(r_smooth) + ', jerk: ' + str((a - self.av_pos['accel'])/self.Tau) +
        #               ', r_clearance: ' + str(r_clerance) + ', fv: [' + str(self.state_fv[1]) + ', ' +
        #               str(self.state_fv[0]) + ']'
        #               ', r_stop: ' + str(r_stop) + ', v^2/s: ' + str(self.av_pos['vy'] ** 2 / self.state_road[0]) +
        #               ', r_dis: ' + str(r_dis) + ', dis: ' + str(self.av_pos['vy'] * self.Tau) +
        #               ', r_speed: ' + str(r_speedlimit) + ', overspeed: ' + str(self.av_pos['vy']-self.Speed_limit))
        # r = r_smooth + r_clerance + r_stop + r_dis + r_finish + r_speedlimit + r_time
        r = r_dis + r_time + r_speedlimit + r_clerance + r_smooth + r_stop + r_finish + r_crash + r_cft + r_notmove

        return r, collision, not_move

    def reward_notmove(self, a, b):
        not_move = 0
        f = 0.
        accel = self.Full_Accel * a - self.Full_Brake * b
        if (self.state[0] <= 0.01) and (self.state[2] < 0) and (-1. <= accel < 0.):
            f = 100. * (self.tools.sigmoid(accel, 4.) - 0.5) if accel <= 0. else 0.
        if (self.state[0] <= 0.01) and (self.state[2] < 0) and (accel <= - 1.):
            not_move = 1
            f = - 1000.
        return f, not_move

    def reward_smooth(self, a, st):
        #############################################################################
        jerk = abs(a - self.state[2]) / self.Tau
        x1 = abs(jerk) - 1.
        f1 = 4. * (- self.tools.sigmoid(x1, 3.) + 0.5) if x1 >= 0. else 0.
        # yaw = (st - self.av_pos['steer']) / self.Tau
        # f2 = - 2 * abs(self.tools.sigmoid(yaw, 2) - 0.5)
        f2 = 0.
        #############################################################################
        # jerk = abs(a - self.state[2]) / self.Tau
        # f1 = - 0.1 * (jerk - 1.) if jerk >= 1. else 0.
        # f2 = 0.
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
        crash_dis = self.state[5] - 10.
        crash_time = crash_dis / (self.state[0] - self.state[4]) if self.state[0] - self.state[4] >= 0.01 else 100.
        x1 = crash_dis - 10.
        x2 = crash_time - 3.
        f1 = 4. * (self.tools.sigmoid(x1, 0.3) - 0.5) if x1 <= 0. else 0.
        f2 = 50. * (self.tools.sigmoid(x2, 1.) - 0.5) if x2 <= 0. else 0.
        crash_left = self.state[7]
        f3 = 0.
        crash_right = self.state[8]
        f4 = 0.
        collision = (crash_dis <= 0.1)
        #############################################################################
        # f_clear = self.state[5] - 10.
        # t_clear = self.state[5] / (self.state[0] - self.state[4]) if self.state[0] - self.state[4] >= 0.1 else 20.
        # ff = - 1. * (10. - f_clear) if f_clear <= 10. else 0.
        # ft = - 10. * (1.5 - f_clear) if t_clear <= 1.5 else 0.
        # fl = 0.
        # fr = 0.
        # collision = (f_clear <= 0.1)
        return f1 + f2 + f3 + f4,  collision

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
        x = self.state[0] - self.Speed_limit
        f = 100. * (- self.tools.sigmoid(x, 1.5) + 0.5) if x >= 0. else 0.
        if x >= 2.:
            f -= 500
        #############################################################################
        # f = - 10. * (self.state[0] - self.Speed_limit) if self.state[0] >= self.Speed_limit else 0.
        return f

    def reward_dis(self):
        dis = (self.state[0]) * self.Tau / (self.Stop_Line - self.state[9]) * 5000.
        return dis

    def reward_finish(self):
        if self.state[6] <= 2.0 and (self.state[0] <= 2.):
            return 1000.
        else:
            return 0.

