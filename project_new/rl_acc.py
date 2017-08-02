import math
import numpy as np
import matplotlib.pyplot as plt


class rl_acc(object):
    def __int__(self):
        self.Tau = 1. / 30
        self.pos = av_pos
        self.target_pos = target_pos

    def update_pos(self, a):
        t = self.Tau
        old_vy = self.pos['vy']
        self.pos['a'] = a
        self.pos['vy'] = a * t
        self.pos['y'] = old_vy * t + 0.5 * a * (t ** 2)

    def acc_target(self, av_pos, target_pos):
        pass


if __name__ == '__main__':
    av_pos = dict()
    target_pos = dict()
    acc = rl_acc()
    acc.acc_target(av_pos, target_pos)
