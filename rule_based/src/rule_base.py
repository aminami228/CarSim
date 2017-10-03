#!/usr/bin/env python
import sys
if "../" not in sys.path:               # Path to utilities and other custom modules
    sys.path.append("../")
import logging
import numpy as np
import json
from utilities.toolfunc import ToolFunc
from interface.inter_rule import InterSim
import time
from rewards.reward_rule import HrlReward
import matplotlib.pyplot as plt
from random import random
import utilities.log_color

__author__ = 'qzq'

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

Vis_dis = 50.
Safe_dis = 30.
Safe_time = 3.
Tho_dis = 10.
Tho_time = 2.


class ReinAcc(object):
    tools = ToolFunc()
    Tau = 1. / 30.
    gamma = 0.99
    buffer_size = 10000
    batch_size = 128
    tau = 0.0001            # Target Network HyperParameters
    LRA = 0.001             # Learning rate for Actor
    LRC = 0.001             # Learning rate for Critic

    explore_iter = 1000000.
    episode_count = 6000000
    max_steps = 2000
    action_dim = 3          # Steering/Acceleration/Brake
    action_size = 3
    his_len = 20
    state_dim = 21

    Speed_limit = 12.

    def __init__(self):
        self.epsilon = 1.
        self.hist_state = None
        self.hist_state_1 = None

        self.sim = InterSim(0, True)
        self.reward = HrlReward()
        self.if_done = False

        self.total_reward = []
        self.loss = []
        self.run_time = []
        self.max_j = []

        self.crash = []
        self.success = []
        self.not_finish = []
        self.overspeed = []
        self.not_move = []
        self.if_train = []
        self.not_stop = []

        self.sub_crash = 0
        self.sub_success = 0
        self.sub_not_finish = 0
        self.sub_overspeed = 0
        self.sub_not_move = 0
        self.sub_not_stop = 0

        self.batch = None
        self.batch_state = None
        self.batch_action = None
        self.batch_reward = None
        self.batch_new_state = None
        self.batch_if_done = None
        self.batch_output = None

        self.start_time = time.time()
        self.end_time = time.time()
        self.total_time = time.time()

    @staticmethod
    def get_action(state):
        # logging.info('...... Getting action ......')
        # # rule 1 ##
        # dis = state[0][15::2]
        # dis_a = dis >= Safe_dis
        # dis_b = dis < 0.
        # dis_ = np.array([dis_a, dis_b])
        # t = state[0][16::2]
        # t_a = t >= Safe_time
        # t_b = t < 0.
        # t_ = np.array([t_a, t_b])
        # if np.any(dis_, axis=0).all() and np.any(t_, axis=0).all():
        #     action = np.array([1.], ndmin=2)
        # else:
        #     action = np.array([-1.], ndmin=2)
        # # rule 2 ##
        # dis = state[0][15::2]
        # dis_a = dis >= Safe_dis
        # dis_b = dis < 0.
        # dis_ = np.array([dis_a, dis_b])
        # t = state[0][16::2]
        # t_a = t >= Safe_time
        # t_b = t < 0.
        # t_ = np.array([t_a, t_b])
        # if np.any(dis_, axis=0).all() and np.any(t_, axis=0).all():
        #     action = np.array([1.], ndmin=2)
        # else:
        #     action = np.array([-1.], ndmin=2)
        # if (state[0][13] > state[0][11] - Safe_dis or (state[0][14] < state[0][12] + Safe_dis)) \
        #         and (action[0][0] == 1.) and (state[0][0] ** 2 >= 2 * 3. * state[0][5]):
        #     action = np.array([-1.], ndmin=2)
        # # rule 3 ##
        action = np.array([1.], ndmin=2)
        return action

    def if_exit(self, step, state, max_j, collision_l, collision_r, collision_f, not_move, not_stop):
        if step >= self.max_steps:
            logging.warn('Not finished with max steps! Dis to SL: {0:.2f}'.format(state[4]) +
                         ', Dis to fv: {0:.2f}'.format(state[14]) + ', Velocity: {0:.2f}'.format(state[0]) +
                         ', Max_j: {0:.2f}'.format(max_j))
            self.sub_not_finish += 1
            self.if_done = True
        elif state[0] >= self.sim.Speed_limit + 2.:
            logging.warn('Exceed Speed Limit! Dis to SL: {0:.2f}'.format(state[4]) +
                         ', Dis to fv: {0:.2f}'.format(state[14]) +
                         ', Velocity: {0:.2f}'.format(state[0]) + ', Max_j: {0:.2f}'.format(max_j))
            self.sub_overspeed += 1
            self.if_done = True
        elif not_move > 0:
            logging.warn('Not move! Dis to SL: {0:.2f}'.format(state[4]) + ', Dis to Center: {0:.2f}'.format(state[6]) +
                         ', Dis to fv: {0:.2f}'.format(state[14]) +
                         ', Dis to hv: [{0:.2f}, {1:.2f}]'.format(state[23], state[33]) +
                         ', Velocity: {0:.2f}'.format(state[0]) + ', Max_j: {0:.2f}'.format(max_j))
            self.sub_not_move += 1
            self.if_done = True
        elif collision_f > 0 or (collision_l > 0) or (collision_r > 0):
            if collision_l > 0:
                v = 'left'
            elif collision_r > 0:
                v = 'right'
            else:
                v = 'front'
            logging.warn('Crash to ' + v + ' vehicles! Dis to SL: {0:.2f}'.format(state[4]) +
                         ', Dis to Center: {0:.2f}'.format(state[6]) +
                         ', Dis to fv: {0:.2f}'.format(state[14]) +
                         ', Dis to hv: [{0:.2f}, {1:.2f}]'.format(state[23], state[33]) +
                         ', Velocity: {0:.2f}'.format(state[0]) + ', Max_j: {0:.2f}'.format(max_j))
            self.sub_crash += 1
            self.if_done = True
        elif not_stop > 0:
            logging.warn('Did not stop at stop line! Dis to SL: {0:.2f}'.format(state[4]) +
                         ', Velocity: ' + str(state[0]) + ', Max_j: {0:.2f}'.format(max_j))
            self.sub_not_stop += 1
            self.if_done = True
        elif state[8] <= - state[2]:
            logging.info('Congratulations! Traverse successfully. ')
            self.sub_success += 1
            self.if_done = True

    def launch_train(self, train_indicator=1):  # 1 means Train, 0 means simply Run
        # logging.info('Launch Training Process')
        state_t = self.sim.get_state()
        state_dim = state_t. shape[1]

        total_time = time.time()
        fre_time = time.time()
        mean_loss = 0.
        for e in range(self.episode_count):
            total_loss = 0.
            total_reward = 0.
            step = 0
            state_t = self.sim.get_state()
            max_j = 0.
            while True:
                action_t = self.get_action(state_t)
                reward_t, collision_l, collision_r, collision_f, not_move, not_stop, jerk = \
                    self.reward.get_reward(state_t[0], action_t[0][0])
                if jerk > max_j:
                    max_j = jerk
                train_time = time.time() - fre_time
                self.sim.update_vehicle(action_t[0][0], reward_t)
                state_t1 = self.sim.get_state()
                fre_time = time.time()
                total_reward += reward_t
                self.if_exit(step, state_t[0], max_j, collision_l, collision_r, collision_f, not_move, not_stop)
                step += 1
                train_time += time.time() - fre_time
                # logging.debug('Episode: ' + str(e) + ', Step: ' + str(step) +
                #               ', Dis to Center: {0:.2f}'.format(state_t[0][5]) +
                #               ', Dis to hv: {0:.2f}'.format(state_t[0][12]) +
                #               ', v: {0:.2f}'.format(state_t[0][0]) + ', a: {0:.2f}'.format(action_t[0][0]) +
                #               ', r: {0:.2f}'.format(reward_t) + ', loss: {0:.3f}'.format(loss) +
                #               ', time: {0:.2f}'.format(train_time))
                fre_time = time.time()
                if self.if_done:
                    break
                self.start_time = time.time()
                state_t = state_t1

            plt.close('all')
            total_step = step + 1
            self.total_reward.append(total_reward)
            self.max_j.append(max_j)

            mean_time = time.time() - total_time
            logging.debug(str(e) + '-th Episode: Steps: ' + str(total_step) + ', Time: {0:.2f}'.format(mean_time) +
                          ', Reward: {0:.2f}'.format(total_reward) +
                          ', Crash: ' + str(self.sub_crash) + ', Unfinished: ' + str(self.sub_not_finish) +
                          ', Overspeed: ' + str(self.sub_overspeed) + ', Not Move: ' + str(self.sub_not_move) +
                          ', Not Stop: ' + str(self.sub_not_stop) + ', Success: ' + str(self.sub_success))
            total_time = time.time()

            visual = True            # True if (e + 1) % 200 == 0 else False
            # if gamma == 0 and e >= 2000:
            #     gamma += 1
            # elif gamma == 1 and e >= 10000:
            #     gamma += 1
            # elif gamma >= 2 and ((e - 10000) % 10000 == 0):
            #     gamma += 1
            # gamma = min(gamma, 6)
            self.sim = InterSim(0, visual)
            self.if_done = False

            if (e + 1) % 100 == 0:
                self.if_train.append(train_indicator)
                self.crash.append(self.sub_crash)
                self.success.append(self.sub_success)
                self.not_finish.append(self.sub_not_finish)
                self.overspeed.append(self.sub_overspeed)
                self.not_move.append(self.sub_not_move)
                self.not_stop.append(self.sub_not_stop)
                self.run_time.append((time.time() - self.total_time) / 60.)

                self.sub_crash = 0
                self.sub_success = 0
                self.sub_not_finish = 0
                self.sub_overspeed = 0
                self.sub_not_move = 0
                self.sub_not_stop = 0
                logging.info('Crash: ' + str(self.crash) + '\nNot Finished: ' + str(self.not_finish) +
                             '\nOverspeed: ' + str(self.overspeed) + '\nNot Move: ' + str(self.not_move) +
                             '\nNot Stop: ' + str(self.not_stop) +
                             '\nSuccess: ' + str(self.success) +
                             '\nTime: ' + str(self.run_time) + '\nTest: ' + str(self.if_train))
                results = {'crash': self.crash, 'unfinished': self.not_finish, 'overspeed': self.overspeed,
                           'stop': self.not_move, 'not_stop': self.not_stop, 'succeess': self.success,
                           'reward': self.total_reward, 'max_j': self.max_j,
                           'time': self.run_time}
                with open('../results/rule1.txt', 'w+') as json_file:
                    jsoned_data = json.dumps(results)
                    json_file.write(jsoned_data)


if __name__ == '__main__':
    plt.ion()
    acc = ReinAcc()
    acc.launch_train(1)
