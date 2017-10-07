#!/usr/bin/env python
import sys
if "../" not in sys.path:               # Path to utilities and other custom modules
    sys.path.append("../")
import logging
import numpy as np
import tensorflow as tf
import json
from hrl_network.actor_net import ActorNetwork
from hrl_network.critic_net import CriticNetwork
from hrl_network.replay import Replay
from utilities.toolfunc import ToolFunc
from keras import backend as keras
from interface.inter_hrl import InterSim
import time
from rewards.reward_ch_action import CHReward
import matplotlib.pyplot as plt
from random import random
import utilities.log_color

__author__ = 'qzq'

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

Safe_dis = 30.
Safe_time = 1.5


class ReinAcc(object):
    tools = ToolFunc()
    Tau = 1. / 30.
    gamma = 0.99
    buffer_size = 10000
    batch_size = 128
    tau = 0.0001            # Target Network HyperParameters
    LRA = 0.001             # Learning rate for Actor
    LRC = 0.001             # Learning rate for Critic

    explore_iter = 100000.
    episode_count = 600000
    max_steps = 2000
    action_dim = 4          # Steering/Acceleration/Brake
    action_size = 4
    his_len = 20
    state_dim = 21

    # Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf_sess = tf.Session(config=config)
    keras.set_session(tf_sess)

    Speed_limit = 12.

    def __init__(self):
        self.epsilon = 1.
        self.hist_state = None
        self.hist_state_1 = None

        self.sim = InterSim(0, True)
        self.reward = CHReward()
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

        self.ch_actor = None
        self.ch_critic = None
        self.buffer = Replay()

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

    def load_weights(self):
        # logging.info('...... Loading weight ......')
        try:
            self.ch_actor.model.load_weights("../weights/actormodel.h5")
            self.ch_critic.model.load_weights("../weights/criticmodel.h5")
            self.ch_actor.target_model.load_weights("../weights/actormodel.h5")
            self.ch_critic.target_model.load_weights("../weights/criticmodel.h5")
            # logging.info("Weight load successfully")
        except:
            logging.warn("Cannot find the weight !")

    def update_weights(self):
        # logging.info('...... Updating weight ......')
        self.ch_actor.model.save_weights("../weights/actormodel.h5", overwrite=True)
        with open("../weights/actormodel.json", "w") as outfile:
            json.dump(self.ch_actor.model.to_json(), outfile)
        self.ch_critic.model.save_weights("../weights/criticmodel.h5", overwrite=True)
        with open("../weights/criticmodel.json", "w") as outfile:
            json.dump(self.ch_critic.model.to_json(), outfile)

    def save_weights(self, gamma, results):
        w = 'w' + str(gamma)
        self.ch_actor.model.save_weights('../' + w + '/actormodel.h5', overwrite=True)
        with open("../" + w + "/actormodel.json", "w") as outfile:
            json.dump(self.ch_actor.model.to_json(), outfile)
        self.ch_critic.model.save_weights("../" + w + "/criticmodel.h5", overwrite=True)
        with open("../" + w + "/criticmodel.json", "w") as outfile:
            json.dump(self.ch_critic.model.to_json(), outfile)
        with open('../' + w + '/g2.txt', 'w+') as json_file:
            jsoned_data = json.dumps(results)
            json_file.write(jsoned_data)

    def update_batch(self, s, a, r, s1):
        # logging.info('...... Updating batch ......')
        self.buffer.add(s, a, r, s1, self.if_done)
        self.batch = self.buffer.get_batch(self.batch_size)
        self.batch_state = np.squeeze(np.asarray([e[0] for e in self.batch]), axis=1)
        self.batch_action = np.asarray([e[1] for e in self.batch])
        self.batch_reward = np.asarray([e[2] for e in self.batch])
        self.batch_new_state = np.squeeze(np.asarray([e[3] for e in self.batch]), axis=1)
        self.batch_if_done = np.asarray([e[4] for e in self.batch])
        self.batch_output = np.asarray([e[2] for e in self.batch])
        target_q_values = self.ch_critic.target_model.predict(
            [self.batch_new_state, self.ch_actor.target_model.predict(self.batch_new_state)])
        for k, done in enumerate(self.batch_if_done):
            self.batch_output[k] = self.batch_reward[k] if done else self.batch_reward[k] + self.gamma * target_q_values[k]

    def update_loss(self):
        # logging.info('...... Updating loss ......')
        loss = self.ch_critic.model.train_on_batch([self.batch_state, self.batch_action], self.batch_output)
        actor_predict = self.ch_actor.model.predict(self.batch_state)
        actor_grad = self.ch_critic.gradients(self.batch_state, actor_predict)
        self.ch_actor.train(self.batch_state, actor_grad)
        self.ch_actor.target_train()
        self.ch_critic.target_train()
        return loss

    def get_action(self, state, train_indicator, gamma, nn):
        dis = state[0][15::2]
        dis_a = dis >= Safe_dis
        dis_b = dis < 0.
        dis_ = np.array([dis_a, dis_b])
        t = state[0][16::2]
        t_a = t >= Safe_time
        t_b = t < 0.
        t_ = np.array([t_a, t_b])
        if (np.any(dis_, axis=0).all() and np.any(t_, axis=0).all()) or (state[0][5] < 0.):
            ha = 1
        else:
            ha = -1
        # if (state[0][13] > state[0][11] - Safe_dis or (state[0][14] < state[0][12] + Safe_dis)) \
        #         and (ha == 1) and (state[0][0] ** 2 >= 2 * 3. * state[0][5]):
        #     action = np.array([-1.], ndmin=2)
        # if state[0][5] < 0.:
        #     ha = 1
        # logging.info('...... Getting action ......')
        noise = []
        zz = train_indicator * max(self.epsilon, 0.)
        action_ori = self.ch_actor.model.predict(state)
        # b = np.random.dirichlet(np.ones(2))
        b = [1.5, 0.] if (ha == 1) else [0., 1.5]
        noise.extend(list(b))
        a1 = action_ori[0][2]
        a2 = action_ori[0][3]
        if gamma == 0:
            noise.append(zz * self.tools.ou(a1, 0.8, 0.5, -0.4))  # full
            noise.append(zz * self.tools.ou(a2, 0.5, 0.5, -0.2))  # full
        elif gamma == 1:
            noise.append(zz * self.tools.ou(a1, 0.8, 0.5, -0.5))
        elif gamma == 2:
            noise.append(zz * self.tools.ou(a1, -0.8, 0.5, 0.5))
        else:
            noise.append(zz * self.tools.ou(a1, -0.2, 0.5, 0.2))
        action_h = (1. - zz) * action_ori[0][0:2] + zz * np.array(noise[0:2]) * (nn > 0.5)
        action_l = action_ori[0][2:] + np.array(noise[2:])
        action = np.array(np.concatenate([action_h, action_l], axis=0), ndmin=2)
        return action

    def if_exit(self, step, state, max_j, collision_l, collision_r, collision_f, not_move, not_stop):
        if step >= self.max_steps:
            logging.warn('Not finished with max steps! Dis to SL: {0:.2f}'.format(state[4]) +
                         ', Velocity: {0:.2f}'.format(state[0]) +
                         ', Max_j: {0:.2f}'.format(max_j) + ', ' + self.sim.cond)
            self.sub_not_finish += 1
            self.if_done = True
        elif state[0] >= self.sim.Speed_limit + 2.:
            logging.warn('Exceed Speed Limit! Dis to SL: {0:.2f}'.format(state[4]) +
                         ', Velocity: {0:.2f}'.format(state[0]) + ', Max_j: {0:.2f}'.format(max_j) +
                         ', ' + self.sim.cond)
            self.sub_overspeed += 1
            self.if_done = True
        elif not_move > 0:
            logging.warn('Not move! Dis to SL: {0:.2f}'.format(state[4]) + ', Dis to Center: {0:.2f}'.format(state[6]) +
                         ', Dis to hv: [{0:.2f}, {1:.2f}]'.format(state[-12], state[-2]) +
                         ', Velocity: {0:.2f}'.format(state[0]) + ', Max_j: {0:.2f}'.format(max_j) +
                         ', ' + self.sim.cond)
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
                         ', Dis to hv: [{0:.2f}, {1:.2f}]'.format(state[-12], state[-2]) +
                         ', Velocity: {0:.2f}'.format(state[0]) + ', Max_j: {0:.2f}'.format(max_j) +
                         ', ' + self.sim.cond)
            self.sub_crash += 1
            self.if_done = True
        elif not_stop > 0:
            logging.warn('Did not stop at stop line! Dis to SL: {0:.2f}'.format(state[4]) +
                         ', Velocity: ' + str(state[0]) + ', Max_j: {0:.2f}'.format(max_j) +
                         ', ' + self.sim.cond)
            self.sub_not_stop += 1
            self.if_done = True
        elif state[9] <= - state[2]:
            logging.info('Congratulations! Traverse successfully. ' + self.sim.cond)
            self.sub_success += 1
            self.if_done = True

    def launch_train(self, train_indicator=1):  # 1 means Train, 0 means simply Run
        # logging.info('Launch Training Process')
        gamma = 0
        state_t = self.sim.get_state()
        state_dim = state_t. shape[1]
        self.ch_actor = ActorNetwork(self.tf_sess, state_dim, self.action_size, self.batch_size,
                                     self.tau, self.LRA)
        self.ch_critic = CriticNetwork(self.tf_sess, state_dim, self.action_size, self.batch_size,
                                       self.tau, self.LRC)
        self.buffer = Replay(self.buffer_size)
        self.load_weights()

        total_time = time.time()
        fre_time = time.time()
        mean_loss = 0.
        for e in range(self.episode_count):
            total_loss = 0.
            total_reward = 0.
            step = 0
            state_t = self.sim.get_state()
            max_j = 0.
            nn = random()
            while True:
                self.epsilon -= 1.0 / self.explore_iter * train_indicator  # if e > 6000 else 0.
                action_t = self.get_action(state_t, train_indicator, gamma, nn)
                h_action = np.argmax(action_t[0][0:2])
                l_acc = action_t[0][2] if (h_action == 0) else (- action_t[0][3])
                reward_t, collision_l, collision_r, collision_f, not_move, not_stop, jerk = \
                    self.reward.get_reward(state_t[0], l_acc)
                if jerk > max_j:
                    max_j = jerk
                train_time = time.time() - fre_time
                self.sim.update_vehicle(l_acc, reward_t)
                state_t1 = self.sim.get_state()
                fre_time = time.time()
                self.update_batch(state_t, action_t[0], reward_t, state_t1)
                loss = self.update_loss() if train_indicator else 0.
                total_reward += reward_t
                self.if_exit(step, state_t[0], max_j, collision_l, collision_r, collision_f, not_move, not_stop)
                step += 1
                total_loss += loss
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
            if train_indicator:
                mean_loss = total_loss / total_step
                self.update_weights()
                self.loss.append(mean_loss)
            self.total_reward.append(total_reward)
            self.max_j.append(max_j)

            mean_time = time.time() - total_time
            logging.debug(str(e) + '-th Episode: Steps: ' + str(total_step) + ', Time: {0:.2f}'.format(mean_time) +
                          ', Reward: {0:.2f}'.format(total_reward) + ' Loss: {0:.3f}'.format(mean_loss) +
                          ', Crash: ' + str(self.sub_crash) + ', Unfinished: ' + str(self.sub_not_finish) +
                          ', Overspeed: ' + str(self.sub_overspeed) + ', Not Move: ' + str(self.sub_not_move) +
                          ', Not Stop: ' + str(self.sub_not_stop) + ', Success: ' + str(self.sub_success))
            total_time = time.time()

            visual = False     #if (e + 1) % 100 == 0 else False
            if gamma == 0 and e >= 5000:
                gamma += 1
                self.epsilon = 1.
            elif gamma == 1 and e >= 10000:
                gamma += 1
            # elif gamma >= 2 and ((e - 10000) % 10000 == 0):
            #     gamma += 1
            # gamma = min(gamma, 6)
            self.sim = InterSim(gamma, visual)
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
                             '\nSuccess: ' + str(self.success) + '\nLoss: ' + str(loss) +
                             '\nTime: ' + str(self.run_time) + '\nTest: ' + str(self.if_train))
                results = {'crash': self.crash, 'unfinished': self.not_finish, 'overspeed': self.overspeed,
                           'stop': self.not_move, 'not_stop': self.not_stop, 'succeess': self.success,
                           'loss': self.loss, 'reward': self.total_reward, 'max_j': self.max_j,
                           'time': self.run_time}
                with open('../results/ch_a4_2.txt', 'w+') as json_file:
                    jsoned_data = json.dumps(results)
                    json_file.write(jsoned_data)
                if train_indicator:
                    self.save_weights(gamma, results)
                train_indicator = 0 if train_indicator == 1 else 1


if __name__ == '__main__':
    plt.ion()
    acc = ReinAcc()
    acc.launch_train(1)
