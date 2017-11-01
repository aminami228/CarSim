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
from random import random, randint
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
    batch_size = 32
    tau = 0.0001            # Target Network HyperParameters
    LRA = 0.001             # Learning rate for Actor
    LRC = 0.001             # Learning rate for Critic

    explore_iter = 100000.
    episode_count = 600000
    max_steps = 1500
    action_dim = 1          # Steering/Acceleration/Brake
    action_size = 1
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

        self.sim = InterSim(0, False)
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
        self.dan = []

        self.sub_crash = 0
        self.sub_success = 0
        self.sub_not_finish = 0
        self.sub_overspeed = 0
        self.sub_not_move = 0
        self.sub_not_stop = 0
        self.sub_dan = 0

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
        if train_indicator:
            left = state[0][15:25]
            dis_l = left[::2]
            dis_a_l = dis_l >= Safe_dis
            dis_b_l = dis_l < 0.
            disl_ = np.array([dis_a_l, dis_b_l])
            t_l = left[1::2]
            t_a_l = t_l >= Safe_time
            t_b_l = t_l < 0.
            tl_ = np.array([t_a_l, t_b_l])
            right = state[0][25:]
            dis_r = right[::2]
            dis_a_r = dis_r >= Safe_dis
            dis_b_r = dis_r < 0.
            disr_ = np.array([dis_a_r, dis_b_r])
            t_r = right[1::2]
            t_a_r = t_r >= Safe_time
            t_b_r = t_r < 0.
            tr_ = np.array([t_a_r, t_b_r])
            if np.any(disl_, axis=0).all() and np.any(tl_, axis=0).all():
                ha = 1
            elif state[0][5] < 0.:
                ha = 1
                if (not (np.any(disr_, axis=0).all() and np.any(tr_, axis=0).all())) and (state[0][6] >= 0.):
                    ha = -1
            else:
                ha = -1
            # logging.info('...... Getting action ......')
            noise = []
            zz = train_indicator * max(self.epsilon, 0.)
            action_ori = self.ch_actor.model.predict(state)
            # b = np.random.dirichlet(np.ones(2))
            # b = [1.5, 0.] if (ha == 1) else [0., 1.5]
            a1 = action_ori[0][0]
            # a2 = action_ori[0][2]
            if nn < 0.8 * zz:
                # b = [1., 0.] if (ha == 1) else [0., 1.]
                # noise.extend(list(b))
                if ha == 1:
                    noise.append(self.tools.ou(a1, 1., 0.8, -0.5))  # full
                #     # noise.extend(list(b))
                #     # noise.append(zz * self.tools.ou(a1, 1., 0.5, -0.4))  # full
                #     # noise.append(zz * self.tools.ou(a2, 0., 0.5, 0.2))  # full
                else:
                    noise.append(self.tools.ou(a1, - 1., 0.5, 0.4))  # full
                #     b = [-1.]
                #     noise.extend(list(b))
                #     noise.append(zz * self.tools.ou(a1, 0., 0.5, 0.2))  # full
                #     noise.append(zz * self.tools.ou(a2, 1., 0.5, -0.4))  # full
            else:
                mu = 2. * random() - 1.
                noise.append(zz * self.tools.ou(a1, mu, 0.8, 0.2))  # full
                # b = np.random.dirichlet(np.ones(2))
                # noise.extend(list(b))
                # noise.append(zz * self.tools.ou(a1, 0.8, 0.5, -0.4))  # full
                # noise.append(zz * self.tools.ou(a2, 0.8, 0.5, -0.4))  # full
            # action_h = np.array(action_ori[0][0] + zz * np.array(noise[0]), ndmin=1)
            action_l = action_ori[0] + np.array(noise)
            # action = np.array(np.concatenate([action_h, action_l], axis=0), ndmin=2)
            # if random() < zz:
            #     action = np.array(np.array(noise), ndmin=2)
            # else:
            #     action = np.array(action_ori, ndmin=2)
            action = np.array(action_l, ndmin=2)
        else:
            action = self.ch_actor.model.predict(state)
        return action

    def if_exit(self, step, state, max_j, collision_l, collision_r, collision_f, not_move, not_stop, dan):
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
        elif dan > 0:
            logging.warn('Uncomfortable accel! Dis to SL: {0:.2f}'.format(state[4]) +
                         ', Velocity: ' + str(state[0]) + ', Max_j: {0:.2f}'.format(max_j) +
                         ', ' + self.sim.cond)
            self.sub_dan += 1
            self.if_done = True
        elif state[9] <= - state[2]:
            logging.info('Congratulations! Traverse successfully. ' + self.sim.cond)
            self.sub_success += 1
            self.if_done = True

    def launch_train(self, train_indicator=1):  # 1 means Train, 0 means simply Run
        # logging.info('Launch Training Process')
        gamma = 0
        ep = False
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
                # h_action = np.argmax(action_t[0][0:2])
                # l_acc = action_t[0][2] if (h_action == 0) else (- action_t[0][3])
                # l_acc = action_t[0][1] if (action_t[0][0] >= 0.) else (- action_t[0][2])
                # if action_t[0][0] >= 0.5:
                #     l_acc = action_t[0][1] - action_t[0][2]
                # elif action_t[0][0] <= -0.5:
                #     l_acc = - action_t[0][2] + action_t[0][1]
                # else:
                #     l_acc = - action_t[0][2] + action_t[0][1]
                # if action_t[0][0] > action_t[0][1]:
                #     lacc = 1.
                # else:
                #     lacc = -1.
                lacc = action_t[0][0]
                reward_t, collision_l, collision_r, collision_f, not_move, not_stop, jerk, dan = \
                    self.reward.get_reward(state_t[0], lacc)
                if jerk > max_j:
                    max_j = jerk
                train_time = time.time() - fre_time
                self.sim.update_vehicle(lacc, reward_t)
                state_t1 = self.sim.get_state()
                fre_time = time.time()
                if train_indicator:
                    self.update_batch(state_t, action_t[0], reward_t, state_t1)
                loss = self.update_loss() if train_indicator else 0.
                total_reward += reward_t
                self.if_exit(step, state_t[0], max_j, collision_l, collision_r, collision_f, not_move, not_stop, dan)
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
                          ', Not Stop: ' + str(self.sub_not_stop) + ', Uncomfort: ' + str(self.sub_dan) +
                          ', Success: ' + str(self.sub_success))
            total_time = time.time()

            visual = False if (e + 1) % 500 == 0 else False
            # if gamma == 0 and e >= 5000:
            #     gamma = 1
            #     self.epsilon = 1.
            # elif e >= 7000:
            #     gamma = randint(0, 2)
            #     # self.epsilon = 1.
            # gamma = 2

            if (e + 1) % 100 == 0:
                self.if_train.append(train_indicator)
                self.crash.append(self.sub_crash)
                self.success.append(self.sub_success)
                self.not_finish.append(self.sub_not_finish)
                self.overspeed.append(self.sub_overspeed)
                self.not_move.append(self.sub_not_move)
                self.not_stop.append(self.sub_not_stop)
                self.dan.append(self.sub_dan)
                self.run_time.append((time.time() - self.total_time) / 60.)

                self.sub_crash = 0
                self.sub_success = 0
                self.sub_not_finish = 0
                self.sub_overspeed = 0
                self.sub_not_move = 0
                self.sub_not_stop = 0
                self.sub_dan = 0
                logging.info('Crash: ' + str(self.crash) + '\nNot Finished: ' + str(self.not_finish) +
                             '\nOverspeed: ' + str(self.overspeed) + '\nNot Move: ' + str(self.not_move) +
                             '\nNot Stop: ' + str(self.not_stop) + '\nUncomfort: ' + str(self.dan) +
                             '\nSuccess: ' + str(self.success) + '\nLoss: ' + str(loss) +
                             '\nTime: ' + str(self.run_time) + '\nTest: ' + str(self.if_train))
                results = {'crash': self.crash, 'unfinished': self.not_finish, 'overspeed': self.overspeed,
                           'stop': self.not_move, 'not_stop': self.not_stop, 'uncomfort': self.dan,
                           'succeess': self.success,
                           'loss': self.loss, 'reward': self.total_reward, 'max_j': self.max_j,
                           'time': self.run_time}
                # with open('../results/ch_rule_g1_5.txt', 'w+') as json_file:
                with open('../results/new2_ha.txt', 'w+') as json_file:
                    jsoned_data = json.dumps(results)
                    json_file.write(jsoned_data)
                if train_indicator:
                    self.save_weights(gamma, results)
                train_indicator = 0 if train_indicator == 1 else 1

            if (e >= 1000 and (np.mean(self.success[-9::2]) >= 90.)) or ep:
                ep = True
                self.epsilon = 1.
                gamma = 0 if random() > (e / 10000) else 2
                if train_indicator:
                    self.sim = InterSim(gamma, visual)
                else:
                    self.sim = InterSim(3, visual)
            else:
                self.sim = InterSim(0, visual)
            self.if_done = False


if __name__ == '__main__':
    plt.ion()
    acc = ReinAcc()
    acc.launch_train(1)
