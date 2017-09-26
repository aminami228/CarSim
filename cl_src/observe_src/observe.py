#!/usr/bin/env python
import sys
if "../" not in sys.path:               # Path to utilities and other custom modules
    sys.path.append("../")
import logging
import numpy as np
import tensorflow as tf
import json
from observe_network.obs_actor_net import ObsActorNetwork
from observe_network.obs_critic_net import ObsCriticNetwork
from observe_network.obs_replay import ObsReplay
from utilities.toolfunc import ToolFunc
from keras import backend as keras
from interface.inter_new import InterSim
import time
from rewards.reward_vm import ObsReward
import matplotlib.pyplot as plt
from random import random
import utilities.log_color

__author__ = 'qzq'

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class ReinAcc(object):
    tools = ToolFunc()

    Tau = 1. / 10.
    gamma = 0.99

    buffer_size = 10000
    batch_size = 128
    tau = 0.0001            # Target Network HyperParameters
    LRA = 0.001             # Learning rate for Actor
    LRC = 0.001             # Learning rate for Critic

    explore_iter = 1000000.
    # explore_iter = 1000.
    episode_count = 6000000
    max_steps = 800
    action_dim = 1          # Steering/Acceleration/Brake
    action_size = 1
    his_len = 20
    state_dim = 21

    # Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf_sess = tf.Session(config=config)
    keras.set_session(tf_sess)

    Speed_limit = 12

    def __init__(self):
        self.epsilon = 1.
        self.hist_state = None
        self.hist_state_1 = None

        self.sim = InterSim(0, True)
        self.reward = ObsReward()
        self.if_pass = False
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

        self.sub_crash = 0
        self.sub_success = 0
        self.sub_not_finish = 0
        self.sub_overspeed = 0
        self.sub_not_move = 0

        self.obs_actor = None
        self.obs_critic = None
        self.buffer = ObsReplay()

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
            self.obs_actor.model.load_weights("../weights/actormodel.h5")
            self.obs_critic.model.load_weights("../weights/criticmodel.h5")
            self.obs_actor.target_model.load_weights("../weights/actormodel.h5")
            self.obs_critic.target_model.load_weights("../weights/criticmodel.h5")
            # logging.info("Weight load successfully")
        except:
            logging.warn("Cannot find the weight !")

    def update_weights(self):
        # logging.info('...... Updating weight ......')
        self.obs_actor.model.save_weights("../weights/actormodel.h5", overwrite=True)
        with open("../weights/actormodel.json", "w") as outfile:
            json.dump(self.obs_actor.model.to_json(), outfile)
        self.obs_critic.model.save_weights("../weights/criticmodel.h5", overwrite=True)
        with open("../weights/criticmodel.json", "w") as outfile:
            json.dump(self.obs_critic.model.to_json(), outfile)

    def save_weights(self, gamma, results):
        w = 'w' + str(gamma)
        self.obs_actor.model.save_weights('../' + w + '/actormodel.h5', overwrite=True)
        with open("../" + w + "/actormodel.json", "w") as outfile:
            json.dump(self.obs_actor.model.to_json(), outfile)
        self.obs_critic.model.save_weights("../" + w + "/criticmodel.h5", overwrite=True)
        with open("../" + w + "/criticmodel.json", "w") as outfile:
            json.dump(self.obs_critic.model.to_json(), outfile)
        with open('../' + w + '/g2.txt', 'w+') as json_file:
            jsoned_data = json.dumps(results)
            json_file.write(jsoned_data)

    def update_batch(self, s, a, r, s1):
        # logging.info('...... Updating batch ......')
        # s = np.reshape(s, (1, self.his_len, self.state_dim))
        # s1 = np.reshape(s1, (1, self.his_len, self.state_dim))
        self.buffer.add(s, a, r, s1, self.if_done)
        self.batch = self.buffer.get_batch(self.batch_size)
        self.batch_state = np.squeeze(np.asarray([e[0] for e in self.batch]), axis=1)
        self.batch_action = np.asarray([e[1] for e in self.batch])
        self.batch_reward = np.asarray([e[2] for e in self.batch])
        self.batch_new_state = np.squeeze(np.asarray([e[3] for e in self.batch]), axis=1)
        self.batch_if_done = np.asarray([e[4] for e in self.batch])
        self.batch_output = np.asarray([e[2] for e in self.batch])
        target_q_values = self.obs_critic.target_model.predict(
            [self.batch_new_state, self.obs_actor.target_model.predict(self.batch_new_state)])
        for k, done in enumerate(self.batch_if_done):
            self.batch_output[k] = self.batch_reward[k] if done else self.batch_reward[k] + self.gamma * target_q_values[k]

    def update_loss(self):
        # logging.info('...... Updating loss ......')
        loss = self.obs_critic.model.train_on_batch([self.batch_state, self.batch_action], self.batch_output)
        actor_predict = self.obs_actor.model.predict(self.batch_state)
        actor_grad = self.obs_critic.gradients(self.batch_state, actor_predict)
        self.obs_actor.train(self.batch_state, actor_grad)
        self.obs_actor.target_train()
        self.obs_critic.target_train()
        return loss

    def get_action(self, state, train_indicator, gamma):
        # logging.info('...... Getting action ......')
        noise = []
        # hist_state = np.reshape(self.hist_state, (1, self.his_len, self.state_dim))
        action_ori = self.obs_actor.model.predict(state)
        for i in range(self.action_size):
            a = action_ori[0][i]
            if gamma == 0:
                noise.append(train_indicator * max(self.epsilon, 0) * self.tools.ou(a, 0.8, 0.5, -0.5))  # full
            elif gamma == 1:
                noise.append(train_indicator * max(self.epsilon, 0) * self.tools.ou(a, -0.8, 0.5, 0.5))
            else:
                noise.append(train_indicator * max(self.epsilon, 0) * self.tools.ou(a, -0.4, 0.5, 0.5))
            # noise.append(train_indicator * max(self.epsilon, 0) * self.tools.ou(a, -0.4, 0.5, 0.3))   # 4v [-2, 2]
            # noise.append(train_indicator * max(self.epsilon, 0) * self.tools.ou(a, -0.2, 0.5, 0.2))  # 2v [-1, 2]
            # noise.append(train_indicator * max(self.epsilon, 0) * self.tools.ou(a, 0.1, 0.2, 0.2))   # 3v [-10, -5]
            # noise.append(train_indicator * max(self.epsilon, 0) * self.tools.ou(a, -0.5, 0.5, 0.3))
            # noise.append(train_indicator * max(self.epsilon, 0) * self.tools.ou(a, -0.4, 0.5, 0.2))
            # noise.append(train_indicator * max(self.epsilon, 0) * self.tools.ou(a, 0.4, 0.5, -0.3)) # tra
        action = action_ori + np.array(noise)
        return action

    def if_exit(self, step, state, max_a, collision_l, collision_r, not_move):
        if step >= self.max_steps:
            logging.warn('Not finished with max steps! Start: {0:.2f}'.format(state[4]) +
                         ', Dis to Center: {0:.2f}'.format(state[5]) +
                         ', Dis to hv: {0:.2f}'.format(state[12]) +
                         ', Velocity: {0:.2f}'.format(state[0]) + ', Max_a: {0:.2f}'.format(max_a) +
                         ', ' + self.sim.cond + ', Lv_ini: {0:.2f}'.format(self.sim.lv_ini))
            self.sub_not_finish += 1
            self.if_pass = False
            self.if_done = True
        elif state[0] >= self.sim.Speed_limit + 2.:
            logging.warn('Exceed Speed Limit! Start: {0:.2f}'.format(state[4]) +
                         ', Dis to Center: {0:.2f}'.format(state[5]) +
                         ', Dis to hv: {0:.2f}'.format(state[12]) +
                         ', Velocity: {0:.2f}'.format(state[0]) + ', Max_a: {0:.2f}'.format(max_a) +
                         ', ' + self.sim.cond + ', Lv_ini: {0:.2f}'.format(self.sim.lv_ini))
            self.sub_overspeed += 1
            self.if_pass = False
            self.if_done = True
        elif not_move > 0:
            logging.warn('Not move! Start: {0:.2f}'.format(state[4]) + ', Dis to Center: {0:.2f}'.format(state[5]) +
                         ', Dis to hv: {0:.2f}'.format(state[12]) +
                         ', Velocity: {0:.2f}'.format(state[0]) + ', Max_a: {0:.2f}'.format(max_a) +
                         ', ' + self.sim.cond + ', Lv_ini: {0:.2f}'.format(self.sim.lv_ini))
            self.sub_not_move += 1
            self.if_pass = False
            self.if_done = True
        elif collision_l > 0 or (collision_r > 0):
            v = 'left' if collision_l > 0 else 'right'
            logging.warn('Crash to ' + v + ' vehicles! Start: {0:.2f}'.format(state[4]) +
                         ', Dis to Center: {0:.2f}'.format(state[5]) +
                         ', Dis to hv: {0:.2f}'.format(state[12]) +
                         ', Velocity: {0:.2f}'.format(state[0]) + ', Max_a: {0:.2f}'.format(max_a) +
                         ', ' + self.sim.cond + ', Lv_ini: {0:.2f}'.format(self.sim.lv_ini))
            self.sub_crash += 1
            self.if_pass = False
            self.if_done = True
        elif state[8] <= - state[2]:

            logging.info('Congratulations! Traverse successfully. ' + self.sim.cond + ' condion.')
            self.sub_success += 1
            self.if_pass = True
            self.if_done = True

    def launch_train(self, train_indicator=1):  # 1 means Train, 0 means simply Run
        gamma = 0
        # logging.info('Launch Training Process')
        state_t = self.sim.get_state()
        state_dim = state_t.shape[1]
        self.obs_actor = ObsActorNetwork(self.tf_sess, state_dim, self.action_size, self.batch_size,
                                         self.tau, self.LRA)
        self.obs_critic = ObsCriticNetwork(self.tf_sess, state_dim, self.action_size, self.batch_size,
                                           self.tau, self.LRC)
        self.buffer = ObsReplay(self.buffer_size)
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
            while True:
                self.epsilon -= 1.0 / self.explore_iter * train_indicator # if e > 6000 else 0.
                action_t = self.get_action(state_t, train_indicator, gamma)
                reward_t, collision_l, collision_r, not_move, jerk = self.reward.get_reward(state_t[0], action_t[0][0])
                if jerk > max_j:
                    max_j = jerk
                train_time = time.time() - fre_time
                self.sim.update_vehicle(action_t[0][0], reward_t)
                state_t1 = self.sim.get_state()
                fre_time = time.time()
                self.update_batch(state_t, action_t[0], reward_t, state_t1)
                loss = self.update_loss() if train_indicator else 0.

                total_reward += reward_t
                self.if_exit(step, state_t[0], max_j, collision_l, collision_r, not_move)
                step += 1
                total_loss += loss
                train_time += time.time() - fre_time
                # if e % 50 == 0:
                # logging.debug('Episode: ' + str(e) + ', Step: ' + str(step) +
                #               ', Dis to Center: {0:.2f}'.format(state_t[0][5]) +
                #               ', Dis to hv: {0:.2f}'.format(state_t[0][12]) +
                #               ', v: {0:.2f}'.format(state_t[0][0]) + ', a: {0:.2f}'.format(action_t[0][0]) +
                #               ', r: {0:.2f}'.format(reward_t) + ', loss: {0:.3f}'.format(loss) +
                #               ', time: {0:.2f}'.format(train_time))
                # fre_time = time.time()
                if self.if_done:
                    break
                self.start_time = time.time()
                state_t = state_t1
                # self.hist_state = self.hist_state_1

            plt.close('all')
            total_step = step + 1
            if train_indicator:
                mean_loss = total_loss / total_step
                self.update_weights()
                self.loss.append(mean_loss)
            self.total_reward.append(total_reward)
            self.max_j.append(max_j)

            # mean_time = total_time / total_step
            mean_time = time.time() - total_time
            logging.debug(str(e) + '-th Episode: Steps: ' + str(total_step) + ', Time: {0:.2f}'.format(mean_time) +
                          ', Reward: {0:.2f}'.format(total_reward) + ' Loss: {0:.3f}'.format(mean_loss) +
                          ', Crash: ' + str(self.sub_crash) + ', Not Finished: ' + str(self.sub_not_finish) +
                          ', Overspeed: ' + str(self.sub_overspeed) + ', Not Move: ' + str(self.sub_not_move) +
                          ', Success: ' + str(self.sub_success))
            total_time = time.time()

            visual = True if (e + 1) % 1000 == 0 else False
            if gamma == 0 and e >= 2000:
                gamma += 1
            elif gamma == 1 and e >= 10000:
                gamma += 1
            elif gamma >= 2 and ((e - 10000) % 10000 == 0):
                gamma += 1
            gamma = min(gamma, 5)
            self.sim = InterSim(gamma, visual)
            # self.sim = InterSim(False)
            # self.sim.draw_scenary(self.sim.av_pos, self.sim.hv_poses, 0.)
            self.if_pass = False
            self.if_done = False
            self.hist_state = None
            self.hist_state_1 = None

            if (e + 1) % 100 == 0:
                self.if_train.append(train_indicator)
                self.crash.append(self.sub_crash)
                self.success.append(self.sub_success)
                self.not_finish.append(self.sub_not_finish)
                self.overspeed.append(self.sub_overspeed)
                self.not_move.append(self.sub_not_move)
                self.run_time.append((time.time() - self.total_time) / 60.)

                self.sub_crash = 0
                self.sub_success = 0
                self.sub_not_finish = 0
                self.sub_overspeed = 0
                self.sub_not_move = 0
                logging.info('Crash: ' + str(self.crash) + '\nNot Finished: ' + str(self.not_finish) +
                             '\nOverspeed: ' + str(self.overspeed) + '\nNot Move: ' + str(self.not_move) +
                             '\nSuccess: ' + str(self.success) + '\nLoss: ' + str(loss) +
                             '\nTime: ' + str(self.run_time) + '\nTest: ' + str(self.if_train))
                results = {'crash': self.crash, 'unfinished': self.not_finish, 'overspeed': self.overspeed,
                           'stop': self.not_move, 'succeess': self.success,
                           'loss': self.loss, 'reward': self.total_reward, 'max_j': self.max_j,
                           'time': self.run_time}
                with open('../results/cl_tra4.txt', 'w+') as json_file:
                    jsoned_data = json.dumps(results)
                    json_file.write(jsoned_data)

                train_indicator = 0 if train_indicator == 1 else 1
                self.save_weights(gamma, results)
                # if len(self.success) % 2 == 0 and (np.mean(self.success[-21::2]) == 100.) \
                #         and (len(self.success) - empty > 20):
                #     self.save_weights(gamma, results)
                #     break
                #     empty = e
                #     gamma += 1
                #     if gamma > 2:
                #         break
                #     train_indicator = 1
                #     self.epsilon = 1.
                    # if (e + 1) % 1000 == 0:
                    #     self.epsilon = 1.0


if __name__ == '__main__':
    plt.ion()
    acc = ReinAcc()
    acc.launch_train(1)
