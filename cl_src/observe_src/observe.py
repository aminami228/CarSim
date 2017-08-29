#!/usr/bin/env python
import sys
sys.path.append('/home/scotty/qzq/git/CarSim_vires/main_project')
import logging
import numpy as np
import tensorflow as tf
import json
from observe_network.obs_actor_net import ObsActorNetword
from observe_network.obs_critic_net import ObsCriticNetwork
from observe_network.obs_replay import ObsReplay
from utilities.toolfunc import ToolFunc
from keras import backend as keras
from interface.inter_sim_hv import InterSim
import time
from rewards.reward_obs import ObsReward
import matplotlib.pyplot as plt
from random import random
import utilities.log_color

__author__ = 'qzq'

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class ReinAcc(object):
    tools = ToolFunc()

    Tau = 1. / 30
    gamma = 0.99
    epsilon = 1.0

    buffer_size = 10000
    batch_size = 32
    tau = 0.0001            # Target Network HyperParameters
    LRA = 0.001             # Learning rate for Actor
    LRC = 0.001             # Learning rate for Critic

    explore_iter = 1000000.
    # explore_iter = 1000.
    episode_count = 20000
    max_steps = 2000

    state_dim = 110
    non_his_dim = 11
    history_len = 50
    his_dim = 4
    predict_len = 50
    action_dim = 1          # Acceleration

    # Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf_sess = tf.Session(config=config)
    keras.set_session(tf_sess)

    Speed_limit = 12

    def __init__(self):
        self.sim = None
        self.history = []
        self.reward = ObsReward()
        self.total_reward = 0
        self.if_pass = False
        self.if_done = False

        self.crash = []
        self.not_stop = []
        self.success = []
        self.not_finish = []
        self.overspeed = []
        self.not_move = []
        self.loss = []

        self.sub_crash = 0
        self.sub_not_stop = 0
        self.sub_success = 0
        self.sub_not_finish = 0
        self.sub_overspeed = 0
        self.sub_not_move = 0

        self.obs_actor = ObsActorNetword(self.tf_sess, self.non_his_dim, self.history_len, self.his_dim,
                                         self.action_dim, self.batch_size, self.tau, self.LRA)
        self.obs_critic = ObsCriticNetwork(self.tf_sess, self.non_his_dim, self.history_len, self.his_dim,
                                           self.action_dim, self.batch_size, self.tau, self.LRC)
        self.buffer = ObsReplay()

        self.batch = None
        self.batch_state = None
        self.batch_his = None
        self.batch_action = None
        self.batch_reward = None
        self.batch_new_state = None
        self.batch_new_his = None
        self.batch_if_done = None
        self.batch_output = None

        self.start_time = time.time()
        self.end_time = time.time()
        self.total_time = 0.

    def load_weights(self):
        # logging.info('...... Loading weight ......')
        try:
            self.obs_actor.model.load_weights("../lstm_weights/actormodel.h5")
            self.obs_critic.model.load_weights("../lstm_weights/criticmodel.h5")
            self.obs_actor.target_model.load_weights("../lstm_weights/actormodel.h5")
            self.obs_critic.target_model.load_weights("../lstm_weights/criticmodel.h5")
            # logging.info("Weight load successfully")
        except:
            logging.warn("Cannot find the weight !")

    def update_weights(self):
        # logging.info('...... Updating weight ......')
        self.obs_actor.model.save_weights("../lstm_weights/actormodel.h5", overwrite=True)
        with open("../lstm_weights/actormodel.json", "w") as outfile:
            json.dump(self.obs_actor.model.to_json(), outfile)
        self.obs_critic.model.save_weights("../lstm_weights/criticmodel.h5", overwrite=True)
        with open("../lstm_weights/criticmodel.json", "w") as outfile:
            json.dump(self.obs_critic.model.to_json(), outfile)

    def update_batch(self, s, h, a, r, s1, h1):
        # logging.info('...... Updating batch ......')
        self.buffer.add(s, h, a, r, s1, h1, self.if_done)
        self.batch = self.buffer.get_batch(self.batch_size)
        self.batch_state = np.squeeze(np.asarray([e[0] for e in self.batch]), axis=1)
        self.batch_his = np.squeeze(np.asarray([e[1] for e in self.batch]), axis=1)
        self.batch_action = np.asarray([e[2] for e in self.batch])
        self.batch_reward = np.asarray([e[3] for e in self.batch])
        self.batch_new_state = np.squeeze(np.asarray([e[4] for e in self.batch]), axis=1)
        self.batch_new_his = np.squeeze(np.asarray([e[5] for e in self.batch]), axis=1)
        self.batch_if_done = np.asarray([e[6] for e in self.batch])
        self.batch_output = np.asarray([e[3] for e in self.batch])
        target_q_values = self.obs_critic.target_model.predict(
            [self.batch_new_state, self.batch_new_his,
             self.obs_actor.target_model.predict([self.batch_new_state, self.batch_new_his])])
        for k, done in enumerate(self.batch_if_done):
            self.batch_output[k] = self.batch_reward[k] if done else \
                self.batch_reward[k] + self.gamma * target_q_values[k]

    def update_loss(self):
        # logging.info('...... Updating loss ......')
        loss = self.obs_critic.model.train_on_batch([self.batch_state, self.batch_his, self.batch_action],
                                                    self.batch_output)
        actor_predict = self.obs_actor.model.predict([self.batch_state, self.batch_his])
        actor_grad = self.obs_critic.gradients(self.batch_state, self.batch_his, actor_predict)
        self.obs_actor.train(self.batch_state, self.batch_his, actor_grad)
        self.obs_actor.target_train()
        self.obs_critic.target_train()
        return loss

    def get_action(self, state_t, state_his, train_indicator):
        # logging.info('...... Getting action ......')
        self.epsilon -= 1.0 / self.explore_iter
        noise = []
        action_ori = self.obs_actor.model.predict([state_t, state_his])
        for i in range(self.action_dim):
            a = action_ori[0][i]
            noise.append(train_indicator * max(self.epsilon, 0) * self.tools.ou(a, -0.5, 0.5, 0.3))
        action = action_ori + np.array(noise)
        return action

    def get_hist_action(self, state_t, train_indicator):
        # logging.info('...... Getting action ......')
        self.epsilon -= 1.0 / self.explore_iter
        noise = []
        action_ori = self.obs_actor.model.predict(state_t)
        for i in range(self.action_dim):
            a = action_ori[0][i]
            noise.append(train_indicator * max(self.epsilon, 0) * self.tools.ou(a, -0.5, 0.5, 0.3))
        action = action_ori + np.array(noise)
        return action

    def if_exit(self, step, state, collision, not_move):
        if step >= self.max_steps:
            logging.warn('Not finished with max steps! Start: ' + str(self.sim.Stop_Line - state[9]) +
                         ', Dis to SL: ' + str(state[6]) + ', Dis to FL: ' + str(state[5]) +
                         ', Velocity: ' + str(state[0]))
            self.sub_not_finish += 1
            self.if_pass = False
            self.if_done = True
        elif state[0] >= self.sim.Speed_limit + 2.:
            logging.warn('Exceed Speed Limit: ' + str(self.sim.Stop_Line - state[9]) + ', Dis to SL: ' + str(state[6]) +
                         ', Dis to FL: ' + str(state[5]) + ', Velocity: ' + str(state[0]))
            self.sub_overspeed += 1
            self.if_pass = False
            self.if_done = True
        elif not_move > 0 and (state[6] >= 2.0):
            logging.warn('Not move! Start: ' + str(self.sim.Stop_Line - state[9]) + ', Dis to SL: ' + str(state[6]) +
                         ', Dis to FL: ' + str(state[5]) + ', Velocity: ' + str(state[0]))
            self.sub_not_move += 1
            self.if_pass = False
            self.if_done = True
        elif not_move > 0 and (state[6] < 2.0):
            logging.info('Congratulations! Reach stop line without crashing and has stopped. Start: ' +
                         str(self.sim.Stop_Line - state[9]) + ', Dis to SL: ' + str(state[6]) +
                         ', Dis to FL: ' + str(state[5]) + ', Velocity: ' + str(state[0]))
            self.sub_success += 1
            self.if_pass = True
            self.if_done = True
        elif collision > 0:
            logging.warn('Crash to other vehicles or road boundary! Start: ' + str(self.sim.Stop_Line - state[9]) +
                         ', Dis to SL: ' + str(state[6]) + ', Dis to FL: ' + str(state[5]) +
                         ', Velocity: ' + str(state[0]))
            self.sub_crash += 1
            self.if_pass = False
            self.if_done = True
        elif collision == 0 and (state[6] <= 1.0) and (state[0] > 5.0):
            logging.warn('No crash and reached stop line. But has not stopped! Start: ' +
                         str(self.sim.Stop_Line - state[9]) + ', Dis to SL: ' + str(state[6]) +
                         ', Dis to FL: ' + str(state[5]) + ', Velocity: ' + str(state[0]))
            self.sub_not_stop += 1
            self.if_pass = False
            self.if_done = True
        elif collision == 0 and (state[6] <= 1.0) and (state[0] <= 5.0):
            logging.info('Congratulations! Reach stop line without crashing and has stopped. Start: ' +
                         str(self.sim.Stop_Line - state[9]) + ', Dis to SL: ' + str(state[6]) +
                         ', Dis to FL: ' + str(state[5]) + ', Velocity: ' + str(state[0]))
            self.sub_success += 1
            self.if_pass = True
            self.if_done = True

    def launch_train(self, train_indicator=1):  # 1 means Train, 0 means simply Run
        # logging.info('Launch Training Process')
        # np.random.seed(1337)
        self.obs_actor = ObsActorNetword(self.tf_sess, self.non_his_dim, self.history_len, self.his_dim,
                                         self.action_dim, self.batch_size, self.tau, self.LRA)
        self.obs_critic = ObsCriticNetwork(self.tf_sess, self.non_his_dim, self.history_len, self.his_dim,
                                           self.action_dim, self.batch_size, self.tau, self.LRC)
        self.buffer = ObsReplay(self.buffer_size)
        self.load_weights()

        self.sim = InterSim()
        for e in range(self.episode_count):
            total_loss = 0.
            total_time = 0.
            # logging.debug("Episode : " + str(e) + " Replay Buffer " + str(self.buffer.count()))
            step = 0
            state_t, state_his, self.history = self.sim.get_state(self.history)
            while True:
                if len(self.history) < self.history_len:
                    self.sim.update_vehicle(0., -1.)
                    state_t, state_his, self.history = self.sim.get_state(self.history)
                else:
                    action_t = self.get_action(state_t, state_his, train_indicator)
                    reward_t, collision, not_move = self.reward.get_reward(state_t[0], state_his[0], action_t[0][0])
                    self.sim.update_vehicle(reward_t, action_t[0][0])
                    state_t1, state_his1, self.history = self.sim.get_state(self.history)
                    self.update_batch(state_t, state_his, action_t[0], reward_t, state_t1, state_his1)
                    loss = self.update_loss() if train_indicator else 0.

                    self.total_reward += reward_t
                    self.if_exit(step, state_t[0], collision, not_move)
                    step += 1
                    total_loss += loss
                    train_time = time.time() - self.start_time
                    # logging.debug('Episode: ' + str(e) + ', Step: ' + str(step) + ', Dis to SL: ' + str(state_t[0][6]) +
                    #               ', Dis to fv: ' + str(state_t[0][5]) + ', v: ' + str(state_t[0][0]) +
                    #               ', a: ' + str(action_t) + ', r: ' + str(reward_t) + ', loss: ' + str(loss) +
                    #               ', time: ' + str(train_time))
                    total_time += train_time
                    if self.if_done:
                        break
                    self.start_time = time.time()
                    state_t = state_t1

            plt.close('all')
            total_step = step + 1
            if train_indicator:
                self.update_weights()

            mean_loss = total_loss / total_step
            mean_time = total_time / total_step
            logging.debug(str(e) + "-th Episode: Steps: " + str(total_step) + ', Time: ' + str(mean_time) +
                          ', Reward: ' + str(self.total_reward) + " Loss: " + str(loss) + ', Crash: ' +
                          str(self.sub_crash) + ', Not Stop: ' + str(self.sub_not_stop) + ', Not Finished: ' +
                          str(self.sub_not_finish) + ', Overspeed: ' + str(self.sub_overspeed) + ', Not Move: ' +
                          str(self.sub_not_move) + ', Success: ' + str(self.sub_success))

            self.sim = InterSim(True) if e % 50 == 0 else InterSim()
            self.total_reward = 0
            self.if_pass = False
            self.if_done = False

            if (e + 1) % 200 == 0:
                self.crash.append(self.sub_crash)
                self.not_stop.append(self.sub_not_stop)
                self.success.append(self.sub_success)
                self.not_finish.append(self.sub_not_finish)
                self.overspeed.append(self.sub_overspeed)
                self.not_move.append(self.sub_not_move)

                self.sub_crash = 0.
                self.sub_not_stop = 0.
                self.sub_success = 0.
                self.sub_not_finish = 0.
                self.sub_overspeed = 0.
                self.sub_not_move = 0.
                logging.info('Crash: ' + str(self.crash) + ', Not Stop: ' + str(self.not_stop) + ', Not Finished: ' +
                             str(self.not_finish) + ', Overspeed: ' + str(self.overspeed) + ', Not Move: ' +
                             str(self.not_move) + ', Success: ' + str(self.success) + ', Loss: ' + str(loss))


if __name__ == '__main__':
    plt.ion()
    acc = ReinAcc()
    acc.launch_train(1)
