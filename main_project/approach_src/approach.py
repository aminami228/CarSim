#!/usr/bin/env python
import sys
sys.path.append('/home/scotty/qzq/git/CarSim_vires/main_project')
import logging
import numpy as np
import tensorflow as tf
import json
from approach_network.app_actor_net import AppActorNetwork
from approach_network.app_critic_net import AppCriticNetwork
from approach_network.app_replay import AppReplay
from utilities.toolfunc import ToolFunc
from keras import backend as keras
from interface.inter_sim import InterSim
import time
from random import random
from reward_app import AppReward
import matplotlib.pyplot as plt
import utilities.log_color

__author__ = 'qzq'

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class ReinAcc(object):
    tools = ToolFunc()

    Tau = 1. / 30
    gamma = 0.99
    epsilon = 0.5

    buffer_size = 10000
    batch_size = 128
    tau = 0.0001            # Target Network HyperParameters
    LRA = 0.001             # Learning rate for Actor
    LRC = 0.001             # Learning rate for Critic

    explore_iter = 1000000.
    # explore_iter = 1000.
    episode_count = 20000
    max_steps = 2000

    action_dim = 1          # Steering/Acceleration/Brake
    action_size = 1

    # Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf_sess = tf.Session(config=config)
    keras.set_session(tf_sess)

    Speed_limit = 12

    def __init__(self):
        self.sim = InterSim()
        self.reward = AppReward()
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

        self.app_actor = AppActorNetwork(self.tf_sess, 9, self.action_dim, 10, self.tau, self.LRA)
        self.app_critic = AppCriticNetwork(self.tf_sess, 9, self.action_dim, 10, self.tau, self.LRC)
        self.buffer = AppReplay()

        self.batch = None
        self.batch_state = None
        self.batch_action = None
        self.batch_reward = None
        self.batch_new_state = None
        self.batch_if_done = None
        self.batch_output = None

        self.start_time = time.time()
        self.end_time = time.time()
        self.total_time = 0.

    def load_weights(self):
        # logging.info('...... Loading weight ......')
        try:
            self.app_actor.model.load_weights("../weights/actormodel.h5")
            self.app_critic.model.load_weights("../weights/criticmodel.h5")
            self.app_actor.target_model.load_weights("../weights/actormodel.h5")
            self.app_critic.target_model.load_weights("../weights/criticmodel.h5")
            # logging.info("Weight load successfully")
        except:
            logging.warn("Cannot find the weight !")

    def update_weights(self):
        # logging.info('...... Updating weight ......')
        self.app_actor.model.save_weights("../weights/actormodel.h5", overwrite=True)
        with open("../weights/actormodel.json", "w") as outfile:
            json.dump(self.app_actor.model.to_json(), outfile)
        self.app_critic.model.save_weights("../weights/criticmodel.h5", overwrite=True)
        with open("../weights/criticmodel.json", "w") as outfile:
            json.dump(self.app_critic.model.to_json(), outfile)

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
        target_q_values = self.app_critic.target_model.predict(
            [self.batch_new_state, self.app_actor.target_model.predict(self.batch_new_state)])
        for k, done in enumerate(self.batch_if_done):
            self.batch_output[k] = self.batch_reward[k] if done else self.batch_reward[k] + self.gamma * target_q_values[k]

    def update_loss(self):
        # logging.info('...... Updating loss ......')
        loss = self.app_critic.model.train_on_batch([self.batch_state, self.batch_action], self.batch_output)
        actor_predict = self.app_actor.model.predict(self.batch_state)
        actor_grad = self.app_critic.gradients(self.batch_state, actor_predict)
        self.app_actor.train(self.batch_state, actor_grad)
        self.app_actor.target_train()
        self.app_critic.target_train()
        return loss

    def get_action(self, state_t, train_indicator):
        # logging.info('...... Getting action ......')
        self.epsilon -= 1.0 / self.explore_iter
        noise = []
        action_ori = self.app_actor.model.predict(state_t)
        for i in range(self.action_size):
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
        state_t = self.sim.get_state()
        state_dim = state_t.shape[1]
        self.app_actor = AppActorNetwork(self.tf_sess, state_dim, self.action_size, self.batch_size, self.tau, self.LRA)
        self.app_critic = AppCriticNetwork(self.tf_sess, state_dim, self.action_size, self.batch_size, self.tau, self.LRC)
        self.buffer = AppReplay(self.buffer_size)
        self.load_weights()

        for e in range(self.episode_count):
            total_loss = 0.
            total_time = 0.
            # logging.debug("Episode : " + str(e) + " Replay Buffer " + str(self.buffer.count()))
            step = 0
            state_t = self.sim.get_state()
            while True:
                action_t = self.get_action(state_t, train_indicator)
                reward_t, collision, not_move = self.reward.get_reward(state_t[0], action_t[0][0])
                self.sim.update_vehicle(reward_t, action_t[0][0])
                state_t1 = self.sim.get_state()
                self.update_batch(state_t, action_t[0], reward_t, state_t1)
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
