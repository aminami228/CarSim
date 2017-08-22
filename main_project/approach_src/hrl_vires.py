#!/usr/bin/env python
from random import random
import numpy as np
import tensorflow as tf
from keras import backend as keras
import json
import time
import logging
import sys
sys.path.append('/home/scotty/qzq/git/CarSim_vires/main_project')
from reward_vires import Reward
from network.ActorNetwork import ActorNetwork
from network.CriticNetwork import CriticNetwork
from network.ReplayBuffer import ReplayBuffer
from interface.vires_sim import InterSim
from utilities.toolfunc import ToolFunc
import utilities.log_color

__author__ = 'qzq'

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class ReinAcc(object):
    tools = ToolFunc()

    Tau = 1. / 30
    gamma = 0.99
    epsilon = 1.

    buffer_size = 10000
    batch_size = 128
    tau = 0.0001            # Target Network HyperParameters
    LRA = 0.001             # Learning rate for Actor
    LRC = 0.001             # Learning rate for Critic

    explore_iter = 1000000.
    episode_count = 20000
    max_steps = 2000

    action_dim = 1          # Steering/Acceleration/Brake
    action_size = 1
    state_dim = 10

    # Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf_sess = tf.Session(config=config)
    keras.set_session(tf_sess)

    Speed_limit = 12

    def __init__(self):
        self.sim = InterSim(self.Speed_limit - 5. * random())
        self.reward = Reward()
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

        self.sub_crash = 0.
        self.sub_not_stop = 0.
        self.sub_success = 0.
        self.sub_not_finish = 0.
        self.sub_overspeed = 0.
        self.sub_not_move = 0.

        self.actor_network = ActorNetwork(self.tf_sess, 9, self.action_dim, 10, self.tau, self.LRA)
        self.critic_network = CriticNetwork(self.tf_sess, 9, self.action_dim, 10, self.tau, self.LRC)
        self.buffer = ReplayBuffer()

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
            self.actor_network.model.load_weights("../weights/actormodel.h5")
            self.critic_network.model.load_weights("../weights/criticmodel.h5")
            self.actor_network.target_model.load_weights("../weights/actormodel.h5")
            self.critic_network.target_model.load_weights("../weights/criticmodel.h5")
            # logging.info("Weight load successfully")
        except:
            logging.warn("Cannot find the weight !")

    def update_weights(self):
        # logging.info('...... Updating weight ......')
        self.actor_network.model.save_weights("../weights/actormodel.h5", overwrite=True)
        with open("../weights/actormodel.json", "w") as outfile:
            json.dump(self.actor_network.model.to_json(), outfile)
        self.critic_network.model.save_weights("../weights/criticmodel.h5", overwrite=True)
        with open("../weights/criticmodel.json", "w") as outfile:
            json.dump(self.critic_network.model.to_json(), outfile)

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
        target_q_values = self.critic_network.target_model.predict(
            [self.batch_new_state, self.actor_network.target_model.predict(self.batch_new_state)])
        for k, done in enumerate(self.batch_if_done):
            self.batch_output[k] = self.batch_reward[k] if done \
                else self.batch_reward[k] + self.gamma * target_q_values[k]

    def update_loss(self):
        # logging.info('...... Updating loss ......')
        loss = self.critic_network.model.train_on_batch([self.batch_state, self.batch_action], self.batch_output)
        actor_predict = self.actor_network.model.predict(self.batch_state)
        actor_grad = self.critic_network.gradients(self.batch_state, actor_predict)
        self.actor_network.train(self.batch_state, actor_grad)
        self.actor_network.target_train()
        self.critic_network.target_train()
        return loss

    def get_action(self, state_t, train_indicator):
        # logging.info('...... Getting action ......')
        self.epsilon -= 1.0 / self.explore_iter
        noise = []
        action_ori = self.sim.Cft_Accel * self.actor_network.model.predict(state_t)
        noise.append(train_indicator * max(self.epsilon, 0) * self.tools.ou(action_ori[0][0], 1.0, 0.3, -0.6))
        action = action_ori + np.array(noise)
        return action

    def if_exit(self, step, state, collision, not_move):
        if step >= self.max_steps:
            logging.warn('Not finished with max steps! Start: ' + str(state[9]) + ', Dis to SL: ' + str(state[4]) +
                         ', Velocity: ' + str(state[0]))
            self.sub_not_finish += 1.
            self.if_pass = False
            self.if_done = True
        elif state[0] >= self.sim.Speed_limit + 2.:
            logging.warn('Exceed Speed Limit: ' + str(state[9]) + ', Dis to SL: ' + str(state[4]) +
                         ', Velocity: ' + str(state[0]))
            self.sub_overspeed += 1.
            self.if_pass = False
            self.if_done = True
        elif not_move > 0:
            logging.warn('Not move! Start: ' + str(state[9]) + ', Dis to SL: ' + str(state[4]) +
                         ', Velocity: ' + str(state[0]))
            self.sub_not_move += 1.
            self.if_pass = False
            self.if_done = True
        elif collision > 0:
            logging.warn('Crash to other vehicles or road boundary! Start: ' + str(state[9]) + ', Dis to SL: '
                         + str(state[4]) + ', Velocity: ' + str(state[0]))
            self.sub_crash += 1.
            self.if_pass = False
            self.if_done = True
        elif collision == 0 and (state[6] <= 1.0) and (state[0] > 2.0):
            logging.warn('No crash and reached stop line. But has not stopped! Start: ' + str(state[9]) +
                         ', Dis to SL: ' + str(state[4]) + ', Velocity: ' + str(state[0]))
            self.sub_not_stop += 1.
            self.if_pass = False
            self.if_done = True
        elif collision == 0 and (state[6] <= 1.0) and (state[0] <= 2.0):
            logging.info('Congratulations! Reach stop line without crashing and has stopped. Start: ' +
                         str(state[9]) + ', Dis to SL: ' + str(state[4]) + ', Velocity: ' + str(state[0]))
            self.sub_success += 1.
            self.if_pass = True
            self.if_done = True

    def launch_train(self, train_indicator=1):
        # logging.info('Launch Training Process')
        state_dim = self.state_dim
        self.actor_network = ActorNetwork(self.tf_sess, state_dim, self.action_size, self.batch_size, self.tau, self.LRA)
        self.critic_network = CriticNetwork(self.tf_sess, state_dim, self.action_size, self.batch_size, self.tau, self.LRC)
        self.buffer = ReplayBuffer(self.buffer_size)
        self.load_weights()

        for e in range(self.episode_count):
            total_time = 0.
            step = 0
            a = 0
            begin_time = time.time()
            while not self.if_done:
                state_t = self.sim.get_state(a)
                a_time = time.time()
                action_t = self.get_action(state_t, train_indicator)
                a = action_t[0][0]
                reward_t, collision, not_move = self.reward.get_reward(state_t[0], a)
                self.sim.update_vehicle(state_t[0], a, time.time() - a_time)
                self.start_time = time.time()
                state_t1 = self.sim.get_state(a)
                self.update_batch(state_t, action_t[0], reward_t, state_t1)
                loss = self.update_loss() if train_indicator else 0.

                self.total_reward += reward_t
                self.if_exit(step, state_t[0], collision, not_move)
                step += 1
                train_time = time.time() - self.start_time
                self.start_time = time.time()
                # logging.debug('Episode: ' + str(e) + ', Step: ' + str(step) + ', Dis to SL: ' + str(state_t[0][6]) +
                #               ', velocity: ' + str(state_t[0][0]) + ', action: ' + str(action_t[0]) +
                #               ', reward: ' + str(reward_t) + ', loss: ' + str(loss) + ', Training time: ' +
                #               str(train_time))
                time.sleep(0.0)

            if train_indicator:
                self.update_weights()

            total_step = step + 1
            mean_time = (time.time() - begin_time) / total_step
            logging.debug(str(e) + "-th Episode: Steps: " + str(total_step) + ', Time: ' + str(mean_time) +
                          ', Reward: ' + str(self.total_reward) + " Loss: " + str(loss) + ', Crash: ' +
                          str(self.sub_crash) + ', Not Stop: ' + str(self.sub_not_stop) + ', Not Finished: ' +
                          str(self.sub_not_finish) + ', Overspeed: ' + str(self.sub_overspeed) + ', Not Move: ' +
                          str(self.sub_not_move) + ', Success: ' + str(self.sub_success))

            self.sim = InterSim(self.Speed_limit / 2 * random())
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
    acc = ReinAcc()
    # time.sleep(1.)
    acc.launch_train()             # 1 means Train, 0 means simply Run