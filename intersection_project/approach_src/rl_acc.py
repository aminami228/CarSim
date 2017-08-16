#!/usr/bin/env python
import sys
sys.path.append('/home/scotty/qzq/git/CarSim_acceleration/intersection_project')
import logging
import numpy as np
import tensorflow as tf
import json
from network.ActorNetwork import ActorNetwork
from network.CriticNetwork import CriticNetwork
from network.ReplayBuffer import ReplayBuffer
from utilities.toolfunc import ToolFunc
from inter_sim import InterSim
from keras import backend as keras
import time
import matplotlib.pyplot as plt
from reward_func import Reward
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

    # Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf_sess = tf.Session(config=config)
    keras.set_session(tf_sess)

    def __init__(self):
        self.sim = InterSim()
        self.reward = Reward()
        self.total_reward = 0
        self.if_pass = False
        self.if_done = False

        self.crash = 0.
        self.not_stop = 0.
        self.success = 0.
        self.not_finish = 0.
        self.overspeed = 0.

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
            self.batch_output[k] = self.batch_reward[k] if done else self.batch_reward[k] + self.gamma * target_q_values[k]

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
        action_ori = 2. * self.sim.Cft_Accel * (self.actor_network.model.predict(state_t) - 0.5)
        for i in range(self.action_size):
            a = action_ori[0][i]
            noise.append(train_indicator * max(self.epsilon, 0) * self.tools.ou(a, 0.5, 0.5, 0.5))
        action = action_ori + np.array(noise)
        return action

    def if_exit(self, step, loc, v, collision):
        if step >= self.max_steps:
            logging.warn('Not finished with max steps! Start: ' + str(self.sim.Start_Pos) + ', Position: ' +
                         str(loc) + ', Velocity: ' + str(v))
            self.not_finish += 1.
            self.if_pass = False
            self.if_done = True
        elif v >= self.sim.Speed_limit + 2.:
            logging.warn('Exceed Speed Limit: ' + str(self.sim.Start_Pos) + ', Position: ' +
                         str(loc) + ', Velocity: ' + str(v))
            self.overspeed += 1.
            self.if_pass = False
            self.if_done = True
        elif collision > 0:
            logging.warn('Crash to other vehicles or road boundary! Start: ' + str(self.sim.Start_Pos) + ', Position: '
                         + str(loc) + ', Velocity: ' + str(v))
            self.crash += 1.
            self.if_pass = False
            self.if_done = True
        elif collision == 0 and (loc >= self.sim.Stop_Line - 1.) and (v > 2.0):
            logging.warn('No crash and reached stop line. But has not stopped! Start: ' + str(self.sim.Start_Pos) +
                         ', Position: ' + str(loc) + ', Velocity: ' + str(v))
            self.not_stop += 1.
            self.if_pass = False
            self.if_done = True
        elif collision == 0 and loc >= self.sim.Stop_Line - 1.0 and (v <= 2.0):
            logging.info('Congratulations! Reach stop line without crashing and has stopped. Start: ' +
                         str(self.sim.Start_Pos) + ', Position: ' + str(loc) + ', Velocity: ' +
                         str(v))
            self.success += 1.
            self.if_pass = True
            self.if_done = True

    def launch_train(self, train_indicator=1):  # 1 means Train, 0 means simply Run
        # logging.info('Launch Training Process')
        # np.random.seed(1337)
        state_t = self.sim.get_state()
        state_dim = state_t.shape[1]
        self.actor_network = ActorNetwork(self.tf_sess, state_dim, self.action_size, self.batch_size, self.tau, self.LRA)
        self.critic_network = CriticNetwork(self.tf_sess, state_dim, self.action_size, self.batch_size, self.tau, self.LRC)
        self.buffer = ReplayBuffer(self.buffer_size)
        self.load_weights()

        for e in range(self.episode_count):
            total_loss = 0.
            total_time = 0.
            # logging.debug("Episode : " + str(e) + " Replay Buffer " + str(self.buffer.count()))
            step = 0
            while True:
                state_t = self.sim.get_state()
                start_time = time.time()
                action_t = self.get_action(state_t, train_indicator)
                reward_t, collision = self.reward.get_reward(state_t[0], action_t[0][0])
                train_time = time.time() - start_time
                self.sim.update_vehicle(reward_t, action_t[0][0])
                state_t1 = self.sim.get_state()
                start_time = time.time()
                self.update_batch(state_t, action_t[0][0], reward_t, state_t1)
                if train_indicator:
                    loss = self.update_loss()
                self.total_reward += reward_t
                self.if_exit(step, self.sim.av_pos['y'], self.sim.av_pos['vy'], collision)
                step += 1
                total_loss += loss
                train_time += time.time() - start_time
                # logging.debug('Episode: ' + str(e) + ', Step: ' + str(step) + ', loc: ' + str(self.sim.av_pos['y']) +
                #               ', velocity: ' + str(self.sim.av_pos['vy']) + ', action: ' + str(action_t) +
                #               ', reward: ' + str(reward_t) + ', loss: ' + str(loss) + ', Training time: ' +
                #               str(train_time))
                total_time += train_time
                if self.if_done:
                    break

            plt.close('all')
            total_step = step + 1
            if train_indicator:
                self.update_weights()

            mean_loss = total_loss / total_step
            mean_time = total_time / total_step
            logging.debug(str(e) + "-th Episode: Steps: " + str(total_step) + ', Time: ' + str(mean_time) +
                          ', Reward: ' + str(self.total_reward) + " Loss: " + str(mean_loss) + ', Crash: ' +
                          str(self.crash) + ', Not Stop: ' + str(self.not_stop) + ', Not Finished: ' +
                          str(self.not_finish) + ', Overspeed: ' + str(self.overspeed) + ', Success: ' +
                          str(self.success))

            self.sim = InterSim(True) if e % 50 == 0 else InterSim()
            self.total_reward = 0
            self.if_pass = False
            self.if_done = False


if __name__ == '__main__':
    plt.ion()
    acc = ReinAcc()
    acc.launch_train()
