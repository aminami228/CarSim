#!/usr/bin/env python
import numpy as np
import tensorflow as tf
import json
from network.ActorNetwork import ActorNetwork
from network.CriticNetwork import CriticNetwork
from network.ReplayBuffer import ReplayBuffer
from utilities.OU import OU
from inter_sim import InterSim
from keras import backend as K

__author__ = 'qzq'


class ReinAcc(object):
    OU = OU()
    buffer_size = 100000
    batch_size = 100
    gamma = 0.99
    tau = 0.0001            # Target Network HyperParameters
    LRA = 0.001             # Learning rate for Actor
    LRC = 0.001             # Learning rate for Critic
    explore_iter = 100000
    episode_count = 20000
    max_steps = 2000
    action_dim = 4          # Steering/Acceleration/Brake
    parameter_acc_dim = 2
    parameter_time_dim = action_dim
    action_size = action_dim + parameter_acc_dim + parameter_time_dim

    # Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf_sess = tf.Session(config=config)
    K.set_session(tf_sess)

    def __init__(self):
        self.actor_network = ActorNetwork()
        self.critic_network = CriticNetwork()
        self.buffer = ReplayBuffer()
        self.batch = None

    def load_weights(self):
        print('...... Loading weight ......')
        try:
            self.actor_network.model.load_weights("actormodel.h5")
            self.critic_network.model.load_weights("criticmodel.h5")
            self.actor_network.target_model.load_weights("actormodel.h5")
            self.critic_network.target_model.load_weights("criticmodel.h5")
            print("Weight load successfully")
        except:
            print("Cannot find the weight !")

    def update_weights(self):
        self.actor_network.model.save_weights("actormodel.h5", overwrite=True)
        with open("actormodel.json", "w") as outfile:
            json.dump(self.actor_network.model.to_json(), outfile)
        self.critic_network.model.save_weights("criticmodel.h5", overwrite=True)
        with open("criticmodel.json", "w") as outfile:
            json.dump(self.critic_network.model.to_json(), outfile)

    def update_batch(self):
        self.batch = self.buffer.get_batch(self.batch_size)
        self.states = np.squeeze(np.asarray([e[0] for e in self.batch]), axis=1)
        self.actions = np.asarray([e[1] for e in self.batch])
        self.rewards = np.asarray([e[2] for e in self.batch])
        self.new_states = np.squeeze(np.asarray([e[3] for e in self.batch]), axis=1)
        self.if_dones = np.asarray([e[4] for e in self.batch])
        self.y_t = np.asarray([e[2] for e in self.batch])
        target_q_values = self.critic_network.target_model.predict(
            [self.new_states, self.actor_network.target_model.predict(self.new_states)])
        for k, done in enumerate(self.if_dones):
            self.y_t[k] = self.rewards[k] if done else self.rewards[k] + self.gamma * target_q_values[k]

    def update_loss(self):
        self.loss += self.critic_network.model.train_on_batch([self.states, self.actions], self.y_t)
        a_for_grad = self.actor_network.model.predict(self.states)
        grads = self.critic_network.gradients(self.states, a_for_grad)
        self.actor_network.train(self.states, grads)
        self.actor_network.target_train()
        self.critic_network.target_train()

    def acc_target(self, av_pos, target_pos):
        pass


if __name__ == '__main__':
    acc = ReinAcc()
    acc.acc_target()
