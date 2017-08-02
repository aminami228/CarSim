#!/usr/bin/env python
import numpy as np
import tensorflow as tf
import json
from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from intersection import UpdateInter
from OU import OU
from keras import backend as K


class MainTrain(object):
    buffer_size = 100000
    batch_size = 100
    gamma = 0.99
    tau = 0.0001  # Target Network HyperParameters
    LRA = 0.001  # Learning rate for Actor
    LRC = 0.001  # Learning rate for Critic
    explore_iter = 100000.
    episode_count = 20000
    max_steps = 2000
    action_dim = 4  # Steering/Acceleration/Brake
    parameter_acc_dim = 2
    parameter_time_dim = action_dim
    action_size = action_dim + parameter_acc_dim + parameter_time_dim

    def __init__(self):
        self.OU = OU()

        self.total_correct = 0
        self.total_wrong = 0
        self.accuracy_all = []
        self.if_done = False
        self.epsilon = 1
        self.total_reward = None
        self.loss = None

        self.sim_inter = UpdateInter()
        self.state_t = []
        self.state_dim = self.sim_inter.state_dim
        self.action_t = []
        self.action_acc = None
        self.action_time = None
        self.Tau = self.sim_inter.Tau

        self.actor = None
        self.critic = None
        self.buff = None

        self.batch = None
        self.states = None
        self.actions = None
        self.rewards = None
        self.new_states = None
        self.if_dones = None
        self.y_t = None

        # Tensorflow GPU optimization
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        K.set_session(self.sess)

    def load_weights(self):
        print("Now we load the weight")
        try:
            self.actor.model.load_weights("actormodel.h5")
            self.critic.model.load_weights("criticmodel.h5")
            self.actor.target_model.load_weights("actormodel.h5")
            self.critic.target_model.load_weights("criticmodel.h5")
            print("Weight load successfully")
        except:
            print("Cannot find the weight")

    def update_weights(self):
        self.actor.model.save_weights("actormodel.h5", overwrite=True)
        with open("actormodel.json", "w") as outfile:
            json.dump(self.actor.model.to_json(), outfile)
        self.critic.model.save_weights("criticmodel.h5", overwrite=True)
        with open("criticmodel.json", "w") as outfile:
            json.dump(self.critic.model.to_json(), outfile)

    def update_batch(self):
        self.batch = self.buff.getBatch(self.batch_size)
        self.states = np.squeeze(np.asarray([e[0] for e in self.batch]), axis=1)
        self.actions = np.asarray([e[1] for e in self.batch])
        self.rewards = np.asarray([e[2] for e in self.batch])
        self.new_states = np.squeeze(np.asarray([e[3] for e in self.batch]), axis=1)
        self.if_dones = np.asarray([e[4] for e in self.batch])
        self.y_t = np.asarray([e[2] for e in self.batch])
        target_q_values = self.critic.target_model.predict(
            [self.new_states, self.actor.target_model.predict(self.new_states)])
        for k, done in enumerate(self.if_dones):
            self.y_t[k] = self.rewards[k] if done else self.rewards[k] + self.gamma * target_q_values[k]

    def update_loss(self):
        self.loss += self.critic.model.train_on_batch([self.states, self.actions], self.y_t)
        a_for_grad = self.actor.model.predict(self.states)
        grads = self.critic.gradients(self.states, a_for_grad)
        self.actor.train(self.states, grads)
        self.actor.target_train()
        self.critic.target_train()

    def action_noise(self, train_indicator):
        self.epsilon -= 1.0 / self.explore_iter
        noise_t = np.zeros([1, self.action_size])
        action_t_original = self.actor.model.predict(self.state_t)
        print("Action ", action_t_original)
        for i in range(self.action_dim):
            noise_t[0][i] = train_indicator * max(self.epsilon, 0) * \
                            self.OU.function(action_t_original[0][i], 0.00, 0.10, 0.20)
        noise_t[0][4] = train_indicator * max(self.epsilon, 0) * \
                        self.OU.function(action_t_original[0][4], -0.05, self.sim_inter.Max_Acc, 1.00)
        noise_t[0][5] = train_indicator * max(self.epsilon, 0) * \
                        self.OU.function(action_t_original[0][5], 0.05, - self.sim_inter.Max_Acc, 1.00)
        for i in range(self.parameter_time_dim):
            noise_t[0][i + self.action_dim + self.parameter_acc_dim] = \
                train_indicator * max(self.epsilon, 0) * \
                self.OU.function(action_t_original[0][i + self.action_dim + self.parameter_acc_dim], 0.01, 0.50, 0.10)
        action = np.zeros([1, self.action_size])
        for i in range(self.action_size):
            action[0][i] = action_t_original[0][i] + noise_t[0][i]
        return action

    def update_action(self, action, train_indicator, e):
        if action == 0:
            process = 'Approach Process'
            self.action_acc = self.action_t[0][4]
            self.action_time = self.action_t[0][6]
        elif action == 1:
            process = 'Observe Process'
            self.action_time = self.action_t[0][7]
        elif action == 2:
            process = 'Wait Process'
            self.action_time = self.action_t[0][8]
        else:
            process = 'Traverse Process'
            self.action_acc = self.action_t[0][5]
            self.action_time = self.action_t[0][9]
        time_step = int(np.ceil(max(self.action_time / self.Tau, 1.0)))

        collision = False
        if_pass = False
        for ts in range(time_step):
            old_av_y = self.sim_inter.av_y
            old_av_velocity = self.sim_inter.av_velocity
            if action == 1:
                self.action_acc = (self.sim_inter.observe_vel - self.sim_inter.av_velocity) / self.sim_inter.Tau
            elif action == 2:
                self.action_acc = (- self.sim_inter.av_velocity) / self.sim_inter.Tau
            reward, collision = self.sim_inter.reward_function(self.action_acc)
            state_t1 = self.sim_inter.update_vehicle(self.action_acc)

            self.buff.add(self.state_t, self.action_t[0], reward, state_t1, self.if_done)
            self.update_batch()

            if train_indicator:
                self.update_loss()
            self.total_reward += reward
            print process, " (", self.action_acc, ", ", self.action_time, ") ", "AV = ", old_av_y, \
                "Velocity = ", old_av_velocity, "Episode", e, "Reward", reward, "Loss", self.loss

            if action == 1 and self.state_t[0][0] <= 0:
                self.state_t = state_t1
                break
            if old_av_y > self.sim_inter.Pass_Point or collision > 0:
                if_pass = old_av_y > self.sim_inter.Pass_Point
                self.if_done = True
                break
            self.state_t = state_t1
        return collision, if_pass

    def launch_train(self, train_indicator=1):  # 1 means Train, 0 means simply Run
        print 'Launch Training Process'
        np.random.seed(1337)

        self.state_t = self.sim_inter.get_state()
        self.state_dim = self.sim_inter.state_dim
        self.actor = ActorNetwork(self.sess, self.state_dim, self.action_size, self.batch_size, self.tau, self.LRA)
        self.critic = CriticNetwork(self.sess, self.state_dim, self.action_size, self.batch_size, self.tau, self.LRC)
        self.buff = ReplayBuffer(self.buffer_size)
        self.load_weights()

        for e in range(self.episode_count):
            print("Episode : " + str(e) + " Replay Buffer " + str(self.buff.count()))

            for j in range(self.max_steps):
                self.loss = 0
                self.total_reward = 0
                self.action_t = self.action_noise(train_indicator)
                choose_action = np.argmax(self.action_t[0][0:4])
                collision, if_pass = self.update_action(choose_action, train_indicator, e)

                if self.if_done:
                    self.sim_inter = UpdateInter()
                    self.state_t = self.sim_inter.get_state()
                    self.if_done = False
                    break

            if train_indicator:
                self.update_weights()

            self.total_correct += int(collision <= 0 and if_pass)
            self.total_wrong += int(collision > 0)
            accuracy = 0
            if self.total_correct + self.total_wrong:
                accuracy = self.total_correct / (self.total_correct + self.total_wrong)

            if np.mod(e, 100) == 0:
                self.accuracy_all.append(accuracy)
                self.total_correct = 0
                self.total_wrong = 0

            print("TOTAL REWARD @ " + str(e) + "-th Episode  : Reward " + str(self.total_reward) +
                  " Collision " + str(collision > 0) + " Accuracy " + str(accuracy) +
                  " All Accuracy " + str(self.accuracy_all))
            print("")
        print("Finish.")


if __name__ == "__main__":
    sim = MainTrain()
    sim.launch_train(train_indicator=1)
