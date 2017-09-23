import numpy as np
import math
from keras.layers import Dense, Flatten, Input, merge, Lambda, Activation, concatenate, LSTM, Dropout
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras.backend as keras
import tensorflow as tf
import sys
import logging
import utilities.log_color

HIDDEN1_UNITS = 128
HIDDEN2_UNITS = 64
HIDDEN3_UNITS = 32
HIDDEN4_UNITS = 32
HIDDEN5_UNITS = 16


class ObsCriticNetwork(object):
    def __init__(self, sess,  his_len, state_dim, action_size, batch_size, sigma, learn_rate):
        self.sess = sess
        self.BATCH_SIZE = batch_size
        self.TAU = sigma
        self.LEARNING_RATE = learn_rate
        self.action_size = action_size
        
        keras.set_session(sess)

        # Now create the model
        self.model, self.action, self.state = self.create_critic_network(his_len, state_dim, action_size)
        self.target_model, self.target_action, self.target_state = \
            self.create_critic_network(his_len, state_dim, action_size)
        self.action_grads = tf.gradients(self.model.output, self.action)  # GRADIENTS for policy update
        self.sess.run(tf.global_variables_initializer())

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU) * critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def create_critic_network(self, his_len, state_dim, action_size):
        logging.info('...... Building critic model ......')
        hist_state = Input(shape=[his_len, state_dim])
        hist1 = LSTM(HIDDEN1_UNITS, return_sequences=False)(hist_state)
        # hist2 = Dropout(0.2)(hist1)
        s0 = Dense(HIDDEN1_UNITS, activation='linear')(hist1)
        action = Input(shape=[action_size], name='action')
        a0 = Dense(action_size, activation='linear')(action)
        h0 = concatenate([s0, a0])
        h1 = Dense(HIDDEN2_UNITS, activation='relu')(h0)
        h2 = Dense(HIDDEN3_UNITS, activation='relu')(h1)
        h3 = Dense(HIDDEN5_UNITS, activation='relu')(h2)
        Q = Dense(1, activation='linear')(h3)
        model = Model(input=[hist_state, action], output=Q)
        adam = Adam(lr=self.LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
        return model, action, hist_state
