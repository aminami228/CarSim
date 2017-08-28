import numpy as np
import math
from keras.layers import Dense, Flatten, Input, merge, Lambda, Activation, concatenate, LSTM
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
    def __init__(self, sess, non_his_size, his_len, his_size, action_size, batch_size, sigma, learn_rate):
        self.sess = sess
        self.BATCH_SIZE = batch_size
        self.TAU = sigma
        self.LEARNING_RATE = learn_rate
        self.action_size = action_size
        
        keras.set_session(sess)

        # Now create the model
        self.model, self.action, self.state = self.create_critic_network(non_his_size, his_len, his_size, action_size)
        self.target_model, self.target_action, self.target_state = \
            self.create_critic_network(non_his_size, his_len, his_size, action_size)
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

    def create_critic_network(self, non_his_size, his_len, his_size, action_size):
        logging.info('...... Building critic model ......')
        non_his_state = Input(shape=(non_his_size,))
        n1 = Dense(non_his_size, activation='linear')(non_his_state)

        his_state = Input(shape=(his_len, his_size,))
        # h0 = keras.reshape(his_state, (-1, his_len, his_size))
        h1 = LSTM(HIDDEN1_UNITS, return_sequences=True)(his_state)
        h2 = LSTM(HIDDEN2_UNITS, return_sequences=False)(h1)

        action = Input(shape=[action_size], name='action')
        a1 = Dense(action_size, activation='linear')(action)

        c1 = concatenate([h2, n1, a1])
        c2 = Dense(HIDDEN2_UNITS, activation='relu')(c1)
        c3 = Dense(HIDDEN3_UNITS, activation='relu')(c2)
        c4 = Dense(HIDDEN4_UNITS, activation='relu')(c3)

        Q = Dense(1, activation='linear')(c4)
        model = Model(input=[non_his_state, his_state, action], output=Q)
        adam = Adam(lr=self.LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
        return model, action, [non_his_state, his_state]
