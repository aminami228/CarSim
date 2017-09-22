import numpy as np
import math
from keras.layers import Dense, Flatten, Input, merge, Lambda, Activation, concatenate
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
HIDDEN4_UNITS = 16


class AppCriticNetwork(object):
    def __init__(self, sess, state_size, action_size, batch_size, sigma, learn_rate):
        self.sess = sess
        self.BATCH_SIZE = batch_size
        self.TAU = sigma
        self.LEARNING_RATE = learn_rate
        self.action_size = action_size
        
        keras.set_session(sess)

        # Now create the model
        self.model, self.action, self.state = self.create_critic_network(state_size, action_size)  
        self.target_model, self.target_action, self.target_state = self.create_critic_network(state_size, action_size)  
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

    def create_critic_network(self, state_size, action_size):
        logging.info('...... Building critic model ......')
        S = Input(shape=[state_size])  
        A = Input(shape=[action_size], name='action2')
        w0 = Dense(state_size, activation='linear')(S)
        a0 = Dense(action_size, activation='linear')(A)
        # h0 = merge([w0, a0], mode='concat')
        h0 = concatenate([w0, a0])
        h1 = Dense(HIDDEN2_UNITS, activation='relu')(h0)
        h2 = Dense(HIDDEN3_UNITS, activation='sigmoid')(h1)
        # h3 = Dense(HIDDEN3_UNITS, activation='relu')(h2)
        h4 = Dense(HIDDEN4_UNITS, activation='relu')(h1)
        V = Dense(1, activation='linear')(h4)
        model = Model(input=[S, A], output=V)
        adam = Adam(lr=self.LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
        return model, A, S 
