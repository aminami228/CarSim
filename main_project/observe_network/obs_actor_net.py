import numpy as np
import math
#from keras.initializations import normal, identity
from keras.initializers import RandomNormal, identity
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Input, Lambda, concatenate, LSTM, Dropout, Merge
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as keras
import logging
import utilities.log_color

HIDDEN1_UNITS = 128
HIDDEN2_UNITS = 64
HIDDEN3_UNITS = 32
HIDDEN4_UNITS = 32
HIDDEN5_UNITS = 16


class ObsActorNetword(object):
    def __init__(self, sess, state_size, his_len, his_size, action_size, batch_size, sigma, learn_rate):
        self.sess = sess
        self.BATCH_SIZE = batch_size
        self.TAU = sigma
        self.LEARNING_RATE = learn_rate

        keras.set_session(sess)

        # Now create the model
        self.model, self.weights, self.state = self.create_actor_network(state_size, his_len, his_size)
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_size)
        self.action_gradient = tf.placeholder(tf.float32,[None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(learn_rate).apply_gradients(grads)
        self.sess.run(tf.global_variables_initializer())

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads})

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU) * actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    @staticmethod
    def create_actor_network(state_size, his_len, his_size):
        # logging.info('...... Building actor model ......')
        non_hist_state = Input(shape=(state_size,))
        hist_state = Input(shape=(his_len, his_size))
        hist1 = LSTM(HIDDEN2_UNITS)(hist_state)
        hist2 = Dropout(0.2)(hist1)
        hist2 = Dropout(0.2)(hist1)

        hist = Sequential()
        hist.add(LSTM(input_shape=(his_len, his_size),
                      output_dim=his_len,
                      return_sequences=True))
        hist.add(Dropout(0.2))
        hist.add(LSTM(HIDDEN2_UNITS, return_sequences=False))
        hist.add(Dropout(0.2))

        merged = Merge([non_hist, hist], mode='concat')
        actor = Sequential()
        actor.add(merged)
        actor.add(Dense(HIDDEN5_UNITS, activation='relu'))
        actor.add(Dense(1, activation='tanh', kernel_initializer=RandomNormal(mean=0.0, stddev=1e-4, seed=None)))

        return actor, actor.trainable_weights, non_hist
