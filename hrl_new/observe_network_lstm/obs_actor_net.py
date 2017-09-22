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


class ObsActorNetwork(object):
    def __init__(self, sess, his_len, state_dim, action_size, batch_size, sigma, learn_rate):
        self.sess = sess
        self.BATCH_SIZE = batch_size
        self.TAU = sigma
        self.LEARNING_RATE = learn_rate

        keras.set_session(sess)

        # Now create the model
        self.model, self.weights, self.state = self.create_actor_network(his_len, state_dim)
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(his_len, state_dim)
        self.action_gradient = tf.placeholder(tf.float32, [None, action_size])
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
    def create_actor_network(his_len, state_dim):
        logging.info('...... Building actor model ......')
        hist_state = Input(shape=(his_len, state_dim))
        # hist0 = keras.reshape(hist_state, (-1, his_len, state_dim))
        hist1 = LSTM(HIDDEN1_UNITS, return_sequences=True)(hist_state)
        # hist2 = Dropout(0.2)(hist1)
        hist3 = LSTM(HIDDEN2_UNITS, return_sequences=False)(hist)
        # hist4 = Dropout(0.2)(hist3)
        h1 = Dense(HIDDEN3_UNITS, activation='relu')(hist4)
        h2 = Dense(HIDDEN5_UNITS, activation='relu')(h1)
        action = Dense(1, activation='tanh', kernel_initializer=RandomNormal(mean=0.0, stddev=1e-4, seed=None))(h2)
        model = Model(input=hist_state, outputs=action)
        return model, model.trainable_weights, hist_state
