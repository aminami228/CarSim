import numpy as np
import math
from keras.initializers import RandomNormal, identity
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout
import tensorflow as tf
import keras.backend as keras
import logging
import utilities.log_color

HIDDEN1_UNITS = 128
HIDDEN2_UNITS = 64
HIDDEN3_UNITS = 32
HIDDEN4_UNITS = 16


class HrlActorNetwork(object):
    def __init__(self, sess, state_size, action_size, batch_size, sigma, learn_rate):
        self.sess = sess
        self.BATCH_SIZE = batch_size
        self.TAU = sigma
        self.LEARNING_RATE = learn_rate

        keras.set_session(sess)
        keras.set_learning_phase(1)

        # Now create the model
        self.model, self.weights, self.state = self.create_actor_network(state_size)
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_size)
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
    def create_actor_network(state_size):
        # logging.info('...... Building actor model ......')
        S  = Input(shape=[state_size])
        h0 = Dense(HIDDEN1_UNITS, activation='relu')(S)
        h0d = Dropout(0.5)(h0)
        h1 = Dense(HIDDEN2_UNITS, activation='relu')(h0d)
        h1d = Dropout(0.5)(h1)
        h3 = Dense(HIDDEN3_UNITS, activation='relu')(h1d)
        a = Dense(1, activation='tanh', kernel_initializer=RandomNormal(mean=0.0, stddev=1e-4, seed=None))(h3)
        # b = Dense(1, activation='sigmoid', kernel_initializer=RandomNormal(mean=0.0, stddev=1e-4, seed=None))(h3)
        # V = concatenate([Action, Parameter_Acc1, Parameter_Acc2, Parameter_Time1, Parameter_Time2,
        #            Parameter_Time3, Parameter_Time4])
        # V = concatenate([a])
        # V = concatenate([a, b])
        # V = merge([Action, Parameter_Acc1, Parameter_Acc2, Parameter_Time1, Parameter_Time2,
        #            Parameter_Time3, Parameter_Time4], mode='concat')
        V = a
        # V = tf.concat(values=[a, b])
        model = Model(input=S, output=V)
        return model, model.trainable_weights, S
