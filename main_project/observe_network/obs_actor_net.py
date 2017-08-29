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
    def __init__(self, sess, non_his_size, his_len, his_size, action_size, batch_size, sigma, learn_rate):
        self.sess = sess
        self.BATCH_SIZE = batch_size
        self.TAU = sigma
        self.LEARNING_RATE = learn_rate

        keras.set_session(sess)

        # Now create the model
        self.model, self.weights, self.state = self.create_actor_network(non_his_size, his_len, his_size)
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(non_his_size, his_len, his_size)
        self.action_gradient = tf.placeholder(tf.float32, [None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(learn_rate).apply_gradients(grads)
        self.sess.run(tf.global_variables_initializer())

    def train(self, non_his, his, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: [non_his, his],
            self.action_gradient: action_grads})

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU) * actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    @staticmethod
    def create_actor_network(non_his_size, his_len, his_size):
        logging.info('...... Building actor model ......')
        non_hist_state = Input(shape=(non_his_size,))
        non1 = Dense(HIDDEN2_UNITS, activation='relu')(non_hist_state)
        non2 = Dense(HIDDEN3_UNITS, activation='sigmoid')(non1)
        non3 = Dense(HIDDEN4_UNITS, activation='relu')(non2)

        hist_state = Input(shape=(his_len, his_size,))
        # hist0 = keras.reshape(hist_state, (-1, his_len, his_size))
        hist1 = LSTM(HIDDEN1_UNITS, return_sequences=True)(hist_state)
        hist2 = Dropout(0.2)(hist1)
        hist3 = LSTM(HIDDEN2_UNITS, return_sequences=False)(hist2)
        hist4 = Dropout(0.2)(hist3)

        merged = concatenate([hist4, non3])
        merged1 = Dense(HIDDEN3_UNITS, activation='linear')(merged)
        merged2 = Dense(HIDDEN5_UNITS, activation='relu')(merged1)
        action = Dense(1, activation='tanh', kernel_initializer=RandomNormal(mean=0.0, stddev=1e-4, seed=None))(merged2)

        model = Model(input=[non_hist_state, hist_state], outputs=action)
        return model, model.trainable_weights, [non_hist_state, hist_state]
