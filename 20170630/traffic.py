#!/usr/bin/env python
from intersectionBoundary import drawIntersection
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from pos import state_table
import scipy.spatial
import math

av_pos = 0
n_degree = 36
gamma= 0.9
learning_rate = 0.01
display_step = 10

state_table(av_pos)
# Network Parameters
n_hidden_1 = 300  # 1st layer number of features
n_hidden_2 = 600  # 2nd layer number of features
state_size = n_degree
p_size = 1
action_size = 1
reward_size = 1


def linetrace(pos, n):
    r1 = 2 - (-4)
    r2 = 4 - pos
    r3 = 4 - 2
    rmax = 80
    theta = np.arange(0.001, np.pi, np.pi/n)
    dist = []
    x = 0
    y = 0
    for i in range(0, len(theta)):
        if pos <= 4:
            if theta[i] <= np.arctan(((-4) - pos)/(2-(-4))):
                dist.append(np.minimum(r1/np.cos(theta[i]),rmax))
                x = np.arange(2 - dist[i] * np.cos(theta[i]), 2, 0.01)
                y = - (x - 2) * np.tan(theta[i]) + pos
            else:
                if theta[i] > np.arctan(((-4) - pos)/(2-(-4))) and theta[i] <= np.arctan(((4) - pos)/(2-(-4))):
                    dist.append(np.minimum(r2/np.sin(theta[i]),rmax))
                    x = np.arange(2 - dist[i] * np.cos(theta[i]), 2, 0.01)
                    y = - (x - 2) * np.tan(theta[i]) + pos
                else:
                    if theta[i] > np.arctan(((4) - pos)/(2-(-4))) and theta[i] <= np.pi/2:
                        dist.append(np.minimum(r1 / np.cos(theta[i]),rmax))
                        x = np.arange(2 - dist[i] * np.cos(theta[i]), 2, 0.01)
                        y = - (x - 2) * np.tan(theta[i]) + pos
                    else:
                        if theta[i] > np.pi/2 and theta[i] <= np.pi/2 + np.arctan(4 - 2)/(4 - pos):
                            dist.append(np.minimum(r3/np.sin(theta[i] - np.pi/2),rmax))
                            x = np.arange(2, 2 - dist[i] * np.cos(theta[i]), 0.01)
                            y = - (x - 2) * np.tan(theta[i]) + pos
                        else:
                            if theta[i] > np.pi/2 + np.arctan(4 - 2)/(4 - pos) and theta[i] < np.pi/2 + np.arctan((4 - 2)/(-4 - pos)):
                                dist.append(np.minimum(r2 / np.cos(theta[i]-np.pi/2),rmax))
                                x = np.arange(2, 2 - dist[i] * np.cos(theta[i]), 0.01)
                                y = - (x - 2) * np.tan(theta[i]) + pos
                            else:
                                dist.append(np.minimum(r3 / np.sin(theta[i] - np.pi/2),rmax))
                                x = np.arange(2, 2 - dist[i] * np.cos(theta[i]), 0.01)
                                y = - (x - 2) * np.tan(theta[i]) + pos
        else:
            if theta[i] <= np.arctan(((4) - pos) / (2 - (-4))):
                dist.append(np.minimum(r2 / np.sin(theta[i]), rmax))
                x = np.arange(2 - dist[i] * np.cos(theta[i]), 2, 0.01)
                y = - (x - 2) * np.tan(theta[i]) + pos
            else:
                if theta[i] > np.arctan(((4) - pos) / (2 - (-4))) and theta[i] <= np.pi / 2:
                    dist.append(np.minimum(r1 / np.cos(theta[i]), rmax))
                    x = np.arange(2 - dist[i] * np.cos(theta[i]), 2, 0.01)
                    y = - (x - 2) * np.tan(theta[i]) + pos
                else:
                    if theta[i] > np.pi / 2 and theta[i] <= np.pi / 2 + np.arctan(4 - 2) / (4 - pos):
                        dist.append(np.minimum(r3 / np.sin(theta[i] - np.pi / 2), rmax))
                        x = np.arange(2, 2 - dist[i] * np.cos(theta[i]), 0.01)
                        y = - (x - 2) * np.tan(theta[i]) + pos
                    else:
                        dist.append(np.minimum(r2 / np.cos(theta[i] - np.pi / 2), rmax))
                        x = np.arange(2, 2 - dist[i] * np.cos(theta[i]), 0.01)
                        y = - (x - 2) * np.tan(theta[i]) + pos
        plt.plot(x, y, 'g')
    return dist


def dis_function(pos, n):
    drawIntersection()
    pos_f = tf.to_float(pos)
    state_dis = linetrace(pos_f, n)
    plt.plot(2, pos, 'r', marker='s',markersize=8)
    return state_dis


def get_state(pos, n):
    # x0 = 2*np.random.rand(1)
    state_dis = dis_function(pos, n)
    return state_dis #np.array(np.concatenate([x0, state_dis]), ndmin=2)


def get_next_state(pos, a, n):
    if a == 0:
        pos = tf.add(pos, 0.1*2)
        s = get_state(pos, n)
    else:
        pos = 4
        s = get_state(pos, n)
    return pos, s


def s_reward(s, n):
    return sum(1/n * np.pi * (s[0,:]**2))


def all_reward(s_, s, a, n):
    if s_reward(s_, n) < s_reward(s, n):
        reward = 1
    else:
        reward = -1
    if a == 1:
        reward += 5
    return reward


def policy_nn(x, weights, biases):
    """Create model."""
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    # Output layer with linear activation
    out_layer = tf.add(tf.matmul(layer_2, weights['out']), biases['out'])
    out_layer = tf.nn.sigmoid(out_layer)
    return out_layer


# Store layers weight & bias
policy_weights = {
    'h1': tf.Variable(tf.truncated_normal([state_size, n_hidden_1], stddev=0.1)),
    'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev=0.1)),
    'out': tf.Variable(tf.truncated_normal([n_hidden_2, action_size], stddev=0.1))
}

policy_biases = {
    'b1': tf.Variable(tf.constant(0.1, shape=[n_hidden_1])),
    'b2': tf.Variable(tf.constant(0.1, shape=[n_hidden_2])),
    'out': tf.Variable(tf.constant(0.1, shape=[action_size]))
}


s_data = tf.placeholder(tf.float32, [None, state_size])
p_data = tf.placeholder(tf.float32, [None, p_size])
a_data = tf.placeholder(tf.float32, [None, action_size])


pos_pred, state_pred = get_next_state(p_data, 0, n_degree)
state_pred = tf.squeeze(state_pred, [2])
pos_pred = tf.squeeze(pos_pred, [1])
action_pred = policy_nn(state_pred, policy_weights, policy_biases)
all_state = state_pred
all_action = action_pred
all_pos = pos_pred
s_ = get_state(p_data, n_degree)
reward = all_reward(s_, state_pred, action_pred, n_degree)
reward = tf.squeeze(reward, [1])
reward_ = np.power(gamma, 0)
i = 1
while pos_pred[0][1] < 4:
    pos_pred, state_pred = get_next_state(pos_pred, action_pred, n_degree)
    state_pred = tf.squeeze(state_pred, [2])
    pos_pred = tf.squeeze(pos_pred, [2])
    action_pred = policy_nn(state_pred, policy_weights, policy_biases)
    new_reward = all_reward(tf.gather(all_state, -1), state_pred, action_pred, n_degree)
    new_reward = tf.squeeze(new_reward,[1])
    reward = tf.concat(0, [reward, new_reward])
    reward_ = tf.concat(0, [reward_, np.power(gamma, i)])
    all_action = tf.concat(0, [all_action, action_pred])
    all_state = tf.concat(0,[all_state, state_pred])
    all_pos = tf.concat(0, [all_pos, pos_pred])
    i = tf.add(i, 1)
reward = tf.reduce_sum(- tf.multiply(reward, reward_))
train = tf.train.AdamOptimizer(learning_rate).minimize(reward)


# Before starting, initialize the variables.  We will 'run' this first.
init = tf.initialize_all_variables()

# Launch the graph.
sess = tf.Session()
sess.run(init)


step = 0
while True:
    state = get_state(av_pos, n_degree)
    sess.run(train, feed_dict={p_data: av_pos})
    if (step % display_step == 0):
        Q = sess.run(reward, feed_dict={p_data: av_pos})
        print("- Q function = {0}".format(Q))
        Action = sess.run(all_action, feed_dict={p_data: av_pos})
        State = sess.run(all_state, feed_dict={p_data: av_pos})
        Pos = sess.run(all_pos, feed_dict={p_data: av_pos})
    step += 1


# plt.show()