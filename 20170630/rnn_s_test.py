#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import math
#import csv
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#import matlab.engine
#import keras


# Parameters
tau = 0.1
learning_rate = 0.01
training_epochs = 20
batch_size = 1
display_step = 200
error_threshold = 20
action_comfort0 = 3
action_comfort1 = 0.1
v_comfort = 10

num_human_vehicle =1
#time_step = 30
ini_min_y = -5.5
ini_max_y = 2
ini_min_x = -2
ini_max_x = 6
mu1 = 1
mu2 = 50
gamma = 0.90
pi = tf.constant(np.pi)/2

# Network Parameters
n_hidden_1 = 300  # 1st layer number of features
n_hidden_2 = 600  # 2nd layer number of features
n_hidden_3 = 600  # 1st layer number of features
n_hidden_4 = 900
n_hidden_5 = 600  # 1st layer number of features
n_hidden_6 = 300
state_size = 6
action_size = 2
reward_size = 1


def gather_cols(params, indices, name=None):
    """Gather columns of a 2D tensor.

    Args:
        params: A 2D tensor.
        indices: A 1D tensor. Must be one of the following types: ``int32``, ``int64``.
        name: A name for the operation (optional).

    Returns:
        A 2D Tensor. Has the same type as ``params``.
    """
    with tf.name_scope(name, "gather_cols",[params, indices]) as scope:
        # Check input
        params = tf.convert_to_tensor(params, name="params")
        indices = tf.convert_to_tensor(indices, name="indices")
        try:
            params.get_shape().assert_has_rank(2)
        except ValueError:
            raise ValueError('\'params\' must be 2D.')
        try:
            indices.get_shape().assert_has_rank(1)
        except ValueError:
            raise ValueError('\'params\' must be 1D.')

        # Define op
        p_shape = tf.shape(params)
        p_flat = tf.reshape(params, [-1])
        i_flat = tf.reshape(tf.reshape(tf.range(0, p_shape[0]) * p_shape[1],
                                       [-1, 1]) + indices, [-1])
        return tf.reshape(tf.gather(p_flat, i_flat),
                          [p_shape[0], -1])


def get_init_state(batch_size, num_human_vehicle):
    x0_in = 2 * np.ones([batch_size, num_human_vehicle])
    x1_in = -6 * np.ones([batch_size, num_human_vehicle])
    x2_in = 2* np.random.rand(batch_size, num_human_vehicle).astype(np.float32)
    x3_in = math.pi/2 * np.ones([batch_size, num_human_vehicle])
    x4_in = -50 * np.ones([batch_size, num_human_vehicle])
    x5_in = 20 * np.ones([batch_size, num_human_vehicle])
    return np.concatenate([x0_in, x1_in, x2_in, x3_in, x4_in, x5_in], axis=1)


def get_next_state(x, a, tau, batch_size):
    #human_action = np.random.rand(batch_size, 1).astype(np.float32) - 0.5
    human_action = tf.zeros([batch_size,1], tf.float32)
    y0_in = tf.add(gather_cols(x, [0]), tau * tf.multiply(gather_cols(x, [2]), tf.cos(gather_cols(x, [3]))))
    y1_in = tf.add(gather_cols(x, [1]), tau * tf.multiply(gather_cols(x, [2]), tf.maximum(tf.sin(gather_cols(x, [3])),0)))
    y2_in = tf.maximum(tf.add(gather_cols(x, [2]), tau * gather_cols(a, [0])), 0)
    y3_in = tf.add(gather_cols(x, [3]), tau * gather_cols(a, [1]))
    y4_in = gather_cols(x, [4])#tf.add(gather_cols(x, [4]), tau * gather_cols(x, [5]))
    y5_in = 0*gather_cols(x, [3])#gather_cols(x, [5]) + tau * human_action
    return tf.stack([y0_in, y1_in, y2_in, y3_in, y4_in, y5_in], axis=1)


def policy_nn(x, weights, biases):
    """Create model."""
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    # # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)

    # layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    # layer_3 = tf.nn.sigmoid(layer_3)
    # # # Hidden layer with RELU activation
    # layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    # layer_4 = tf.nn.sigmoid(layer_4)
    #
    # layer_5 = tf.add(tf.matmul(layer_4, weights['h5']), biases['b5'])
    # layer_5 = tf.nn.sigmoid(layer_5)
    # # # Hidden layer with RELU activation
    # layer_6 = tf.add(tf.matmul(layer_5, weights['h6']), biases['b6'])
    # layer_6 = tf.nn.sigmoid(layer_6)


    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


def action_threshold(x, a, n):
    t = - tf.maximum((gather_cols(a, [0]) - 3 * tf.ones([n, 1], dtype=tf.float32)),
                     tf.zeros([n, 1], dtype=tf.float32))\
        -tf.maximum(v_comfort* tf.ones([n, 1], dtype=tf.float32) - gather_cols(x, [2]), 0)
        #- tf.maximum(gather_cols(a, [1]) - 1 * tf.ones([n, 1], dtype=tf.float32), tf.zeros([n, 1], dtype=tf.float32))
    return t


def max_velocity(x):
    t = mu1 * gather_cols(x,[2]) # mu1 * (tf.multiply(gather_cols(x,[2]),tf.sin(pi))- tf.multiply(gather_cols(x,[2]),tf.cos(pi)))
    return t


def no_crash(x,n):
    t = - mu2 * tf.multiply(tf.multiply(tf.multiply(tf.maximum((gather_cols(x,[1]) - ini_min_y * tf.ones([n, 1],dtype=tf.float32)), 0),
                                                    tf.maximum(ini_max_y * tf.ones([n, 1],dtype=tf.float32) - gather_cols(x,[1]), 0)),
                                        tf.maximum(gather_cols(x,[4]) - ini_min_x * tf.ones([n, 1],dtype=tf.float32), 0)),
                            tf.maximum((6*tf.ones([n, 1],dtype=tf.float32) - gather_cols(x,[4])), 0))
    return t


def follow(x,n):
    ##GS
    #t = - 10 * tf.maximum(tf.abs(gather_cols(x, [0]) - 2* tf.ones([n,1])),0)
    ## TL
    t1 = - 10 * tf.maximum(tf.abs(gather_cols(x, [1]) - 2 * tf.ones([n, 1])), 0)
    t2 = - 10 * tf.maximum((gather_cols(x, [0]) + 6 * tf.ones([n, 1])), 0)
    return t1 + t2

def get_loss_function(x, a, n):
    # action_threshold = - tf.maximum((gather_cols(a,[0]) - 3*tf.ones([n, 1],dtype=tf.float32)),
    #                                 tf.zeros([n, 1],dtype=tf.float32))\
    #                    - tf.maximum(gather_cols(a, [1])- 1*tf.ones([n, 1],dtype=tf.float32), tf.zeros([n, 1], dtype=tf.float32))
    # max_velocity = tf.log(tf.multiply(gather_cols(x,[2]),tf.sin(pi))\
    #                - tf.multiply(gather_cols(x,[2]),tf.cos(pi)))
    # less_time = 0 # - mu1 * gather_cols(x,[4])
    # no_crash = - mu2 * tf.multiply(tf.multiply(tf.multiply(tf.maximum((gather_cols(x,[1]) - ini_min_y * tf.ones([n, 1],dtype=tf.float32)), 0),
    #                               tf.maximum(ini_max_y * tf.ones([n, 1],dtype=tf.float32) - gather_cols(x,[1]), 0)),
    #                               tf.maximum(gather_cols(x,[1]) - ini_min_x * tf.ones([n, 1],dtype=tf.float32), 0)),
    #                               tf.maximum((6*tf.ones([n, 1],dtype=tf.float32) - gather_cols(x,[1])), 0))
    reward = action_threshold(x, a, n) + max_velocity(x) + follow(x,n) #no_crash
    return -reward


# def draw_picture(state):
#     fig = plt.figure()
#     ax = plt.axes(xlim=(-50, 50), ylim=(-10, 10))
#     particles, = ax.plot([], [], 'bo', lw=2)
#
#     def init():
#         # initiate animation
#         particles.set_data([], [])
#         return particles
#
#     def animate(i):
#         y = []
#         x = []
#         y.append(state[i, 0])
#         x.append(state[i, 1])
#         particles.set_data(x, y)
#         return particles
#
#     anim = animation.FuncAnimation(fig, animate, np.arange(0, len(state)),
#                                    interval=25, init_func=init, repeat=False)
#     plt.show()


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

x_data = tf.placeholder(tf.float32, [None, state_size])
a_data = tf.placeholder(tf.float32, [None, action_size])
state_all_data = tf.placeholder(tf.float32, [None, state_size])
a_all_data = tf.placeholder(tf.float32, [None, action_size])
r_all_data = tf.placeholder(tf.float32, [None, reward_size])




def find_index(x, A):
    a = tf.equal(tf.reduce_sum(A - x, 1), tf.constant(0, dtype=tf.float32))
    b = tf.where(a)
    # b = b[0]
    return b

a = tf.constant(np.array([[2,2],[2,3],[4,3], [5,3]]), dtype=tf.float32)
b = tf.constant(np.array([2,2], ndmin=2), dtype=tf.float32)
c = tf.equal(tf.reduce_sum(a-b, 1),0)
# d = tf.where(c)
# d = d[0]
d = find_index(b, a)[0]
# d = d[0]
# t = a[[d[0]+1],:]
t = tf.add(d,tf.constant([1], dtype=tf.int64))
# d = tf.squeeze(d,
# [0])

# Construct model
x0 = get_init_state(batch_size, num_human_vehicle)
#time_step = int((6-x0[0,4])/x0[0,5] * 10) + 1
time_step = int(np.sqrt([2*8/action_comfort0]) /tau) + 1
action_pred = policy_nn(x_data, policy_weights, policy_biases)
next_state_pred = get_next_state(x_data, action_pred, tau, batch_size)
next_state_pred = tf.squeeze(next_state_pred, [2])
loss_ = np.logspace(0, time_step - 1, time_step, base=gamma)
loss = get_loss_function(next_state_pred, action_pred, 1)
loss = tf.squeeze(loss, [1])
all_action = action_pred
all_state = next_state_pred
for t in range(1, time_step):
    new_action = policy_nn(next_state_pred, policy_weights, policy_biases)
    next_state_pred = get_next_state(next_state_pred, new_action, tau, batch_size)
    next_state_pred = tf.squeeze(next_state_pred, [2])
    new_loss = get_loss_function(next_state_pred, new_action,1)
    new_loss = tf.squeeze(new_loss,[1])
    loss = tf.concat(0, [loss, new_loss])
    all_action = tf.concat(0, [all_action, new_action])
    all_state = tf.concat(0,[all_state, next_state_pred])
loss = tf.reduce_sum(tf.multiply(loss,loss_))
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Minimize the mean squared errors.
#loss = tf.reduce_mean(get_loss_function(x_data, a_data), 1)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate)
#train = optimizer.minimize(loss)
#train = tf.train.AdamOptimizer(1e-4).minimize(loss)

# Before starting, initialize the variables.  We will 'run' this first.
init = tf.initialize_all_variables()

# Launch the graph.
sess = tf.Session()
sess.run(init)


cc = sess.run(c)
dd = sess.run(d)
print(dd)
tt = sess.run(t)
print(tt)
print(dd)


epo = 0
while True:
    action = np.zeros([1, 2], dtype=np.float32)
    #for t in range(1, time_step):
    #    new_action = sess.run(action_pred, feed_dict={x_data: next_state})
    #    action = np.vstack([action, new_action])
    #    next_state = sess.run(next_state_pred, feed_dict={x_data: next_state, a_data: action[[t], :]})
    #    next_state = np.squeeze(next_state, axis=(2,))
    #    state = np.vstack([state, next_state])
    #print(np.shape(action))
    #print(np.shape(state))
    step = 0
    qc = []
    while True:
        state = get_init_state(batch_size, num_human_vehicle)
        sess.run(train, feed_dict={x_data: state})
        if (step % display_step == 0):
            Q = sess.run(loss, feed_dict={x_data: state})
            print("Q function = {0}".format(Q))
            Action = sess.run(all_action, feed_dict={x_data: state})
            State = sess.run(all_state, feed_dict={x_data: state})
            print(epo)
            print(step/200)
            print(max(State[:,2]))
            #print(State[:, 1])
            #draw_picture(State)
            sts = "/home/zhiqian/Documents/tensorflow/save_data/state_%d_%d.csv" % (epo, step/200)
            acs = "/home/zhiqian/Documents/tensorflow/save_data/action_%d_%d.csv" % (epo, step/200)
            np.savetxt(sts, State, delimiter=',')
            np.savetxt(acs, Action, delimiter=',')
            if State[-1, 1] > 2 and max(State[:,2]) < v_comfort:
                break
        step += 1
    if epo > training_epochs:
        break
    epo += 1