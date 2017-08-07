#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from pos import state_table
from pos import get_state


av_pos = -10
n_degree = 36
gamma= 0.9
learning_rate = 0.01
display_step = 10

all_pos, all_state, all_state_reward = state_table(av_pos)
all_pos = tf.constant(all_pos)
all_state = tf.constant(all_state)
all_state_reward = tf.constant(all_state_reward)


# Network Parameters
n_hidden_1 = 300  # 1st layer number of features
n_hidden_2 = 600  # 2nd layer number of features
state_size = n_degree
p_size = 1
action_size = 1
reward_size = 1


def find_index(x, A):
    a = tf.equal(tf.reduce_sum(A - x, 1), tf.constant(0, dtype=tf.float64))
    b = tf.where(a)
    # b = b[0]
    return b


def get_next_state(s, a):
    action_zeros = tf.less_equal(a, 0.5)
    index = find_index(s, all_state)
    index = index[0]
    next_s = tf.cond(action_zeros, all_state[[index+1],:], all_state[[-1],:])
    # if a == 0:
    #     pos = tf.add(pos, 0.1*2)
    #     s = get_state(pos, n)
    # else:
    #     pos = 4
    #     s = get_state(pos, n)
    return next_s


def s_reward(s):
    index = find_index(s, all_state)
    index = index[0]
    return all_state_reward[index, :]


def a_reward(s_, s, a):
    condition1 = tf.less(s_reward(s_), s_reward(s))
    reward = tf.cond(condition1, 1, -1)
    condition2 = tf.greater_equal(a, 0.5)
    reward = tf.cond(condition2, reward + 5, reward)
    # if s_reward(s_) < s_reward(s):
    #     reward = 1
    # else:    # if a == 1:
    #     reward += 5
    #     reward = -1
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


def training_loop(state_pred, action_pred, state_stack, action_stack, reward_stack, reward_, i):
    pre_state = state_pred
    state_pred = get_next_state(state_pred, action_pred)
    action_pred = policy_nn(state_pred, policy_weights, policy_biases)
    new_reward = a_reward(pre_state, state_pred, action_pred)
    state_stack = tf.concat(0, [state_stack, state_pred])
    action_stack = tf.concat(0, [action_stack, action_pred])
    reward_stack = tf.concat(0, [reward_stack, new_reward])
    reward_ = tf.concat(0, [reward_, np.power(gamma, i)])
    i = tf.add(i, 1)
    return state_pred, action_pred, state_stack, action_stack, reward_stack, reward_, i


s_data = tf.placeholder(tf.float64, [None, state_size])
a_data = tf.placeholder(tf.float64, [None, action_size])


state_pred = get_next_state(s_data, 0)
# state_pred = tf.squeeze(state_pred, [2])
# pos_pred = tf.squeeze(pos_pred, [1])
action_pred = policy_nn(state_pred, policy_weights, policy_biases)
reward = a_reward(s_data, state_pred, action_pred)
state_stack = state_pred
action_stack = action_pred
reward_stack = reward
# all_pos = pos_pred
# reward = tf.squeeze(reward, [1])
reward_ = np.power(gamma, 0)
i = tf.constant(1)
if_exit = tf.not_equal(tf.shape(find_index(state_pred, all_state))[0], 0)
train_result = tf.while_loop(if_exit, training_loop, [state_pred, action_pred, state_stack, action_stack, reward_stack, reward_, i])
# while pos_pred[0][1] < 4:
#     state_pred = get_next_state(pos_pred, action_pred)
#     # state_pred = tf.squeeze(state_pred, [2])
#     # pos_pred = tf.squeeze(pos_pred, [2])
#     # action_pred = policy_nn(state_pred, policy_weights, policy_biases)
#     # new_reward = all_reward(tf.gather(all_state, -1), state_pred, action_pred, n_degree)
#     # new_reward = tf.squeeze(new_reward,[1])
#     # reward = tf.concat(0, [reward, new_reward])
#     # reward_ = tf.concat(0, [reward_, np.power(gamma, i)])
#     all_action = tf.concat(0, [all_action, action_pred])
#     all_state = tf.concat(0,[all_state, state_pred])
#     all_pos = tf.concat(0, [all_pos, pos_pred])
#     i = tf.add(i, 1)
# reward = tf.reduce_sum(- tf.multiply(reward, reward_))
output = tf.reduce_sum(- tf.multiply(train_result[4], train_result[5]))
train = tf.train.AdamOptimizer(learning_rate).minimize(output)


# Before starting, initialize the variables.  We will 'run' this first.
init = tf.initialize_all_variables()

# Launch the graph.
sess = tf.Session()
sess.run(init)


step = 0
while True:
    state = get_state(av_pos, n_degree)
    sess.run(train, feed_dict={s_data: state, a_data: 0})
    if (step % display_step == 0):
        Q = sess.run(reward, feed_dict={s_data: state, a_data: 0})
        print("- Q function = {0}".format(Q))
        # Action = sess.run(action_stack, feed_dict={p_data: av_pos})
        # State = sess.run(all_state, feed_dict={p_data: av_pos})
        # Pos = sess.run(all_pos, feed_dict={p_data: av_pos})
    step += 1