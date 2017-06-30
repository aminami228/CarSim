#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from pos import state_table
from pos import current_state


av_pos = -10
n_degree = 36
gamma= 0.9
learning_rate = 0.01
display_step = 10

all_pos, all_state, all_state_reward = state_table(av_pos)
all_pos_tf = tf.constant(all_pos)
final_index = tf.constant(len(all_state))-1
all_state_tf = tf.constant(all_state, dtype=tf.float32)
all_state_reward_tf = tf.constant(all_state_reward, dtype=tf.float32)


# Network Parameters
n_hidden_1 = 300  # 1st layer number of features
n_hidden_2 = 600  # 2nd layer number of features
state_size = n_degree
p_size = 1
action_size = 1
reward_size = 1


def find_index(x, A):
    a = tf.equal(tf.reduce_sum(A - x, 1), tf.constant(0, dtype=tf.float32))
    b = tf.where(a)
    # b = b[0]
    return b


def get_next_state(s, a):
    action_zeros = tf.less_equal(a, tf.constant(0.5))
    index = find_index(s, all_state)[0]
    index_val = index[0]
    index_val += 1
    def s1(): return tf.gather(all_state_tf, [index_val])
    def s2(): return tf.gather(all_state_tf, [final_index])
    next_s = tf.cond(action_zeros, s1, s2)
    return next_s


def s_reward(s):
    index = find_index(s, all_state)[0]
    return tf.gather(all_state_reward_tf, index)


def a_reward(s_, s, a):
    condition1 = tf.less(s_reward(s_), s_reward(s))[0]
    condition1 = condition1[0]
    def r1(): return tf.constant(1.0)
    def r2(): return tf.constant(0.0)
    reward = tf.cond(condition1, r1, r2)
    condition2 = tf.greater_equal(a, 0.5)
    def rb1(): return reward+5
    def rb2(): return reward
    reward = tf.cond(condition2, rb1, rb2)
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


def action_pred(s):
    a = policy_nn(s, policy_weights, policy_biases)[0]
    a = a[0]
    action_zeros = tf.less_equal(a, tf.constant(0.5))
    def a0(): return tf.constant(0.0)
    def a1(): return tf.constant(1.0)
    a = tf.cond(action_zeros, a0, a1)
    return a


# def training_loop(next_state, a_data, state_stack, action_stack, reward_stack, i):
#     pre_state = next_state
#     next_state = get_next_state(next_state, a_data)
#     a_data = action_pred(next_state)
#     new_reward = a_reward(pre_state, next_state, a_data)
#     state_stack = tf.concat(0, [state_stack, next_state])
#     action_stack = tf.concat(0, [action_stack, a_data])
#     reward_stack = tf.concat(0, [reward_stack, new_reward])
#     i += 1
#     return next_state, action_pred, state_stack, action_stack, reward_stack, reward_


s_data = tf.placeholder(tf.float32, [None, state_size])
# state_stack = s_data
a1_data = action_pred(s_data)
# action_stack = a_data
next_state = get_next_state(s_data, a1_data)
a_data = action_pred(s_data)
reward = a_reward(s_data, next_state, a_data)
i = 0
# state_stack = next_state
# action_stack = a_data
# reward_stack = reward

# def if_exit(n, a): return tf.not_equal(tf.shape(find_index(n, a))[0], 0)
# train_result = tf.while_loop(if_exit(next_state, all_state), training_loop, [next_state, a_data, state_stack, action_stack, reward_stack, i])
reward_ = np.logspace(0, i - 1, i, base=gamma)


# Before starting, initialize the variables.  We will 'run' this first.
init = tf.global_variables_initializer()

# Launch the graph.
sess = tf.Session()
sess.run(init)


epo = 0
while True:
    state = np.array(all_state[0, :], ndmin=2)
    R_p = 0
    while True:
        R = sess.run(reward, feed_dict={s_data: state})
        A = sess.run(a_data, feed_dict={s_data: state})
        print(R)
        print(A)
        if R <= R_p:
            break
        R_p = R
        state = sess.run(next_state, feed_dict={s_data: state})
    epo += 1
    print(epo)

    # if (step % display_step == 0):
    #     Q = sess.run(reward, feed_dict={s_data: state})
    #     print("- Q function = {0}".format(Q))
    #     # Action = sess.run(action_stack, feed_dict={p_data: av_pos})
    #     # State = sess.run(all_state, feed_dict={p_data: av_pos})
    #     # Pos = sess.run(all_pos, feed_dict={p_data: av_pos})
    # step += 1