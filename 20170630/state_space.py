#!/usr/bin/env python
from intersectionBoundary import drawIntersection
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from random import randint
import random

#pos_x = 2
sce = randint(0,2)
pos_x = sce * 48 + 2
fv_pos = random.random() * 25 - 30
fv_pos_x = np.random.normal(pos_x, 0.1)
av_pos = -30
no_v = 10
hv_pos_f = np.random.uniform(-164, pos_x - 6, [1,no_v])
hv_pos_fy = np.random.normal(-2, 0.1, no_v)
hv_pos_b = np.random.uniform(pos_x + 2, 260, [1,no_v])
hv_pos_by = np.random.normal(2, 0.1, no_v)
n_degree = 30
gamma= 0.9
learning_rate = 0.01
display_step = 30
tau = 0.1
vel_fv = 10
vel_av = 10
vel_hv = 15
rx = 2
lx = 6
app_inter_ypos = -4
pass_inter_ypos = 4
resol = 2*np.pi/n_degree

# Network Parameters
n_hidden_1 = 300  # 1st layer number of features
n_hidden_2 = 600  # 2nd layer number of features
state_size = n_degree
p_size = 1
action_size = 1
reward_size = 1
epi = 0.001


def linetrace(pos, hull, f_pos, h_posf, h_posb):
    rmax = 80
    theta = np.arange(0, 2.0*np.pi, resol)
    min_dist = []
    x_point = np.append(hull[:,0], fv_pos_x)
    x_point = np.append(x_point, h_posf[0])
    x_point = np.append(x_point, h_posb[0])
    y_point = np.append(hull[:, 1], f_pos)
    y_point = np.append(y_point, hv_pos_fy)
    y_point = np.append(y_point, hv_pos_by)

    all_points = np.vstack((x_point, y_point)).T
    # plt.plot(all_points[:, 1], all_points[:, 0], 'k')
    # plt.draw()
    for i in range(0, len(theta)):
        select_points = all_points[np.logical_and((all_points[:,0] - pos_x) * np.sin(theta[i]) -
                                                  (all_points[:,1] - pos) * np.cos(theta[i]) > 0,
                                                  (all_points[:, 0] - pos_x) * np.sin(theta[i] + resol) - (
                                                  all_points[:, 1] - pos) * np.cos(theta[i] + resol) < 0)]
        node = np.array([pos_x, pos])
        dist_2 = np.sum((select_points - node)**2, axis=1)
        if not len(dist_2):
            min_dist.append(rmax)
        else:
            dist = np.sqrt(dist_2)
            nn = np.argmin(dist)
            plt.plot(select_points[nn,0], select_points[nn,1], 'r*')
            min_dist.append(min(dist))
        x = np.array([pos_x, pos_x - min_dist[i] * np.cos(theta[i])])
        y = [pos, pos - min_dist[i] * np.sin(theta[i])]
        plt.plot(x, y, 'k')
        plt.draw()
    return min_dist


def dis_function(pos, fpos, hv_posf, hv_posb):
    hull = drawIntersection()
    if sce == 0:
        hull = np.vstack((hull[0], hull[1], hull[2], hull[3]))
    else:
        if sce == 1:
            hull = np.vstack((hull[2], hull[3], hull[4], hull[5]))
        else:
            hull = np.vstack((hull[4], hull[5], hull[6], hull[7]))
    plt.draw()
    state_dis = linetrace(pos, hull, fpos, hv_posf, hv_posb)
    plt.plot(pos_x, pos, 'r.')
    plt.plot(pos_x, fpos, 'g.')
    plt.plot(hv_posb, 2 * np.ones([1,no_v]), 'g.')
    plt.plot(hv_posf, -2 * np.ones([1,no_v]), 'g.')
    plt.draw()
    return state_dis


def get_state(vel, pos, fpos, hv_posf, hv_posb):
    # x0 = 2*np.random.rand(1)
    state_dis = dis_function(pos, fpos, hv_posf, hv_posb)
    state_all = np.hstack((vel, state_dis, pos, fpos, max(hv_posf[hv_posf <= (pos_x + 2)]), min(hv_posb[hv_posb >= (pos_x - 2)])))
    # print(state_all[1:n_degree + 1] > 1.5)
    return state_all #np.array(np.concatenate([x0, state_dis]), ndmin=2)


def get_next_state(pos, a, n):
    if a == 0:
        pos = tf.add(pos, 0.1*2)
        s = get_state(pos, n, fv_pos)
    else:
        pos = pass_inter_ypos
        s = get_state(pos, n, fv_pos)
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

def reward_fun(s):
    # print(s[0][1:n_degree])
    # print(np.log(np.maximum(s[0][1:n_degree]-1, 0)))
    # print(sum(np.log(np.maximum(s[0][1:n_degree]-1, 0))))
    # print(s[0][s[0][1:n_degree + 1] > 1.5])
    good_r = len(s[0][s[0][1:n_degree + 1] > 1.5])
    bad_r = 100 * len(s[0][s[0][1:n_degree + 1] <= 1.5])
    # print(good_r)
    reward = - bad_r
    # reward = 0.1*s[0][0] + sum(np.log(np.maximum(s[0][1:n_degree]-1, 0) + 0.001))
    return reward


def current_state(vel, pos, fpos, hv_posf, hv_posb):
    plt.ion()
    new_state = np.array(get_state(vel, pos, fpos, hv_posf, hv_posb), ndmin=2)
    new_reward = reward_fun(new_state) #s_reward(new_state, n_degree)
    plt.pause(0.01)
    plt.clf()
    return new_state, new_reward


def state_table(vel, av_pos, fv_pos, hv_posf, hv_posb):
    plt.ion()
    state = np.array(get_state(vel, av_pos,fv_pos, hv_posf, hv_posb), ndmin=2)
    state_reward = reward_fun(state)
        # np.array(get_state(av_pos,fv_pos, hv_posf, hv_posb), ndmin=2)
    # state_reward = [s_reward(state, n_degree)]
    pos = av_pos
    i = 1
    while av_pos <= pass_inter_ypos:
        fv_pos += tau * vel_fv
        av_pos += tau * vel_av
        hv_posb -= tau * vel_hv
        hv_posf += tau * vel_hv
        new_state = np.array(get_state(vel, av_pos,fv_pos, hv_posf, hv_posb), ndmin=2)
        new_reward = reward_fun(new_state)
        # new_reward = s_reward(new_state, n_degree)
        # print(new_reward)
        # print(state_reward)
        state = np.vstack((state, new_state))
        state_reward = np.vstack((state_reward, new_reward))
        pos = np.vstack((pos, av_pos))
        plt.show()
        plt.pause(0.01)
        plt.clf()
        i += 1
    return pos, state, state_reward

pos, state, state_reward = state_table(vel_av, av_pos, fv_pos,  hv_pos_f, hv_pos_b)