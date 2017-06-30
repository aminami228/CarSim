#!/usr/bin/env python
from intersectionBoundary import drawIntersection
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from random import randint
import random

#pos_x = 2
pos_x = randint(0,2) * 48 + 2
fv_pos = random.random() * 25 - 30
av_pos = -50
no_v = 10
hv_pos_f = np.random.uniform(-64, pos_x - 6, [1,no_v])
hv_pos_b = np.random.uniform(pos_x + 2, 160, [1,no_v])
n_degree = 30
gamma= 0.9
learning_rate = 0.01
display_step = 10
tau = 0.1
vel_fv = 10
vel_av = 10
vel_hv = 15
rx = 2
lx = 6
app_inter_ypos = -4
pass_inter_ypos = 4

# Network Parameters
n_hidden_1 = 300  # 1st layer number of features
n_hidden_2 = 600  # 2nd layer number of features
state_size = n_degree
p_size = 1
action_size = 1
reward_size = 1
epi = 0.001


def linetrace(pos, n, fpos):
    rmax = 80
    theta = np.arange(-np.pi/6, 7/6*np.pi, 4/3*np.pi/n)
    dist = []
    for i in range(0, len(theta)):
        if pos <= app_inter_ypos:
            if theta[i] <= np.arctan((app_inter_ypos - pos)/(rx)):
                dist.append(np.minimum(rx/np.cos(theta[i]),rmax))
                x = np.arange(pos_x, pos_x + dist[i] * np.cos(theta[i]), 0.01)
                y = (x - pos_x) * np.tan(theta[i]) + pos
            else:
                if theta[i] > np.arctan((app_inter_ypos - pos)/(rx)) and theta[i] <= np.arctan((pass_inter_ypos - pos)/(rx)):
                    dist.append(np.minimum((pass_inter_ypos - pos) / np.sin(theta[i]), rmax))
                    x = np.arange(pos_x, pos_x + dist[i] * np.cos(theta[i]), 0.01)
                    y = (x - pos_x) * np.tan(theta[i]) + pos
                else:
                    if theta[i] > np.arctan((pass_inter_ypos - pos)/(rx)) and theta[i] < np.pi/2 - epi:
                        if_np_front_vehicle = np.minimum(rx / np.cos(theta[i]), rmax)
                        dist.append(np.minimum(if_np_front_vehicle, (fpos - pos)))
                        x = np.arange(pos_x, pos_x + dist[i] * np.cos(theta[i]), 0.01)
                        y = (x - pos_x) * np.tan(theta[i]) + pos
                    else:
                        if theta[i] >= np.pi/2 - epi and theta[i] <= np.pi/2 + epi:
                            if_np_front_vehicle = rmax
                            dist.append(np.minimum(if_np_front_vehicle, (fpos - pos)))
                            y = np.arange(pos, pos + dist[i], 0.01)
                            x = pos_x * np.ones(len(y))
                        else:
                            if theta[i] > np.pi/2 and theta[i] <= np.pi/2 + np.arctan(lx/(pass_inter_ypos - pos)):
                                dist.append(np.minimum(lx/np.sin(theta[i] - np.pi/2),rmax))
                                x = np.arange(pos_x - dist[i] * np.cos(np.pi - theta[i]), pos_x, 0.01)
                                y = (pos_x - x) * np.tan(np.pi - theta[i]) + pos
                            else:
                                if theta[i] > np.pi/2 + np.arctan(lx/(pass_inter_ypos - pos)) and \
                                                theta[i] < np.pi - (np.arctan((app_inter_ypos - pos)/lx)):
                                    dist.append(np.minimum((pass_inter_ypos - pos) / np.cos(theta[i]-np.pi/2),rmax))
                                    x = np.arange(pos_x + dist[i] * np.cos(theta[i]), pos_x, 0.01)
                                    y = (pos_x - x) * np.tan(np.pi - theta[i]) + pos
                                else:
                                    dist.append(np.minimum(lx / np.cos(np.pi - theta[i]),rmax))
                                    x = np.arange(pos_x - dist[i] * np.cos(np.pi - theta[i]), pos_x, 0.01)
                                    y = (pos_x - x) * np.tan(np.pi - theta[i]) + pos
        else:
            if theta[i] <= -np.arctan((pos - app_inter_ypos)/(rx)):
                dist.append(np.minimum(rx / np.cos(theta[i]), rmax))
                x = np.arange(pos_x, pos_x + dist[i] * np.cos(theta[i]), 0.01)
                y = (x - pos_x) * np.tan(theta[i]) + pos
            else:
                if theta[i] > -np.arctan((pos - app_inter_ypos)/(rx)) and theta[i] < 0:
                    dist.append(np.minimum((pos - app_inter_ypos) / np.sin(-theta[i]), rmax))
                    x = np.arange(pos_x, pos_x + dist[i] * np.cos(theta[i]), 0.01)
                    y = (x - pos_x) * np.tan(theta[i]) + pos
                else:
                    if theta[i] == 0:
                        dist.append(rmax)
                        x = np.arange(pos_x, pos_x + rmax, 0.01)
                        y = pos * np.ones([1, len(x)])
                    else:
                        if theta[i] > 0 and theta[i] <= np.arctan((pass_inter_ypos - pos)/(rx)):
                            dist.append(np.minimum((pass_inter_ypos - pos) / np.sin(theta[i]), rmax))
                            x = np.arange(pos_x, pos_x + dist[i] * np.cos(theta[i]), 0.01)
                            y = (x - pos_x) * np.tan(theta[i]) + pos
                        else:
                            if theta[i] > np.arctan((pass_inter_ypos - pos) / (rx)) and theta[i] < np.pi / 2 - epi:
                                dist.append(np.minimum(rx / np.cos(theta[i]), rmax))
                                x = np.arange(pos_x, pos_x + dist[i] * np.cos(theta[i]), 0.01)
                                y = (x - pos_x) * np.tan(theta[i]) + pos
                            else:
                                if theta[i] >= np.pi / 2 - epi and theta[i] <= np.pi/2 + epi:
                                    dist.append(rmax)
                                    y = np.arange(pos, pos + rmax, 0.01)
                                    x = pos_x * np.ones(len(y))
                                else:
                                    if theta[i] > np.pi / 2 and \
                                                    theta[i] <= np.pi / 2 + np.arctan(lx / (pass_inter_ypos - pos)):
                                        dist.append(np.minimum(lx / np.sin(theta[i] - np.pi / 2), rmax))
                                        x = np.arange(pos_x - dist[i] * np.cos(np.pi - theta[i]), pos_x, 0.01)
                                        y = (pos_x - x) * np.tan(np.pi - theta[i]) + pos
                                    else:
                                        if theta[i] > np.pi / 2 + np.arctan(lx / (pass_inter_ypos - pos)) and theta[i] < np.pi:
                                            dist.append(
                                                np.minimum((pass_inter_ypos - pos) / np.cos(theta[i] - np.pi / 2), rmax))
                                            x = np.arange(pos_x + dist[i] * np.cos(theta[i]), pos_x, 0.01)
                                            y = (pos_x - x) * np.tan(np.pi - theta[i]) + pos
                                        else:
                                            if theta[i] == np.pi:
                                                dist.append(rmax)
                                                x = np.arange(pos_x - rmax, pos_x, 0.01)
                                                y = pos * np.ones([1, len(x)])
                                            else:
                                                if theta[i] > np.pi and theta[i] < np.pi + np.arctan((pos - app_inter_ypos)/(lx)):
                                                    dist.append(np.minimum((pos - app_inter_ypos) / np.sin(theta[i] - np.pi), rmax))
                                                    x = np.arange(pos_x - dist[i] * np.cos(theta[i] - np.pi), pos_x, 0.01)
                                                    y = (pos_x - x) * np.tan(np.pi - theta[i]) + pos
                                                else:
                                                    dist.append(np.minimum(lx / np.cos(np.pi - theta[i]), rmax))
                                                    x = np.arange(pos_x - dist[i] * np.cos(np.pi - theta[i]), pos_x, 0.01)
                                                    y = (pos_x - x) * np.tan(np.pi - theta[i]) + pos
        plt.plot(x, y, 'k')
        plt.draw()
    return dist


def dis_function(pos, n, fpos):
    drawIntersection()
    state_dis = linetrace(pos, n, fpos)
    plt.plot(pos_x, pos, 'r', marker='s',markersize=8)
    plt.plot(pos_x, fpos, 'g', marker='s',markersize=8)
    plt.plot(hv_pos_b, 2 * np.ones([1,no_v]), 'g', marker='s', markersize=8)
    plt.plot(hv_pos_f, -2 * np.ones([1,no_v]), 'g', marker='s', markersize=8)
    plt.draw()
    return state_dis


def get_state(pos, n, fpos):
    # x0 = 2*np.random.rand(1)
    state_dis = dis_function(pos, n, fpos)
    return state_dis #np.array(np.concatenate([x0, state_dis]), ndmin=2)


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


def current_state(pos, fpos):
    plt.ion()
    new_dis = np.array(get_state(pos, n_degree, fpos), ndmin=2)
    new_area = s_reward(new_dis, n_degree)
    plt.pause(0.01)
    plt.clf()
    return new_dis, new_area


def state_table(av_pos, fv_pos, hv_pos_b, hv_pos_f):
    plt.ion()
    state = np.array(get_state(av_pos, n_degree,fv_pos), ndmin=2)
    state_reward = [s_reward(state, n_degree)]
    pos = av_pos
    i = 1
    while av_pos <= pass_inter_ypos:
        fv_pos += tau * vel_fv
        hv_pos_b -= tau * vel_hv
        hv_pos_f += tau * vel_hv
        av_pos += tau * vel_av
        new_state = np.array(get_state(av_pos, n_degree, fv_pos), ndmin=2)
        new_reward = s_reward(new_state, n_degree)
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

pos, state, state_reward = state_table(av_pos, fv_pos, hv_pos_b, hv_pos_f)