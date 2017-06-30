#!/usr/bin/env python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
from random import randint
import random
import math


#Road 1: 193 - 217, Road 2 347 - 371, Road 3 501 - 525     *3    X: 161 - 185
class Empty_IntersectionMap(object):
    def __init__(self):
        self.map = mpimg.imread('map2.png')
        map_x = np.shape(self.map)[0]
        map_y = np.shape(self.map)[1]
        self.Scenary = randint(0, 2)
        self.Speed_limit = 25 #m/s
        self.av_velocity = np.random.normal(self.Speed_limit, 5)
        self.av_x = 300
        self.av_y = self.Scenary*154 + 211
        self.fv_velocity = np.random.normal(self.Speed_limit, 5)
        self.fv_y = np.random.normal(self.av_y, 1)
        self.fv_x = random.random() * 100 + 150
        self.fv_a = 3  # m/s
        self.Lidar_NO = 60
        self.Lidar_len_step = 0.5
        self.Lidar_len_Max = 240
        self.state_vm = []
        self.Resolution = 2*np.pi/self.Lidar_NO
        self.Inter_Origin = np.array([[173, 205],[173, 359], [173, 513]])
        self.Stop_Line_Y = np.array([np.arange(205, 217), np.arange(359, 371), np.arange(513, 525)])
        self.Stop_Line_X = [192, 195, 198]
        self.Tau = 0.1
        self.Goal = [155, self.av_y]
        self.Approach = 185
        self.Pass_Intersection = self.Goal[0]
        self.state_dim = 0
        self.Max_Acc = 10
        self.Comfort_Acc = 4
        self.observe_vel = 5   #m/s

    def show_map(self):
        plt.imshow(self.map)
        plt.plot(self.Inter_Origin[:,1], self.Inter_Origin[:,0],'k.')
        plt.plot(self.Goal[1], self.Goal[0], 'g.', markersize=8)
        for i in range(3):
            plt.plot(self.Stop_Line_Y[i,:], self.Stop_Line_X[i]*np.ones([len(self.Stop_Line_Y[i,:])]), 'k.')
        plt.plot(self.av_y, self.av_x, 'gs', markersize=8)
        # plt.plot(self.fv_y, self.Fv_x, 'gs', markersize=8)
        # plt.plot(self.Hv_f_y, self.hv_f_x * np.ones([1, self.Vehicle_NO]), 'gs', markersize=8)
        # plt.plot(self.Hv_b_y, self.hv_b_x * np.ones([1, self.Vehicle_NO]), 'gs', markersize=8)
        plt.show()
        # plt.pause(t)

    def add_vehicle(self):
        self.map = mpimg.imread('map2.png')
        self.map[math.floor(self.fv_x - 6):math.ceil(self.fv_x + 6),
        math.floor(self.fv_y - 3): math.ceil(self.fv_y + 3)] = [0.0, 0.0, 0.0, 1.0]
        self.show_map()

    def visibilty_map(self):
        state_vm = []
        for theta in np.arange(5/6*np.pi, 13/6*np.pi, self.Resolution):
            l = 0
            while True:
                if self.map[math.ceil(self.av_x + (l +self.Lidar_len_step)*np.sin(theta)),
                            math.ceil(self.av_y + (l + self.Lidar_len_step) * np.cos(theta))][0] <= 0.5 \
                        or l >= self.Lidar_len_Max:
                    state_vm.append(l)
                    plt.plot([self.av_y, self.av_y + l*np.cos(theta)], [self.av_x, self.av_x + l*np.sin(theta)], 'r')
                    plt.plot(self.av_y + l*np.cos(theta), self.av_x + l*np.sin(theta), 'r*', markersize=8)
                    plt.plot(self.av_y, self.av_x, 'gs', markersize=5)
                    plt.draw()
                    break
                else:
                    l += self.Lidar_len_step
        return state_vm

    def get_state(self):
        self.state_vm = self.visibilty_map()
        all_state = np.array(np.hstack((self.av_velocity, self.state_vm, self.av_x, self.av_y, self.Speed_limit, self.fv_y,
                                    self.Stop_Line_X)), ndmin=2)
        self.state_dim = len(all_state[0,:])
        return all_state

    def update_vehicle(self, a):
        self.av_velocity += a * self.Tau
        self.av_velocity = max(self.av_velocity, 0)
        self.av_x -= self.Tau * self.av_velocity - 0.5 * a * (self.Tau**2)
        if self.fv_x > self.Stop_Line_X[self.Scenary]:
            stop_t = 2 * (self.fv_x - self.Approach) / self.fv_velocity
            stop_a = self.fv_velocity / stop_t
            self.fv_x -= self.Tau * self.fv_velocity + 0.5 * stop_a * (self.Tau **2)
            self.fv_velocity -= stop_a * self.Tau
            self.fv_velocity = max(self.fv_velocity, 0)
        else:
            if self.fv_x >= 0:
                if self.fv_velocity < self.Speed_limit:
                    self.fv_x -= self.fv_velocity * self.Tau - 0.5 * self.fv_a * (self.Tau ** 2)
                    self.fv_velocity += self.fv_a * self.Tau
                    self.fv_velocity = max(self.fv_velocity, 0)
                else:
                    self.fv_x -= self.fv_velocity * self.Tau
            else:
                self.fv_velocity = 0
        self.add_vehicle()
        new_state = self.get_state()
        return new_state

    def reward_function(self, a):
        if self.av_x <= self.Pass_Intersection:
            r = 100
        else:
            r = 0
        r -= 500 * int(sum(i <= 5 for i in self.state_vm) > 0)
        r -= self.Tau * 10
        r -= int(self.av_velocity > self.Speed_limit)
        r -= int(a > self.Comfort_Acc)
        stop_line = self.Stop_Line_X[self.Scenary] + 0.1 * np.random.randn(1)
        if self.av_x - stop_line <= 1 and self.av_x - stop_line >= 0:
            r -= self.av_velocity
        collision = sum(i <= 5 for i in self.state_vm)
        return r, collision


# def pass_intersection(a):
#     plt.ion()
#     intersection = Empty_IntersectionMap()
#     intersection.add_vehicle()
#     new_state = intersection.get_state()
#     plt.pause(0.1)
#     plt.clf()
#     i = 1
#     while intersection.av_x >= intersection.Pass_Intersection:
#         new_state = intersection.update_vehicle(a)
#         new_reward, collision = intersection.reward_function(a)
#         plt.pause(0.01)
#         plt.clf()
#         i += 1
#         if collision > 0:
#             break
#     return new_state, new_reward
#
# if __name__ == '__main__':
#    pass_intersection(0)
