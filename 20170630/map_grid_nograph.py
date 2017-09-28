#!/usr/bin/env python
import matplotlib.image as mpimg
import numpy as np
from random import randint
import random
import math


#Road 1: 193 - 217, Road 2 347 - 371, Road 3 501 - 525     *3    X: 161 - 185
Inter_Origin = np.array([[173, 205],[173, 359], [173, 513]])
Lidar_NO = 36
Resolution = 2*np.pi/Lidar_NO
Approach = 185
stop_line = [192, 195, 198]
Vehicle_NO = 10
Lidar_len_Max = 240
Lidar_len_step = 0.5
Stop_Line_Y = np.array([np.arange(205, 217), np.arange(359, 371), np.arange(513, 525)])
Comfort_Acc = 4

class IntersectionMap(object):
    def __init__(self, Visual):
        self.map = mpimg.imread('map3.png')
        map_y = np.shape(self.map)[1]
        self.Tau = 0.2
        self.Scenary = randint(0, 2)
        self.Speed_limit = 25 #m/s
        self.av_velocity = np.random.normal(self.Speed_limit, 1)
        self.av_x = random.random() * 50 + 200
        self.av_y = self.Scenary*154 + 211
        self.fv_velocity = np.random.normal(self.Speed_limit, 5)
        self.fv_y = np.random.normal(self.av_y, 1)
        self.fv_x = random.random() * 100 + 100
        self.fv_a = 3 #m/s
        self.hv_velocity = np.random.normal(self.Speed_limit, 5)
        self.hv_f_x = np.random.normal(179, 1, Vehicle_NO)
        self.hv_f_y = np.random.uniform(0 - 200, self.av_y + 6, [1, Vehicle_NO])
        self.hv_b_x = np.random.normal(167, 1, Vehicle_NO)
        self.hv_b_y = np.random.uniform(self.av_y - 6, map_y + 200,[1, Vehicle_NO])
        self.state_vm = []
        self.center = Inter_Origin[self.Scenary]
        self.Goal = [[155, self.av_y], [179, self.av_y - 24], [167, self.av_y + 12]]
        self.Pass_Intersection = self.Goal[0][0]
        self.state_dim = 0
        self.Max_Acc = 10   #m/s
        self.observe_vel = 5
        self.Visual = Visual
        self.Stop_Line_X = np.random.normal(stop_line[self.Scenary], 2)
        self.head_angle = 0

    def add_vehicle(self, choose_action):
        self.map = mpimg.imread('map3.png')
        self.map[math.floor(self.fv_x - 6):math.ceil(self.fv_x + 6),
        math.floor(self.fv_y - 3) : math.ceil(self.fv_y + 3)] = [0.0, 0.0, 0.0, 1.0]
        for i in range(Vehicle_NO):
            self.map[math.floor(self.hv_f_x[i] - 3):math.ceil(self.hv_f_x[i] + 3),
            math.floor(self.hv_f_y[0][i] - 6): math.ceil(self.hv_f_y[0][i] + 6)] = [0.0, 0.0, 0.0, 1.0]
            self.map[math.floor(self.hv_b_x[i] - 3):math.ceil(self.hv_b_x[i] + 3),
            math.floor(self.hv_b_y[0][i] - 6): math.ceil(self.hv_b_y[0][i] + 6)] = [0.0, 0.0, 0.0, 1.0]

    def visibilty_map(self, choose_action):
        state_vm = []
        # MU
        for theta in np.arange(4 / 3 * np.pi, 5 / 3 * np.pi, Resolution):
            l = 0
            while True:
                if self.map[math.ceil(self.av_x + (l + Lidar_len_step) * np.sin(theta)),
                            math.ceil(self.av_y + (l + Lidar_len_step) * np.cos(theta))][0] <= 0.5 \
                        or l >= Lidar_len_Max:
                    state_vm.append(l)
                    break
                else:
                    l += Lidar_len_step
        # RU
        for theta in np.arange(4/3*np.pi, 13/6*np.pi, Resolution):
            l = 0
            while True:
                if self.map[math.ceil(self.av_x + (l + Lidar_len_step)*np.sin(theta)),
                            math.ceil(self.av_y + 3 + (l + Lidar_len_step) * np.cos(theta))][0] <= 0.5 \
                        or l >= Lidar_len_Max:
                    state_vm.append(l)
                    break
                else:
                    l += Lidar_len_step
        # RD
        for theta in np.arange(11/6*np.pi, 13/6*np.pi, Resolution):
            l = 0
            while True:
                if self.map[math.ceil(self.av_x + 12 + (l + Lidar_len_step)*np.sin(theta)),
                            math.ceil(self.av_y + 3 + (l + Lidar_len_step) * np.cos(theta))][0] <= 0.5 \
                        or l >= Lidar_len_Max:
                    state_vm.append(l)
                    break
                else:
                    l += Lidar_len_step
        # LD
        for theta in np.arange(5/6*np.pi, 7/6*np.pi, Resolution):
            l = 0
            while True:
                if self.map[math.ceil(self.av_x + 12 + (l + Lidar_len_step)*np.sin(theta)),
                            math.ceil(self.av_y - 3 + (l + Lidar_len_step) * np.cos(theta))][0] <= 0.5 \
                        or l >= Lidar_len_Max:
                    state_vm.append(l)
                    break
                else:
                    l += Lidar_len_step
        # LM
        for theta in np.arange(5 / 6 * np.pi, 7 / 6 * np.pi, Resolution):
            l = 0
            while True:
                if self.map[math.ceil(self.av_x + 6 + (l + Lidar_len_step) * np.sin(theta)),
                            math.ceil(self.av_y - 3 + (l + Lidar_len_step) * np.cos(theta))][0] <= 0.5 \
                        or l >= Lidar_len_Max:
                    state_vm.append(l)
                    break
                else:
                    l += Lidar_len_step
        # RM
        for theta in np.arange(11/6*np.pi, 13/6*np.pi, Resolution):
            l = 0
            while True:
                if self.map[math.ceil(self.av_x + 6 + (l + Lidar_len_step) * np.sin(theta)),
                            math.ceil(self.av_y + 3 + (l + Lidar_len_step) * np.cos(theta))][0] <= 0.5 \
                        or l >= Lidar_len_Max:
                    state_vm.append(l)
                    break
                else:
                    l += Lidar_len_step
        # LU
        for theta in np.arange(5/6*np.pi, 5/3*np.pi, Resolution):
            l = 0
            while True:
                if self.map[math.ceil(self.av_x + (l + Lidar_len_step)*np.sin(theta)),
                            math.ceil(self.av_y - 3 + (l + Lidar_len_step) * np.cos(theta))][0] <= 0.5 \
                        or l >= Lidar_len_Max:
                    state_vm.append(l)
                    break
                else:
                    l += Lidar_len_step
        return state_vm

    def get_state(self, choose_action):
        self.state_vm = self.visibilty_map(choose_action)
        d_SL = max(self.Stop_Line_X - self.av_x, 0)
        ##################################################
        #Test1, Test 2
        # all_state = np.array(
        #     np.hstack((self.av_velocity, self.state_vm, self.av_x, 18, 6, d_SL)), ndmin=2)
        ##################################################
        # Test version 2
        all_state = np.array(
            np.hstack((self.state_vm, self.av_velocity, self.av_x - self.center[0], self.av_y - self.center[1], 6, 18, d_SL)), ndmin=2)
        ##################################################
        self.state_dim = len(all_state[0, :])
        return all_state

    def update_vehicle(self, a, choose_action):
        old_velocity = self.av_velocity
        self.av_velocity += a * self.Tau
        self.av_velocity = max(self.av_velocity, 0)
        ####################################################################
        #Without steering angle
        self.av_x -= 0.5 * (self.av_velocity + old_velocity) * self.Tau
        ####################################################################
        if self.fv_x > self.Stop_Line_X:
            stop_t = 2 * (self.fv_x - Approach) / self.fv_velocity
            stop_a = self.fv_velocity / stop_t
            self.fv_x -= self.Tau * self.fv_velocity + 0.5 * stop_a * (self.Tau **2)
            self.fv_velocity -= stop_a * self.Tau
            self.fv_velocity = max(self.fv_velocity, 0)
        elif self.fv_x >= 40:
            if self.fv_velocity < self.Speed_limit:
                self.fv_x -= self.fv_velocity * self.Tau - 0.5 * self.fv_a * (self.Tau ** 2)
                self.fv_velocity += self.fv_a * self.Tau
                self.fv_velocity = max(self.fv_velocity, 0)
            else:
                self.fv_x -= self.fv_velocity * self.Tau
        else:
            self.fv_velocity = 0
        self.hv_f_y += self.Tau * self.hv_velocity
        self.hv_b_y -= self.Tau * self.hv_velocity
        self.add_vehicle(choose_action)
        new_state = self.get_state(choose_action)
        return new_state

    def reward_function(self, a):
        if self.av_x <= self.Pass_Intersection:
            r = 500 + 0.5 * self.av_velocity
        else:
            r = 0.5 * self.av_velocity
        r -= 500 * int(sum(i <= 2.4 for i in self.state_vm) > 0)
        r -= self.Tau * 10
        # r -= 0.1 * max((self.av_velocity - self.Speed_limit), 0)
        r -= 0.1 * max(abs(a) - Comfort_Acc, 0)
        if self.av_x - self.Stop_Line_X <= 3 and self.av_x - self.Stop_Line_X >= 0:
            r -= max(self.av_velocity - 3, 0)
        collision = sum(i <= 2.4 for i in self.state_vm)
        return r,collision