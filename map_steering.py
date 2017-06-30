#!/usr/bin/env python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from random import randint
import random
import math


#Road 1: 193 - 217, Road 2 347 - 371, Road 3 501 - 525     *3    X: 161 - 185
class IntersectionMap(object):
    def __init__(self, Visual):
        self.map = mpimg.imread('map3.png')
        map_x = np.shape(self.map)[0]
        map_y = np.shape(self.map)[1]
        self.Scenary = randint(0, 2)
        self.Speed_limit = 25 #m/s
        self.av_velocity = np.random.normal(self.Speed_limit, 1)
        self.av_x = random.random() * 50 + 200
        self.av_y = self.Scenary*154 + 211
        ########################################################################################
        # No human vehicles
        # self.fv_velocity = np.random.normal(self.Speed_limit, 5)
        # self.fv_y = np.random.normal(self.av_y, 1)
        # self.fv_x = random.random() * 100 + 100
        # self.fv_a = 3 #m/s
        # self.Vehicle_NO = 10
        # self.hv_velocity = np.random.normal(self.Speed_limit, 5)
        # self.hv_f_x = np.random.normal(179, 1, self.Vehicle_NO)
        # self.hv_f_y = np.random.uniform(0 - 200, self.av_y + 6, [1, self.Vehicle_NO])
        # self.hv_b_x = np.random.normal(167, 1, self.Vehicle_NO)
        # self.hv_b_y = np.random.uniform(self.av_y - 6, map_y + 200,[1, self.Vehicle_NO])
        ########################################################################################
        self.Lidar_NO = 60
        self.Lidar_len_step = 0.5
        self.Lidar_len_Max = 240
        self.state_vm = []
        self.Resolution = 2*np.pi/self.Lidar_NO
        self.Inter_Origin = np.array([[173, 205],[173, 359], [173, 513]])
        self.Stop_Line_Y = np.array([np.arange(205, 217), np.arange(359, 371), np.arange(513, 525)])
        stop_line = [192, 195, 198]
        self.Tau = 0.2
        self.Goal = [[179, self.av_y - 24], [155, self.av_y], [167, self.av_y + 12]]
        self.Approach = 185
        self.Upper = 161
        self.Pass_Intersection = self.Goal[self.Scenary]
        self.state_dim = 0
        self.Max_Acc = 10
        self.Comfort_Acc = 4
        self.observe_vel = 5   #m/s
        self.Visual = Visual
        self.Stop_Line_X = stop_line[self.Scenary] + 0.1 * np.random.randn(1)
        self.head_angle = 0
        self.LEFT_LANE = self.Scenary * 154 + 205
        self.RIGHT_LANE = self.Scenary * 154 + 217
        self.dist = 100

    def show_map(self):
        if self.Visual:
            plt.imshow(self.map)
            plt.plot(self.Inter_Origin[:,1], self.Inter_Origin[:,0],'k.')
            plt.plot(self.Goal[self.Scenary][1], self.Goal[self.Scenary][0], 'g.', markersize=8)
            for i in range(3):
                plt.plot(self.Stop_Line_Y[i,:], self.Stop_Line_X*np.ones([len(self.Stop_Line_Y[i,:])]), 'k.')
            plt.plot(self.av_y, self.av_x, 'gs', markersize=8)
            plt.show()

    def add_vehicle(self):
        self.map = mpimg.imread('map3.png')
        ########################################################################################
        # No human vehicles
        # self.map[math.floor(self.fv_x - 6):math.ceil(self.fv_x + 6),
        # math.floor(self.fv_y - 3) : math.ceil(self.fv_y + 3)] = [0.0, 0.0, 0.0, 1.0]
        # for i in range(self.Vehicle_NO):
        #     self.map[math.floor(self.hv_f_x[i] - 3):math.ceil(self.hv_f_x[i] + 3),
        #     math.floor(self.hv_f_y[0][i] - 6): math.ceil(self.hv_f_y[0][i] + 6)] = [0.0, 0.0, 0.0, 1.0]
        #     self.map[math.floor(self.hv_b_x[i] - 3):math.ceil(self.hv_b_x[i] + 3),
        #     math.floor(self.hv_b_y[0][i] - 6): math.ceil(self.hv_b_y[0][i] + 6)] = [0.0, 0.0, 0.0, 1.0]
        ########################################################################################
        if self.Visual:
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
                    if self.Visual:
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
        d_SL = max(self.Stop_Line_X - self.av_x, 0)
        ##################################################
        #Test1, Test 2
        # all_state = np.array(
        #     np.hstack((self.av_velocity, self.state_vm, self.av_x, 18, 6, d_SL)), ndmin=2)
        ##################################################
        #Test 3 Tau = 0.2, Steering angle
        all_state = np.array(
            np.hstack((self.av_velocity, self.state_vm, self.av_x, self.av_y - self.LEFT_LANE, self.av_y - self.RIGHT_LANE,
                       self.head_angle, d_SL, self.Goal[self.Scenary][0] - self.av_x, self.Goal[self.Scenary][1] - self.av_y)), ndmin=2)
        # all_state = np.array(np.hstack((self.av_velocity, self.state_vm, self.av_x, self.av_y, self.fv_y, self.Speed_limit,
        #                             self.Stop_Line_X, max(self.hv_f_y [self.hv_f_y  <= (self.av_y + 6)]),
        #                             min(self.hv_b_y[self.hv_b_y >= (self.av_y - 6)]))), ndmin=2)
        # all_state = np.array(np.hstack((self.av_velocity, self.state_vm, self.av_x, self.av_y, self.fv_y, self.Speed_limit,
        #                                 self.Stop_Line_X, np.maximum(self.av_y - self.hv_f_y, np.zeros([1,self.Vehicle_NO]))[0,:],
        #                                 np.maximum(self.hv_b_y - self.av_y, np.zeros([1, self.Vehicle_NO]))[0,:])), ndmin=2)
        self.state_dim = len(all_state[0,:])
        return all_state

    def update_vehicle(self, a, s):
        old_velocity = self.av_velocity
        self.av_velocity += a * self.Tau
        self.av_velocity = max(self.av_velocity, 0)
        ####################################################################
        #Without steering angle
        # self.av_x -= 0.5 * (self.av_velocity + old_velocity) * self.Tau
        ####################################################################
        #With steering angle
        self.head_angle += s
        if self.head_angle < -np.pi /2:
            self.head_angle = -np.pi /2
        else:
            if self.head_angle > np.pi / 2:
                self.head_angle = np.pi / 2
        delta_d = 0.5 * (self.av_velocity + old_velocity) * self.Tau
        self.av_x -= delta_d * np.cos(self.head_angle)
        self.av_y += delta_d * np.sin(self.head_angle)
        #####################################################################
        ########################################################################################
        # No human vehicles
        # if self.fv_x > self.Stop_Line_X:
        #     stop_t = 2 * (self.fv_x - self.Approach) / self.fv_velocity
        #     stop_a = self.fv_velocity / stop_t
        #     self.fv_x -= self.Tau * self.fv_velocity + 0.5 * stop_a * (self.Tau **2)
        #     self.fv_velocity -= stop_a * self.Tau
        #     self.fv_velocity = max(self.fv_velocity, 0)
        # else:
        #     if self.fv_x >= 40:
        #         if self.fv_velocity < self.Speed_limit:
        #             self.fv_x -= self.fv_velocity * self.Tau - 0.5 * self.fv_a * (self.Tau ** 2)
        #             self.fv_velocity += self.fv_a * self.Tau
        #             self.fv_velocity = max(self.fv_velocity, 0)
        #         else:
        #             self.fv_x -= self.fv_velocity * self.Tau
        #     else:
        #         self.fv_velocity = 0
        # self.hv_f_y += self.Tau * self.hv_velocity
        # self.hv_b_y -= self.Tau * self.hv_velocity
        ########################################################################################
        self.add_vehicle()
        self.dist = np.sqrt(
            (self.av_x - self.Goal[self.Scenary][0]) ** 2 + (self.av_y - self.Goal[self.Scenary][1]) ** 2)
        new_state = self.get_state()
        return new_state

    def reward_function(self, a):
        ####################################################################
        # Without steering angle
        # if self.av_x <= self.Pass_Intersection:
        #     r = 500 + 0.5 * self.av_velocity
        # else:
        #     r = 0.5 * self.av_velocity
        ####################################################################
        # With steering angle
        collision = 0
        if self.dist <= 1:
            r = 500
        else:
            r = 0

        centre_line = self.Scenary * 154 + 211
        if self.av_x > self.Approach:
            phi = np.arctan((centre_line - self.av_y) / (self.Approach - self.av_x))
        else:
            phi = np.arctan((self.Goal[self.Scenary][1] - self.av_y) / (self.Goal[self.Scenary][0] - self.av_x))
        r += 0.5 * self.av_velocity * np.cos((phi - self.head_angle))

        if self.av_x >= self.Approach or self.av_x >= self.Upper:
            if np.logical_or(self.av_y < self.LEFT_LANE, self.av_y > self.RIGHT_LANE):
                collision += 1
                r -= 500
            else:
                r -= abs(self.av_y - centre_line)

        r -= 500 * int(sum(i <= 5 for i in self.state_vm) > 0)
        collision += sum(i <= 5 for i in self.state_vm)

        r -= self.Tau * 10
        r -= 0.1 * max(abs(a) - self.Comfort_Acc, 0)
        # r -= 0.1 * self.dist

        if self.av_x - self.Stop_Line_X <= 3 and self.av_x - self.Stop_Line_X >= 0:
            r -= max(self.av_velocity - 3, 0)
        return  r,collision


# def pass_intersection(a):
#     plt.ion()
#     intersection = IntersectionMap(True)
#     intersection.add_vehicle()
#     new_state = intersection.get_state()
#     plt.pause(0.1)
#     plt.clf()
#     i = 1
#     print(intersection.dist)
#     while intersection.dist >= 1: #intersection.av_x >= intersection.Pass_Intersection:
#         new_state = intersection.update_vehicle(a, 0.1)
#         new_reward, collision = intersection.reward_function(a)
#         plt.pause(0.1)
#         plt.clf()
#         i += 1
#         # if collision > 0:
#         #     break
#         print(intersection.dist)
#     return new_state
#
# if __name__ == '__main__':
#    pass_intersection(0)
