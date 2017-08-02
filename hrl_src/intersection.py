#!/usr/bin/env python
import Queue
from threading import Thread
import vires_comm
import numpy as np
from control import SimVires


class UpdateInter(object):
    Lidar_NO = 36
    Resolution = 2 * 180 / Lidar_NO
    # stop_line = [192, 195, 198]
    Stop_Line = 142.0
    # Comfort_Acc = 12
    Comfort_Acc = 1.0
    Pass_Point = 180.0
    # self.Tau = 0.1
    # self.Tau = 0.0666
    Tau = 0.033
    Speed_limit = 30.0  # m/s
    Max_Acc = 3.0  # m/s
    observe_vel = 12.0   # m/s

    def __init__(self):
        SimVires.restart_sim()
        self.av_velocity = np.random.normal(self.Speed_limit, 1.)
        self.av_x = 0.0
        self.av_y = 0.
        self.state_vm = []
        self.state_dim = None

        self.state_q = Queue.LifoQueue()
        self.action_q = Queue.LifoQueue()
        self.state_thread = Thread(target=vires_comm.vires_state, args=(self.state_q,))
        self.state_thread.start()
        self.action_thread = Thread(target=vires_comm.vires_action, args=(self.action_q,))
        self.action_thread.start()
        self.action = {}

    def get_state(self):
        curr_state = self.state_q.get()
        self.av_velocity = curr_state['position']['v']
        self.av_x = curr_state['position']['x']
        self.av_y = curr_state['position']['y']
        state = list()
        state.append(curr_state['position']['v'])
        sensor_data = curr_state['sensor']
        for theta, dist in sensor_data.iteritems():
            state.append(dist)
            self.state_vm.append(dist)
        state.append(curr_state['position']['x'] - 16.)
        state.append(curr_state['position']['y'] - 153.)
        state.append(6)
        state.append(18)
        state.append(max(- self.Stop_Line + curr_state['position']['y'], 0.))
        self.state_dim = len(state)
        return np.array(state, ndmin=2)

    def update_vehicle(self, a):
        old_velocity = self.av_velocity
        self.av_velocity += a * self.Tau
        self.av_velocity = max(self.av_velocity, 0)
        self.av_y += 0.5 * (self.av_velocity + old_velocity) * self.Tau
        # SimRun.update_speed(old_velocity)
        SimVires.update_speed(self.av_velocity)
        print("a = ", a, ", speed = ", self.av_velocity, ", old_speed = ", old_velocity, "position = ", self.av_y)
        new_state = self.get_state()
        return new_state

    def reward_function(self, a):
        # distance = 2.4
        distance = 1
        if self.av_y >= self.Pass_Point:
            r = 500 + 0.1 * self.av_velocity
        else:
            r = 0.1 * self.av_velocity
        r -= 500 * int(sum(i <= distance for i in self.state_vm) > 0)
        r -= self.Tau * 10
        # r -= 0.1 * max((self.av_velocity - self.Speed_limit), 0)
        r -= 0.5 * max(abs(a) - self.Comfort_Acc, 0)
        if - self.av_y + self.Stop_Line <= 5 and -self.av_y + self.Stop_Line >= 0:
            # r -= max(self.av_velocity - 9, 0)
            r -= max(self.av_velocity - 18, 0)
        collision = sum(i <= distance for i in self.state_vm)
        return r, collision
