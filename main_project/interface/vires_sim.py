import numpy as np
from random import randint
from utilities.toolfunc import ToolFunc
import Queue
from threading import Thread
from interface.scp_comm import SCPComm
from interface.rdb_comm import RDBComm
import time
import logging
import utilities.log_color

__author__ = 'qzq'


class InterSim(object):
    Tau = 1. / 30
    Speed_limit = 12        # m/s
    Scenary = randint(0, 2)
    Inter_Ori = 0.
    Stop_Line = 142.733
    Pass_Point = 168
    Inter_Low = 147.355
    Inter_Up = 157.931
    Inter_Left = 11.545
    Inter_Right = 19.571
    Vehicle_NO = 1
    Lane_Left = 14.935
    Lane_Right = 17.683
    Cft_Accel = 3.     # m/s**2
    Visual = False
    collision = False

    tools = ToolFunc()
    scp = SCPComm()
    rdb = RDBComm()

    def __init__(self, speed):
        self.scp.restart_scp(speed)

        self.av_q = Queue.LifoQueue()
        self.hv_q = Queue.LifoQueue()
        self.av_thread = Thread(target=self.rdb.update_state, args=(self.av_q, 'ego'))
        self.av_thread.start()
        self.hv_thread = Thread(target=self.rdb.update_state, args=(self.hv_q,))
        self.hv_thread .start()
        self.action = {}

        av_state = self.av_q.get()
        av_pos = av_state['position']
        self.av_pos = dict()
        self.av_pos['y'] = av_pos['y']
        self.Start_Pos = av_pos['y']
        self.av_pos['x'] = av_pos['x']
        self.av_pos['vx'] = 0.
        self.av_pos['vy'] = av_pos['v']
        self.av_pos['heading'] = av_pos['h']
        self.av_pos['accel'] = av_pos['a']
        self.av_pos['steer'] = 0.
        self.av_size = [4.445, 1.803]
        self.hv_poses = []
        hv_state = self.hv_q.get()
        for i in range(self.Vehicle_NO):
            hv_pos = dict()
            curr_hv = hv_state['position']
            hv_pos['y'] = curr_hv['y']
            hv_pos['x'] = curr_hv['x']
            hv_pos['vx'] = 0.
            hv_pos['vy'] = curr_hv['v']
            self.hv_poses.append(hv_pos)

        self.state = None
        self.state_dim = None
        self.state_av = []
        self.state_fv = []
        self.state_road = []

    def get_state(self):
        av_state = self.av_q.get()
        av_pos = av_state['position']
        self.av_pos['y'] = av_pos['y']
        self.av_pos['x'] = av_pos['x']
        self.av_pos['vx'] = 0.
        self.av_pos['vy'] = av_pos['v']
        self.av_pos['heading'] = av_pos['h']
        self.av_pos['accel'] = av_pos['a']
        self.av_pos['steer'] = 0.
        self.state_av = [self.av_pos['vy'], self.av_pos['heading'], self.av_pos['accel'], self.av_pos['steer']]

        self.hv_poses = []
        hv_state = self.hv_q.get()
        fv_pos = hv_state['position']
        self.state_fv = [fv_pos['v'], fv_pos['y'] - self.av_pos['y']]

        sl_dis = self.Stop_Line - self.av_pos['y']
        ll = self.av_pos['x'] - self.av_size[1] / 2 - self.Lane_Left
        lr = self.Lane_Right - (self.av_pos['x'] + self.av_size[1] / 2)
        start_pos = self.Start_Pos
        self.state_road = [sl_dis, ll, lr, start_pos]

        self.state = np.array(self.state_av + self.state_fv + self.state_road, ndmin=2)
        self.state_dim = self.state.shape[1]
        return self.state

    def update_vehicle(self, start_time, a=0., st=0.):
        new_av_vel = self.av_pos['vy'] + a * (time.time() - start_time)
        self.scp.scp_control(new_av_vel)
        av_state = self.av_q.get()
        av_pos = av_state['position']
        logging.error('Sim Vel: ' + str(new_av_vel) + ', Real_vel: ' + str(av_pos['vy']))


if __name__ == '__main__':
    sim = InterSim(0.)
    # plt.ion()
    while sim.av_pos['y'] <= sim.Pass_Point:
        sim.get_state()
        # sim.update_vehicle()
