import numpy as np
from random import randint, random
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
    Start_Pos = - 42.

    tools = ToolFunc()

    def __init__(self, speed):
        self.scp = SCPComm()
        self.rdb = RDBComm()
        self.scp.restart_scp(speed)

        # self.av_q = Queue.LifoQueue()
        # self.hv_q = Queue.LifoQueue()
        # self.av_thread = Thread(target=self.rdb.update_state, args=(self.av_q, 'ego'))
        # self.hv_thread = Thread(target=self.rdb.update_state, args=(self.hv_q,))
        # self.av_thread.setDaemon(True)
        # self.hv_thread.setDaemon(True)
        # self.av_thread.start()
        # self.hv_thread.start()
        self.state_q = Queue.LifoQueue()
        self.state_thread = Thread(target=self.rdb.update_state, args=(self.state_q,))
        self.state_thread.setDaemon(True)
        self.state_thread.start()

        self.av_pos = dict()
        self.av_size = [4.445, 1.803]

        self.state = None
        self.state_dim = None
        self.state_av = []
        self.state_fv = []
        self.state_road = []

    def get_state(self, a):
        if not self.state_q.empty():
            vehicle_state = self.state_q.get()
            logging.debug('Get data')
            av_pos = vehicle_state['av']
            self.av_pos['y'] = av_pos['y']
            self.av_pos['x'] = av_pos['x']
            self.av_pos['vx'] = 0.
            self.av_pos['vy'] = av_pos['v']
            self.av_pos['heading'] = av_pos['h']
            self.av_pos['accel'] = a
            self.av_pos['steer'] = 0.
            self.state_av = [self.av_pos['vy'], self.av_pos['heading'], self.av_pos['accel'], self.av_pos['steer']]

            fv_pos = vehicle_state['hv']
            logging.debug('Successfully update new hv state from Vires !!!')
            self.state_fv = [fv_pos['v'], fv_pos['y'] - self.av_pos['y']]
            logging.debug('HV: ' + str(self.state_fv))

            sl_dis = self.Stop_Line - self.av_pos['y']
            ll = self.av_pos['x'] - self.av_size[1] / 2 - self.Lane_Left
            lr = self.Lane_Right - (self.av_pos['x'] + self.av_size[1] / 2)
            start_pos = self.Start_Pos
            self.state_road = [sl_dis, ll, lr, start_pos]
            #
            self.state = np.array(self.state_av + self.state_fv + self.state_road, ndmin=2)
            self.state_dim = self.state.shape[1]
            if self.state_dim != 10:
                logging.error('Wrong states get !!!')
            return self.state
        else:
            return None

    def update_vehicle(self, state, a=0., st=0.):
        # new_av_vel = self.av_pos['vy'] + a * self.Tau
        # new_av_vel = state[0] + 0.1
        new_av_vel = 10.
        self.scp.scp_control(new_av_vel)
        # self.scp.scp_control(0.)
        # av_state = self.av_q.get()
        # av_pos = av_state['position']
        # logging.debug('Sim Vel: ' + str(new_av_vel) + ', Real_vel: ' + str(av_pos['v']))


if __name__ == '__main__':
    sim = InterSim(0.)
    # plt.ion()
    while sim.av_pos['y'] <= sim.Pass_Point:
        sim.get_state()
        # sim.update_vehicle()
