import socket
import time
from vires_types import RDB_MSG_HDR_t, RDB_MSG_ENTRY_HDR_t, RDB_OBJECT_STATE_t, RDB_SENSOR_OBJECT_t
from ctypes import sizeof
import logging
import utilities.log_color


class RDBComm(object):
    BUFFER = 204800
    SCP_MAGIC_NO = 40108
    RDB_MAGIC_NO = 35712

    EGO = 'AV'
    NEIGHBOR = 'New Player 02'
    SCENE = 'test.xml'

    def __init__(self):
        self.rdb_buff = None
        self.entry_idx = None
        self.remain_bytes = 0
        self.entry = None

        self.rdb_ego_port = 48195
        self.rdb_neighbor_port = 48190

    def update_buffer(self):
        size = self.entry.headerSize + self.entry.dataSize
        self.remain_bytes -= size
        if self.remain_bytes > 0:
            self.entry_idx += size

    @staticmethod
    def get_pos(data):
        vehicle_state = dict()
        vehicle_state['x'] = data.base.pos.x
        vehicle_state['y'] = data.base.pos.y
        vehicle_state['h'] = data.base.pos.h
        vehicle_state['v'] = data.ext.speed.y
        vehicle_state['a'] = data.ext.accel.y
        return vehicle_state

    def process_ego(self, vehicle):
        vehicle_state = None
        sensor_state = None
        rdb_hdr = RDB_MSG_HDR_t.from_buffer(self.rdb_buff[:sizeof(RDB_MSG_HDR_t)])
        if rdb_hdr.magicNo != self.RDB_MAGIC_NO:
            logging.error('Wrong RDB info for ' + vehicle + ' vehicle !!!')
            return vehicle_state, sensor_state
        self.entry_idx = rdb_hdr.headerSize
        self.remain_bytes = rdb_hdr.dataSize
        while self.remain_bytes > 0:
            self.entry = RDB_MSG_ENTRY_HDR_t.from_buffer(
                self.rdb_buff[self.entry_idx:self.entry_idx + sizeof(RDB_MSG_HDR_t)])
            data_idx = self.entry_idx + self.entry.headerSize
            n_elements = self.entry.dataSize / self.entry.elementSize if self.entry.elementSize > 0. else 0.
            for n in range(n_elements):
                if self.entry.pkgId == 9:                            # RDB_PKG_ID_OBJECT_STATE
                    data = RDB_OBJECT_STATE_t.from_buffer(self.rdb_buff[data_idx:data_idx + sizeof(RDB_OBJECT_STATE_t)])
                    if data.base.name == self.EGO:
                        vehicle_state = self.get_pos(data)
                if vehicle != 'ego' and self.entry.pkgId == 17:      # RDB_PKG_ID_SENSOR_OBJECT
                    data = RDB_SENSOR_OBJECT_t.from_buffer(self.rdb_buff[data_idx:data_idx + sizeof(RDB_SENSOR_OBJECT_t)])
                    sensor_state[data.sensorPos.h] = data.dist
                data_idx += self.entry.elementSize
            self.update_buffer()
        return vehicle_state, sensor_state

    def update_state(self, state_q, vehicle='front'):
        """
        Receive ego vehicle info & ego vehicle sensors.
        TCP_NODELAY is intended to disable/enable segment buffering,
        so data can be sent out to peer as quickly as possible.
        This is typically used to improve network utilisation.
        """
        if vehicle == 'ego':
            logging.info('Update Vires Ego Vehicle State')
        else:
            logging.info('Update Vires Front Vehicle State')
        connect_port = self.rdb_ego_port if vehicle == 'ego' else self.rdb_neighbor_port
        rdb_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        rdb_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        rdb_sock.connect_ex(('127.0.0.1', connect_port))
        self.rdb_buff = bytearray(self.BUFFER)

        while True:
            n_bytes = rdb_sock.recv_into(self.rdb_buff)         # blocking?
            vehicle_state, sensor_state = self.process_ego(vehicle)
            full_state = dict()
            if vehicle_state is not None:
                full_state['position'] = vehicle_state
                full_state['sensor'] = sensor_state
                full_state['collision'] = False
                state_q.put(full_state)
            time.sleep(0)
