import socket
import time
from vires_types import RDB_MSG_HDR_t, RDB_MSG_ENTRY_HDR_t, RDB_OBJECT_STATE_t, RDB_SENSOR_OBJECT_t
from ctypes import sizeof
import logging


class RDBComm(object):
    BUFFER = 204800
    SCP_MAGIC_NO = 40108
    RDB_MAGIC_NO = 35712
    MIN_SENSOR = 3

    EGO = 'AV'
    NEIGHBOR = 'HV1'
    SCENE = 'test.xml'

    def __init__(self):
        self.scp_port = 48179
        self.rdb_ego_port = 48195
        self.rdb_hv_port = 48190

        self.get_ego_time = time.time()
        self.get_hv_time = time.time()
        self.get_sensor_time = time.time()

    def update_state(self, state_q, vehicle):
        connect_port = self.rdb_ego_port if vehicle == 'ego' else self.rdb_hv_port
        rdb_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        rdb_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        rdb_sock.connect_ex(('127.0.0.1', connect_port))
        rdb_buff = bytearray(self.BUFFER)

        while True:
            n_bytes = rdb_sock.recv_into(rdb_buff)  # blocking?
            av_state = dict()
            hv_state = dict()
            av_sensor = dict()
            rdb_hdr = RDB_MSG_HDR_t.from_buffer(rdb_buff[:sizeof(RDB_MSG_HDR_t)])
            if rdb_hdr.magicNo != self.RDB_MAGIC_NO:
                logging.error('Wrong RDB port for vehicle !!! RDB magic NO is ' + str(rdb_hdr.magicNo))
                continue

            entry_idx = rdb_hdr.headerSize
            remain_bytes = rdb_hdr.dataSize
            while remain_bytes > 0:
                entry = RDB_MSG_ENTRY_HDR_t.from_buffer(rdb_buff[entry_idx:entry_idx + sizeof(RDB_MSG_HDR_t)])
                if (vehicle == 'ego' and (entry.pkgId != 9 and (entry.pkgId != 17))) or\
                        (vehicle != 'ego' and (entry.pkgId != 9)):
                    pass
                else:
                    data_idx = entry_idx + entry.headerSize
                    n_elements = 0
                    if entry.elementSize > 0:
                        n_elements = entry.dataSize / entry.elementSize
                    for n in range(n_elements):
                        data = RDB_OBJECT_STATE_t.from_buffer(rdb_buff[data_idx:data_idx + sizeof(RDB_OBJECT_STATE_t)])
                        if vehicle == 'ego' and (entry.pkgId == 9) and (data.base.name == self.EGO):
                            av_state['x'] = data.base.pos.x
                            av_state['y'] = data.base.pos.y
                            av_state['h'] = data.base.pos.h
                            av_state['v'] = data.ext.speed.y
                            av_state['a'] = data.ext.accel.y
                        elif vehicle != 'ego' and (entry.pkgId == 9) and (data.base.name == self.NEIGHBOR):
                            hv_state['x'] = data.base.pos.x
                            hv_state['y'] = data.base.pos.y
                            hv_state['h'] = data.base.pos.h
                            hv_state['v'] = data.ext.speed.y
                            hv_state['a'] = data.ext.accel.y
                        elif vehicle == 'ego' and (entry.pkgId == 17):
                            data = RDB_SENSOR_OBJECT_t.from_buffer(rdb_buff[data_idx:data_idx + sizeof(RDB_SENSOR_OBJECT_t)])
                            av_sensor[data.sensorPos.h] = data.dist
                        data_idx += entry.elementSize
                remain_bytes -= (entry.headerSize + entry.dataSize)
                entry_idx = entry_idx + (entry.headerSize + entry.dataSize)

            full_state = dict()
            if vehicle == 'ego' and (len(av_state) == 5) and (len(av_sensor) >= 10):
                full_state['av'] = av_state
                full_state['sensor'] = av_sensor
                state_q.put(full_state)
                # logging.debug('after put av: ' + str(state_q.qsize()) + ', ' + str(state_q.queue[-1]))
            if vehicle != 'ego' and (len(hv_state) == 5):
                state_q.put(hv_state)
                # logging.debug('after put hv: ' + str(state_q.qsize()) + ', ' + str(state_q.queue[-1]))
            time.sleep(0.001)
