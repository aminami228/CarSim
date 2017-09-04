import socket
import time
from vires_types import RDB_MSG_HDR_t, RDB_MSG_ENTRY_HDR_t, RDB_OBJECT_STATE_t, RDB_SENSOR_OBJECT_t, RDB_DRIVER_CTRL_t
from ctypes import sizeof
import logging


class RDBComm(object):
    BUFFER = 64*1024      # 64k Max size of TCP packets; Tip: Max ethernet packet size ~1500 Bytes
    SCP_MAGIC_NO = 40108
    RDB_MAGIC_NO = 35712
    MIN_SENSOR = 3

    EGO = 'AV'
    NEIGHBOR = 'HV1'
    SCENE = 'test.xml'

    def __init__(self):
        self.scp_port = 48179
        self.rdb_sensor_port = 48195
        self.rdb_port = 48190
        self.rdb_ego_port = 48195

        self.get_ego_time = time.time()
        self.get_hv_time = time.time()
        self.get_sensor_time = time.time()

    def update_state(self, state_q, vehicle):
        connect_port = self.rdb_ego_port if vehicle == 'ego' else self.rdb_port
        rdb_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        rdb_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        rdb_sock.connect_ex(('127.0.0.1', connect_port))
        rdb_buff = bytearray(self.BUFFER)

        while True:
            av_state = dict()
            hv_state = dict()
            # av_sensor = dict()

            # Read data
            n_bytes = rdb_sock.recv_into(rdb_buff)  # blocking?
            # logging.debug("Received n_bytes={} of data".format(n_bytes))
            if n_bytes == 0:
                logging.error("Ethernet frame dropped? n_bytes==0")
                continue
            rdb_hdr = RDB_MSG_HDR_t.from_buffer(rdb_buff[:sizeof(RDB_MSG_HDR_t)])
            if rdb_hdr.magicNo != self.RDB_MAGIC_NO:
                return
            entry_idx = rdb_hdr.headerSize
            remain_bytes = rdb_hdr.dataSize
            entry = RDB_MSG_ENTRY_HDR_t.from_buffer(rdb_buff[entry_idx:entry_idx+sizeof(RDB_MSG_HDR_t)])
            while remain_bytes > 0:
                # process RDB_MSG_t data
                data_idx = entry_idx + entry.headerSize
                n_elements = 0
                if entry.elementSize > 0:
                    n_elements = entry.dataSize / entry.elementSize
                for n in range(n_elements):
                    if entry.pkgId == 9:       # RDB_OBJECT_STATE_t
                        data = RDB_OBJECT_STATE_t.from_buffer(rdb_buff[data_idx:data_idx + sizeof(RDB_OBJECT_STATE_t)])
                        full_state = dict()
                        if vehicle == 'ego' and (data.base.name == self.EGO or data.base.id == 1):
                            av_state['x'] = data.base.pos.x
                            av_state['y'] = data.base.pos.y
                            av_state['h'] = data.base.pos.h
                            av_state['v'] = data.ext.speed.y
                            av_state['a'] = data.ext.accel.y
                            full_state['av'] = av_state
                            state_q.put(full_state)
                            # logging.debug('name: ' + str(data.base.name) + ', id: ' + str(data.base.id))
                            # logging.debug('Added av state #: ' + str(state_q.qsize())+';' + str(state_q))
                        if vehicle == 'hv' and (data.base.name == self.NEIGHBOR or data.base.id in [3, 14]):
                            hv_state['x'] = data.base.pos.x
                            hv_state['y'] = data.base.pos.y
                            hv_state['h'] = data.base.pos.h
                            hv_state['v'] = data.ext.speed.y
                            hv_state['a'] = data.ext.accel.y
                            state_q.put(hv_state)
                            # logging.debug('Added hv state #: ' + str(state_q.qsize()) + ', ' + str(state_q.queue[-1]))
                    data_idx += entry.elementSize
                # Advance in the buffer
                remain_bytes -= (entry.headerSize + entry.dataSize)
                if remain_bytes > 0:
                    entry_idx += (entry.headerSize + entry.dataSize)
                    entry = RDB_MSG_ENTRY_HDR_t.from_buffer(rdb_buff[entry_idx:entry_idx + sizeof(RDB_MSG_HDR_t)])

    def rdb_control(self, action_q):
        connect_port = self.rdb_port
        rdb_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        rdb_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        rdb_sock.connect_ex(('127.0.0.1', connect_port))
        rdb_buff = bytearray(self.BUFFER)

        while True:
            n_bytes = rdb_sock.recv_into(rdb_buff)  # blocking?
            rdb_hdr = RDB_MSG_HDR_t.from_buffer(rdb_buff[:sizeof(RDB_MSG_HDR_t)])
            if rdb_hdr.magicNo != self.RDB_MAGIC_NO:
                logging.error('Wrong RDB port for vehicle !!! RDB magic NO is ' + str(rdb_hdr.magicNo))
                continue

            entry_idx = rdb_hdr.headerSize
            remain_bytes = rdb_hdr.dataSize
            while remain_bytes > 0:
                entry = RDB_MSG_ENTRY_HDR_t.from_buffer(rdb_buff[entry_idx:entry_idx + sizeof(RDB_MSG_HDR_t)])
                if entry.pkgId != 26:
                    pass
                else:
                    data_idx = entry_idx + entry.headerSize
                    n_elements = 0
                    if entry.elementSize > 0:
                        n_elements = entry.dataSize / entry.elementSize
                    for n in range(n_elements):
                        data = RDB_DRIVER_CTRL_t.from_buffer(rdb_buff[data_idx:data_idx + sizeof(RDB_DRIVER_CTRL_t)])
                        if not action_q.empty():
                            full_action = action_q.get()
                            data.steeringTgt = full_action['steer']
                            logging.debug('Control steering: ' + str(full_action['steer']) + ', Update Time: ' +
                                          str(time.time() - self.update_steer))
                            self.update_steer = time.time()
                        else:
                            data.steeringTgt = 0.
                            # logging.error('No steering angle get !!!')
                        rdb_sock.send(bytearray(rdb_hdr) + bytearray(entry) + bytearray(data))
                        time.sleep(0.001)
                        data_idx += entry.elementSize
                remain_bytes -= (entry.headerSize + entry.dataSize)
                entry_idx = entry_idx + (entry.headerSize + entry.dataSize)
