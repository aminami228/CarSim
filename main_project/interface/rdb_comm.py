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
        self.rdb_neighbor_port = 48190

    def update_state(self, state_q):
        """
        Receive ego vehicle info & ego vehicle sensors.
        TCP_NODELAY is intended to disable/enable segment buffering,
        so data can be sent out to peer as quickly as possible.
        This is typically used to improve network utilisation.
        """
        # connect_port = self.rdb_ego_port if (vehicle == 'ego') else self.rdb_neighbor_port
        connect_port = self.rdb_neighbor_port
        # connect_port = self.rdb_ego_port
        rdb_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        rdb_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        rdb_sock.connect_ex(('127.0.0.1', connect_port))
        rdb_buff = bytearray(self.BUFFER)

        while True:
            n_bytes = rdb_sock.recv_into(rdb_buff)  # blocking?
            av_state = dict()
            hv_state = dict()
            rdb_hdr = RDB_MSG_HDR_t.from_buffer(rdb_buff[:sizeof(RDB_MSG_HDR_t)])
            if rdb_hdr.magicNo != self.RDB_MAGIC_NO:
                logging.error('Wrong RDB port for vehicle !!! RDB magic NO is ' + str(rdb_hdr.magicNo))
                continue

            entry_idx = rdb_hdr.headerSize
            remain_bytes = rdb_hdr.dataSize
            while remain_bytes > 0:
                entry = RDB_MSG_ENTRY_HDR_t.from_buffer(rdb_buff[entry_idx:entry_idx + sizeof(RDB_MSG_HDR_t)])
                data_idx = entry_idx + entry.headerSize
                n_elements = 0
                if entry.elementSize > 0:
                    n_elements = entry.dataSize / entry.elementSize
                for n in range(n_elements):
                    if entry.pkgId == 9:  # RDB_PKG_ID_OBJECT_STATE
                        data = RDB_OBJECT_STATE_t.from_buffer(rdb_buff[data_idx:data_idx + sizeof(RDB_OBJECT_STATE_t)])
                        if data.base.name == self.EGO:
                            av_state['x'] = data.base.pos.x
                            av_state['y'] = data.base.pos.y
                            av_state['h'] = data.base.pos.h
                            av_state['v'] = data.ext.speed.y
                            av_state['a'] = data.ext.accel.y
                            # logging.error(self.EGO + ': ' + str(av_state))
                        if data.base.name == self.NEIGHBOR:
                            hv_state['x'] = data.base.pos.x
                            hv_state['y'] = data.base.pos.y
                            hv_state['h'] = data.base.pos.h
                            hv_state['v'] = data.ext.speed.y
                            hv_state['a'] = data.ext.accel.y
                            # logging.error(self.NEIGHBOR + ': ' + str(hv_state))
                    data_idx += entry.elementSize

                remain_bytes -= (entry.headerSize + entry.dataSize)
                if remain_bytes > 0:
                    entry_idx = entry_idx + (entry.headerSize + entry.dataSize)

            full_state = dict()
            if len(av_state) == 5 and (len(hv_state) == 5):
                    # logging.debug('Update ' + vehicle + ' vehicle state. Ego vehicle state: ' +
                    # str(len(vehicle_state)))
                    # full_state['sensor'] = sensor_state
                    full_state['av'] = av_state
                    full_state['hv'] = hv_state
                    state_q.put(full_state)
                    logging.debug(str(state_q.qsize()))
            # if len(hv_state) == 5:
            #     # logging.debug('Update ' + vehicle + ' vehicle state. HV vehicle state: ' +
            #     # str(len(vehicle_state)))
            #     full_state['hv'] = hv_state
            #     state_q.put(full_state)
            else:
                continue
            time.sleep(0.01)
