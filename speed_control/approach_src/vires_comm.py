import socket
import time
import numpy as np
import vires_types as vt
import xml.etree.ElementTree as etree
from threading import Thread
from ctypes import sizeof
import logging

__author__ = {'qzq', 'shuang'}


DEFAULT_BUFFER = 204800
RDB_PORT = 48195
RDB_MAGIC_NO = 35712
RDB_CONTROL_PORT = 48190
SCP_PORT = 48179
SCP_MAGIC_NO = 40108
# RDB_PORT = 35712
# SCP_PORT = 40108
# RDB_PORT = 48195
# SCP_PORT = 48195
# RDB_PORT = 48195

# name of agent in Vires scenario
AV_NAME = "AV"
NEIGHBOR_NAME = "New Player 02"

# we can query for scenario details
SCENE_FILE = 'intersection_speed_control.xml'


class ViresComm(object):
    def __init__(self):
        self.collision = 0
        self.dest_pos = None
        self.scp_msg = vt.SCP_MSG_HDR_t()

        self.acc = 0.

    def create_sensors(self, scp_sock):
        """
        creates a new sensor via SCP
        use UDP here so all sensor messages use one port
        problem is that the channel gets flooded
        """
        self.scp_msg.magicNo = SCP_MAGIC_NO
        self.scp_msg.version = 1
        self.scp_msg.sender = "python_scp"
        self.scp_msg.receiver = "any"

        theta_d = list(np.arange(-120, 128, 8))
        for i, theta in enumerate(theta_d):
            msg_text = "<Sensor name=\"scpsensor" + str(i + 1) + "\" type=\"radar\" enable=\"true\">\
                    <Load     lib=\"libModuleSingleRaySensor.so\" path=\"/home/member/Documents/VTD/Data/Projects/Current/Plugins/ModuleManager\" persistent=\"true\" />\
                    <Frustum  near=\"0.0\" far=\"80.0\" left=\"1.0\" right=\"1.0\" bottom=\"0.05\" top=\"0.05\" />\
                    <Cull     maxObjects=\"1\" enable=\"true\" />\
                    <Port     name=\"RDBout\" number=\"48197\" type=\"UDP\" sendEgo=\"true\" />\
                    <Player   default=\"true\" />\
                    <Position dx=\"3.5\" dy=\"0.0\" dz=\"0.5\" dhDeg=\"" + str(theta) + "\" dpDeg=\"0.0\" drDeg=\"0.0\" />\
                    <Database resolveRepeatedObjects=\"true\" continuousObjectTesselation=\"2.0\" />\
                    <Filter   objectType=\"all\"/>\
                    <Debug    enable=\"false\" />\
                    <Config updateRatio=\"1.0\" useTimeServer=\"true\" />\
                    <Origin type=\"player\" />\
                    </Sensor>"
            self.scp_msg.dataSize = len(msg_text)
            scp_sock.send(bytearray(self.scp_msg) + bytearray(msg_text))

    @staticmethod
    def get_scp_msg(SCP_sock, scp_buf):
        """
        receive from socket and convert to elementTree
        loop just in case we don't get a proper message
        """
        while True:
            n_bytes = SCP_sock.recv_into(scp_buf)  # blocking
            scp_hdr = vt.SCP_MSG_HDR_t.from_buffer(scp_buf[:sizeof(vt.SCP_MSG_HDR_t)])
            if scp_hdr.magicNo != SCP_MAGIC_NO:
                continue
            data_idx = sizeof(vt.SCP_MSG_HDR_t)
            data = str(scp_buf[data_idx:data_idx + scp_hdr.dataSize - 1])
            try:
                msg_root = etree.fromstring(data)
                return msg_root
            except:
                continue

    def scp_query(self, query_txt, SCP_sock, scp_buf):
        self.scp_msg.magicNo = SCP_MAGIC_NO
        self.scp_msg.version = 1
        self.scp_msg.sender = "python_scp"
        self.scp_msg.receiver = "any"
        self.scp_msg.dataSize = len(query_txt)
        SCP_sock.send(bytearray(self.scp_msg) + bytearray(query_txt))

        """
        now wait for reply
        assume first reply we get is ours
        better way could be to check reply tag against query tag
        """
        while True:
            scp_msg_tree = self.get_scp_msg(SCP_sock, scp_buf)
            if scp_msg_tree.tag == 'Reply':
                return scp_msg_tree

    def scp_state(self):
        """
        create a separate SCP thread to detect collision
        not ideal but alternative is to open another RDB socket
        on 48190 where scoring info (with collision) is sent
        """

        """create scp connection to initialize sensors and get collision notification"""
        SCP_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        SCP_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        SCP_sock.connect(('127.0.0.1', SCP_PORT))
        scp_buf = bytearray(DEFAULT_BUFFER)

        """get goal location by querying scenario file"""
        scenario_query = "<Query entity=\"traffic\"><GetScenario filename=\"" + SCENE_FILE + "\"/></Query>"
        scenario_reply = self.scp_query(scenario_query, SCP_sock, scp_buf)

        """get goal location on path for our AV"""
        # for player in scenario_reply.iter('Player'):
        #     if player.find('Description').attrib['Name'] == AV_NAME:
        #         global dest_pos
        #         dest_pos = player.find('./Init/PathRef').attrib['TargetS']

        while True:
            scp_msg_tree = self.get_scp_msg(SCP_sock, scp_buf)
            """collision check"""
            if scp_msg_tree.find('Collision') is not None:
                if scp_msg_tree.tag == 'Player' and scp_msg_tree.attrib['name'] == 'AV':
                    logging.error('COLLISION!!')
                    # will remain flagged until the action thread resets it
                    self.collision = 1
            time.sleep(0)

    @staticmethod
    def state_rdb_frame(rdb_buf, vehicle_state, sensor_response):
        """
        extract sensor obj info, vehicle position, & position in path
        """
        rdb_hdr = vt.RDB_MSG_HDR_t.from_buffer(rdb_buf[:sizeof(vt.RDB_MSG_HDR_t)])
        if rdb_hdr.magicNo != RDB_MAGIC_NO:
            return
        entry_idx = rdb_hdr.headerSize
        n_remainingBytes = rdb_hdr.dataSize
        entry = vt.RDB_MSG_ENTRY_HDR_t.from_buffer(rdb_buf[entry_idx:entry_idx + sizeof(vt.RDB_MSG_HDR_t)])
        while n_remainingBytes > 0:
            """process data"""
            data_idx = entry_idx + entry.headerSize
            n_elements = 0
            if entry.elementSize > 0:
                n_elements = entry.dataSize / entry.elementSize
            for n in range(n_elements):
                if entry.pkgId == 9:           # RDB_PKG_ID_OBJECT_STATE
                    data = vt.RDB_OBJECT_STATE_t.from_buffer(rdb_buf[data_idx:data_idx + sizeof(vt.RDB_OBJECT_STATE_t)])
                    # print data.base.id
                    if data.base.name == AV_NAME:
                        vehicle_state['x'] = data.base.pos.x
                        vehicle_state['y'] = data.base.pos.y
                        vehicle_state['h'] = data.base.pos.h
                        vehicle_state['v'] = data.ext.speed.y
                        vehicle_state['a'] = data.ext.accel.y
                    # elif entry.pkgId == 26:
                    #     data = RDB_DRIVER_CTRL_t.from_buffer(rdb_buf[data_idx:data_idx+sizeof(RDB_DRIVER_CTRL_t)])
                    #     vehicle_state['accel'] = data.accelTgt
                elif entry.pkgId == 17:        # RDB_PKG_ID_SENSOR_OBJECT
                    data = vt.RDB_SENSOR_OBJECT_t.from_buffer(rdb_buf[data_idx:data_idx + sizeof(vt.RDB_SENSOR_OBJECT_t)])
                    sensor_response[data.sensorPos.h] = data.dist
                data_idx = data_idx + entry.elementSize

            n_remainingBytes = n_remainingBytes - (entry.headerSize + entry.dataSize)
            if n_remainingBytes > 0:
                entry_idx = entry_idx + (entry.headerSize + entry.dataSize)
                entry = vt.RDB_MSG_ENTRY_HDR_t.from_buffer(rdb_buf[entry_idx:entry_idx + sizeof(vt.RDB_MSG_HDR_t)])

    def action_rdb_frame(self, RDB_sock, rdb_buf, action_q, neighbor_state_q):
        """
        extract sensor obj info, vehicle position, & position in path
        """
        rdb_hdr = vt.RDB_MSG_HDR_t.from_buffer(rdb_buf[:sizeof(vt.RDB_MSG_HDR_t)])
        if rdb_hdr.magicNo != RDB_MAGIC_NO:
            return
        entry_idx = rdb_hdr.headerSize
        n_remainingBytes = rdb_hdr.dataSize
        entry = vt.RDB_MSG_ENTRY_HDR_t.from_buffer(rdb_buf[entry_idx:entry_idx + sizeof(vt.RDB_MSG_HDR_t)])
        while n_remainingBytes > 0:
            data_idx = entry_idx + entry.headerSize
            n_elements = 0
            if entry.elementSize > 0:
                n_elements = entry.dataSize / entry.elementSize
            full_state = {}
            human_vehicle_state = {}
            for n in range(n_elements):
                if entry.pkgId == 9:  # RDB_PKG_ID_OBJECT_STATE
                    data = vt.RDB_OBJECT_STATE_t.from_buffer(rdb_buf[data_idx:data_idx + sizeof(vt.RDB_OBJECT_STATE_t)])
                    if data.base.name == NEIGHBOR_NAME:
                        human_vehicle_state['x'] = data.base.pos.x
                        human_vehicle_state['y'] = data.base.pos.y
                        human_vehicle_state['h'] = data.base.pos.h
                        human_vehicle_state['v'] = data.ext.speed.y
                        human_vehicle_state['a'] = data.ext.accel.y
                elif entry.pkgId == 26:  # RDB_PKG_ID_SENSOR_OBJECT
                    data = vt.RDB_DRIVER_CTRL_t.from_buffer(rdb_buf[data_idx:data_idx + sizeof(vt.RDB_DRIVER_CTRL_t)])
                    if not action_q.empty():
                        self.acc = action_q.get_nowait()
                        data.accelTgt = self.acc
                        RDB_sock.send(bytearray(rdb_hdr) + bytearray(entry) + bytearray(data))

                data_idx = data_idx + entry.elementSize

            if len(human_vehicle_state) == 5:
                full_state['position'] = human_vehicle_state
                neighbor_state_q.put(full_state)
            n_remainingBytes = n_remainingBytes - (entry.headerSize + entry.dataSize)
            if n_remainingBytes > 0:
                entry_idx = entry_idx + (entry.headerSize + entry.dataSize)
                entry = vt.RDB_MSG_ENTRY_HDR_t.from_buffer(rdb_buf[entry_idx:entry_idx + sizeof(vt.RDB_MSG_HDR_t)])

    def vires_state(self, state_q):
        """
        worker function for main training script
        handles state info
        """

        RDB_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        """
        TCP_NODELAY is intended to disable/enable segment buffering
        so data can be sent out to peer as quickly as possible.
        This is typically used to improve network utilisation.
        """

        RDB_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        conn_err = RDB_sock.connect_ex(('127.0.0.1', RDB_PORT))
        rdb_buf = bytearray(DEFAULT_BUFFER)

        logging.info("vires state thread started")
        # expected num of laser scans
        n_scans = 3
        full_state = {}

        scp_state_thread = Thread(target=self.scp_state)
        # Make this a daemon so that Ctrl+C etc is obeyed by the main thread as well as these
        scp_state_thread.daemon = True
        scp_state_thread.start()
        while True:
            n_bytes = RDB_sock.recv_into(rdb_buf)      # blocking?
            sensor_response = {}
            vehicle_state = {}
            self.state_rdb_frame(rdb_buf, vehicle_state, sensor_response)

            # TODO: figure out how to detect goal reached
            """send state once we got all scans"""
            if len(sensor_response) < n_scans:
                continue
            # full_state['sensor'] = sensor_response
            full_state['position'] = vehicle_state
            # full_state['speed']=vehicle_speed
            full_state['collision'] = self.collision
            state_q.put(full_state)
            time.sleep(0)

    def vires_rdb_action(self, action_q, neighbor_state_q):
        """
        worker function for main training script
        handles actions (restarts also)
        """
        RDB_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        """
        TCP_NODELAY is intended to disable/enable segment buffering
        so data can be sent out to peer as quickly as possible.
        This is typically used to improve network utilisation.
        """
        RDB_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        conn_err = RDB_sock.connect_ex(('127.0.0.1', RDB_CONTROL_PORT))
        rdb_buf = bytearray(DEFAULT_BUFFER)

        """keep track of how many unique sensor readings we got - hack"""
        logging.info("vires state thread started")

        while True:
            n_bytes = RDB_sock.recv_into(rdb_buf)    # blocking?
            self.action_rdb_frame(RDB_sock, rdb_buf, action_q, neighbor_state_q)
            time.sleep(0)

    def scp_reset(self, scp_msg, SCP_sock):
        msg_text = "<EgoCtrl><Speed value=\"0.0\"/></EgoCtrl>"
        scp_msg.dataSize = len(msg_text)
        SCP_sock.send(bytearray(scp_msg) + bytearray(msg_text))
        msg_text = "<SimCtrl><Restart/></SimCtrl>"
        scp_msg.dataSize = len(msg_text)
        SCP_sock.send(bytearray(scp_msg) + bytearray(msg_text))
        time.sleep(1)
        msg_text = "<Traffic><Collision enable=\"true\"/></Traffic>"
        scp_msg.dataSize = len(msg_text)
        SCP_sock.send(bytearray(scp_msg) + bytearray(msg_text))
        self.collision = 0

    @staticmethod
    def scp_vel_ctrl(scp_msg, SCP_sock, speed):
        """only works for 'preparation' mode"""
        msg_text = "<EgoCtrl><Speed value=\"" + str(speed) + "\"/></EgoCtrl>"
        scp_msg.dataSize = len(msg_text)
        SCP_sock.send(bytearray(scp_msg) + bytearray(msg_text))

    def vires_scp_action(self, action_q):
        """
        worker function for main training script
        handles actions (restarts also)
        """
        SCP_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        """
        TCP_NODELAY is intended to disable/enable segment buffering
        so data can be sent out to peer as quickly as possible.
        This is typically used to improve network utilisation.
        """
        SCP_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        SCP_sock.connect(('127.0.0.1', SCP_PORT))
        scp_buf = bytearray(DEFAULT_BUFFER)
        scp_msg = vt.SCP_MSG_HDR_t()
        scp_msg.magicNo = SCP_MAGIC_NO
        scp_msg.version = 1
        scp_msg.sender = "python_scp"
        scp_msg.receiver = "any"

        """wait for actions from the agent"""
        while True:
            curr_action = action_q.get()  # blocking read
            if 'reset' in curr_action:
                self.scp_reset(scp_msg, SCP_sock)
            elif 'vel' in curr_action:
                self.scp_vel_ctrl(curr_action['vel'])
            time.sleep(0)
