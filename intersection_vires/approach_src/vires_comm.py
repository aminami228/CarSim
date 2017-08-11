import socket
import time
import numpy as np

#import struct, collections
from ctypes import *
import Queue
from vires_types import *
import xml.etree.ElementTree as ET
from threading import Thread


DEFAULT_BUFFER = 204800
#RDB_PORT = 35712
#SCP_PORT = 40108
#RDB_PORT = 48195
#SCP_PORT = 48195
#RDB_PORT = 48195
RDB_PORT = 48195
RDB_CONTROL_PORT = 48190
SCP_PORT = 48179
#RDB_MAGIC_NO=35712

# name of agent in Vires scenario
AV_NAME = "AV"
NEIGHBOR_NAME = "New Player 02"

# lazy way to communicate between threads
collision_flag = 0
dest_pos = None

# we can query for scenario details
SCENE_FILE = 'test.xml'
#SCENE_FILE = 'traffic_demo.xml'
global acc
acc=0

# creates a new sensor via SCP
# use UDP here so all sensor messages use one port
# problem is that the channel gets flooded
def create_sensors(SCP_sock):
    scp_msg = SCP_MSG_HDR_t()
    scp_msg.magicNo = SCP_MAGIC_NO
    scp_msg.version = 1
    scp_msg.sender = "python_scp"
    scp_msg.receiver = "any"

    theta_d = np.append(np.arange(-120, 120, 8), 120)
    s_id = 1

    for theta in theta_d:
        msg_text = "<Sensor name=\"scpsensor" + str(s_id) + "\" type=\"radar\" enable=\"true\">\
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
        scp_msg.dataSize = len(msg_text)
        SCP_sock.send(bytearray(scp_msg) + bytearray(msg_text))
        s_id = s_id + 1

# create a separate SCP thread to detect collision
# not ideal but alternative is to open another RDB socket
# on 48190 where scoring info (with collision) is sent
def scp_state():
    # create scp connection to initialize sensors and get collision notification
    SCP_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    SCP_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    SCP_sock.connect(('127.0.0.1', SCP_PORT))
    scp_buf = bytearray(DEFAULT_BUFFER)

    def scp_query(query_txt):
        scp_msg = SCP_MSG_HDR_t()
        scp_msg.magicNo = SCP_MAGIC_NO
        scp_msg.version = 1
        scp_msg.sender = "python_scp"
        scp_msg.receiver = "any"
        scp_msg.dataSize = len(query_txt)
        SCP_sock.send(bytearray(scp_msg) + bytearray(query_txt))

        # now wait for reply
        # assume first reply we get is ours
        # better way could be to check reply tag against query tag
        while True:
            scp_msg_tree = get_scp_msg()

            if scp_msg_tree.tag == 'Reply':
                return scp_msg_tree


    # receive from socket and convert to elementTree
    def get_scp_msg():

        # loop just in case we don't get a proper message
        while True:
            n_bytes = SCP_sock.recv_into(scp_buf) # blocking
            scp_hdr = SCP_MSG_HDR_t.from_buffer(scp_buf[:sizeof(SCP_MSG_HDR_t)])

            if scp_hdr.magicNo != SCP_MAGIC_NO:
                continue

            data_idx = sizeof(SCP_MSG_HDR_t)
            data = str(scp_buf[data_idx:data_idx + scp_hdr.dataSize-1])

            try:
                msg_root = ET.fromstring(data)
                return msg_root
            except:
                continue


    # get goal location by querying scenario file
    scenario_query = "<Query entity=\"traffic\"><GetScenario filename=\"" + SCENE_FILE + "\"/></Query>"
    scenario_reply = scp_query(scenario_query)

    """
    # get goal location on path for our AV
    for player in scenario_reply.iter('Player'):
        if player.find('Description').attrib['Name'] == AV_NAME:

            global dest_pos
            dest_pos = player.find('./Init/PathRef').attrib['TargetS']
    """

    while True:

        scp_msg_tree = get_scp_msg()

        # collision check
        if scp_msg_tree.find('Collision') is not None:
            if scp_msg_tree.tag == 'Player' and scp_msg_tree.attrib['name'] == 'AV':
                print "COLLISION!!"

                # will remain flagged until the action thread resets it
                global collision_flag
                collision_flag = 1

        time.sleep(0)

# worker function for main training script
# handles state info
def vires_state(state_q):

    RDB_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # TCP_NODELAY is intended to disable/enable segment buffering
    # so data can be sent out to peer as quickly as possible.
    # This is typically used to improve network utilisation.
    RDB_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    conn_err = RDB_sock.connect_ex(('127.0.0.1', RDB_PORT))
    rdb_buf = bytearray(DEFAULT_BUFFER)
    print conn_err

    # keep track of how many unique sensor readings we got - hack
    sensor_responses = {}

    # x, y, theta (heading )
    vehicle_state = {}
    # vehicle_speed=0.0

    def process_rdb_frame():

        # extract sensor obj info, vehicle position, & position in path

        rdb_hdr = RDB_MSG_HDR_t.from_buffer(rdb_buf[:sizeof(RDB_MSG_HDR_t)])
        # print rdb_hdr.magicNo

        if rdb_hdr.magicNo != RDB_MAGIC_NO:
            return

        entry_idx = rdb_hdr.headerSize
        n_remainingBytes = rdb_hdr.dataSize
        entry = RDB_MSG_ENTRY_HDR_t.from_buffer(rdb_buf[entry_idx:entry_idx+sizeof(RDB_MSG_HDR_t)])
       # print entry_idx

        while n_remainingBytes > 0:

            # process data
            data_idx = entry_idx + entry.headerSize

            n_elements = 0
            if entry.elementSize > 0:
                n_elements = entry.dataSize / entry.elementSize

            #print n_elements

            for n in range(n_elements):

                if entry.pkgId == 9: # RDB_PKG_ID_OBJECT_STATE
                    data = RDB_OBJECT_STATE_t.from_buffer(rdb_buf[data_idx:data_idx+sizeof(RDB_OBJECT_STATE_t)])
                    #print data.base.id
                    if data.base.name == AV_NAME:
                        vehicle_state['x'] = data.base.pos.x
                        vehicle_state['y'] = data.base.pos.y
                        vehicle_state['h'] = data.base.pos.h
                        vehicle_state['v'] = data.ext.speed.y
                        vehicle_state['a'] = data.ext.accel.y
                # elif entry.pkgId == 26:
                #     data = RDB_DRIVER_CTRL_t.from_buffer(rdb_buf[data_idx:data_idx+sizeof(RDB_DRIVER_CTRL_t)])
	            # global acc
                #     vehicle_state['accel'] = data.accelTgt
                elif entry.pkgId == 17: #RDB_PKG_ID_SENSOR_OBJECT
                    data = RDB_SENSOR_OBJECT_t.from_buffer(rdb_buf[data_idx:data_idx+sizeof(RDB_SENSOR_OBJECT_t)])
                    sensor_responses[data.sensorPos.h] = data.dist
                data_idx = data_idx + entry.elementSize

            # advance in buffer
            n_remainingBytes = n_remainingBytes - (entry.headerSize + entry.dataSize)
            if n_remainingBytes > 0:
                entry_idx = entry_idx + (entry.headerSize + entry.dataSize)
                entry = RDB_MSG_ENTRY_HDR_t.from_buffer(rdb_buf[entry_idx:entry_idx+sizeof(RDB_MSG_HDR_t)])


    print "vires state thread started"
    # expected num of laser scans
    n_scans = 3

    full_state = {}
    global collision_flag

    scp_state_thread = Thread(target=scp_state)
    #Make this a daemon so that Ctrl+C etc is obeyed by the main thread as well as these
    scp_state_thread.daemon=True
    scp_state_thread.start()
    while True:

        sensor_responses = {}

        n_bytes = RDB_sock.recv_into(rdb_buf) # blocking?

        process_rdb_frame()

        #print "len(sensor_responses)="+str(len(sensor_responses))


        # ToDo: figure out how to detect goal reached

        # send state once we got all scans
        #print("length sensor responses:",len(sensor_responses))
        if len(sensor_responses) < n_scans:
            continue

        full_state['sensor'] = sensor_responses
        full_state['position'] = vehicle_state
        #full_state['speed']=vehicle_speed
        full_state['collision'] = collision_flag
        state_q.put(full_state)

        time.sleep(0)

# worker function for main training script
# handles actions (restarts also)
def vires_action(action_q, neighbor_state_q):
    RDB_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # TCP_NODELAY is intended to disable/enable segment buffering
    # so data can be sent out to peer as quickly as possible.
    # This is typically used to improve network utilisation.
    RDB_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    conn_err = RDB_sock.connect_ex(('127.0.0.1', RDB_CONTROL_PORT))
    rdb_buf = bytearray(DEFAULT_BUFFER)
    print "action error:",conn_err

    # keep track of how many unique sensor readings we got - hack
    sensor_responses = {}

    # x, y, theta (heading )
    vehicle_state = {}
    #vehicle_speed=0.0

    def process_rdb_frame():

        # extract sensor obj info, vehicle position, & position in path

        rdb_hdr = RDB_MSG_HDR_t.from_buffer(rdb_buf[:sizeof(RDB_MSG_HDR_t)])
        #print rdb_hdr.magicNo

        if rdb_hdr.magicNo != RDB_MAGIC_NO:
            return

        entry_idx = rdb_hdr.headerSize
        n_remainingBytes = rdb_hdr.dataSize
        entry = RDB_MSG_ENTRY_HDR_t.from_buffer(rdb_buf[entry_idx:entry_idx+sizeof(RDB_MSG_HDR_t)])
       # print entry_idx

        while n_remainingBytes > 0:

            # process data
            data_idx = entry_idx + entry.headerSize

            n_elements = 0
            if entry.elementSize > 0:
                n_elements = entry.dataSize / entry.elementSize

            #print n_elements
            global acc
            full_state = {}
            human_vehicle_state={}
            for n in range(n_elements):
                #print entry.pkgId
                if entry.pkgId == 9: # RDB_PKG_ID_OBJECT_STATE
                    data = RDB_OBJECT_STATE_t.from_buffer(rdb_buf[data_idx:data_idx+sizeof(RDB_OBJECT_STATE_t)])
                    #print data.base.id
                    if data.base.name == NEIGHBOR_NAME:
                        #print data.base.name
                        human_vehicle_state['x'] = data.base.pos.x
                        human_vehicle_state['y'] = data.base.pos.y
                        human_vehicle_state['h'] = data.base.pos.h
                        human_vehicle_state['v'] = data.ext.speed.y
                        human_vehicle_state['a'] = data.ext.accel.y

                elif entry.pkgId == 26: #RDB_PKG_ID_SENSOR_OBJECT
                    data = RDB_DRIVER_CTRL_t.from_buffer(rdb_buf[data_idx:data_idx+sizeof(RDB_DRIVER_CTRL_t)])
                    if not action_q.empty():
                        acc=action_q.get_nowait()
                        data.accelTgt=acc
                        RDB_sock.send(bytearray(rdb_hdr) + bytearray(entry)+bytearray(data))

                data_idx = data_idx + entry.elementSize
            if len(human_vehicle_state)==5:
                 full_state['position'] = human_vehicle_state
                 neighbor_state_q.put(full_state)
            # advance in buffer
            n_remainingBytes = n_remainingBytes - (entry.headerSize + entry.dataSize)
            if n_remainingBytes > 0:
                entry_idx = entry_idx + (entry.headerSize + entry.dataSize)
                entry = RDB_MSG_ENTRY_HDR_t.from_buffer(rdb_buf[entry_idx:entry_idx+sizeof(RDB_MSG_HDR_t)])


    print "vires state thread started"

    while True:

        sensor_responses = {}

        n_bytes = RDB_sock.recv_into(rdb_buf) # blocking?

        process_rdb_frame()

        time.sleep(0)
