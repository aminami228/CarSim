import socket
import time
import numpy as np
import tty
import sys
import termios
import struct, collections
import select
from vires_types import *
from ctypes import *
from threading import Thread
global steer_increase
global acc_increase
acc_increase=0
steer_increase=0

def rdb_state():    
    #RDB connection settings
    RDB_PORT = 48190
    DEFAULT_BUFFER = 204800

    RDB_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    RDB_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    conn_err=RDB_sock.connect(('127.0.0.1', RDB_PORT))
    rdb_buf = bytearray(DEFAULT_BUFFER)
    
    #keyboard settings
    orig_settings = termios.tcgetattr(sys.stdin)
    tty.setraw(sys.stdin)
    x = 0
    
    #select input: keyboard/port message
    inputs=[RDB_sock,sys.stdin]
    outputs=[]
    
    def process_rdb_frame():
	rdb_hdr = RDB_MSG_HDR_t.from_buffer(rdb_buf[:sizeof(RDB_MSG_HDR_t)])
        if rdb_hdr.magicNo != RDB_MAGIC_NO:
            return
	entry_idx = rdb_hdr.headerSize
        n_remainingBytes = rdb_hdr.dataSize
        entry = RDB_MSG_ENTRY_HDR_t.from_buffer(rdb_buf[entry_idx:entry_idx+sizeof(RDB_MSG_HDR_t)])
	while n_remainingBytes > 0:

            # process data
            data_idx = entry_idx + entry.headerSize

            n_elements = 0
            if entry.elementSize > 0:
                n_elements = entry.dataSize / entry.elementSize

            #print n_elements
            #startCam=False

            for n in range(n_elements):
               if entry.pkgId == 26: #RDB_PKG_ID_SENSOR_OBJECT
                    data = RDB_DRIVER_CTRL_t.from_buffer(rdb_buf[data_idx:data_idx+sizeof(RDB_DRIVER_CTRL_t)]) 
                    global steer_increase
                    global acc_increase
		    #print data.steeringTgt, data.accelTgt
                    data.steeringTgt+=steer_increase
                    data.accelTgt+=acc_increase
		    #data.validityFlags |= RDB_DRIVER_INPUT_VALIDITY_NONE | RDB_DRIVER_INPUT_VALIDITY_FLAGS | RDB_PKG_ID_OPTIX_BUFFER|RDB_DRIVER_INPUT_VALIDITY_ADD_ON | RDB_DRIVER_INPUT_VALIDITY_MODIFIED | RDB_DRIVER_INPUT_VALIDITY_TGT_STEERING |RDB_DRIVER_INPUT_VALIDITY_TGT_ACCEL | RDB_DRIVER_INPUT_VALIDITY_STEERING_SPEED
		    #data.validityFlags=18658
		    print data.validityFlags
                    RDB_sock.send(bytearray(rdb_hdr) + bytearray(entry)+bytearray(data))
                    steer_increase=0
                    acc_increase=0

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

    while True:
        try:
            inputready,outputready,exceptready = select.select(inputs, outputs, [])
        except select.error, e:
            break
        except socket.error, e:
            break
        
        for s in inputready:
            if s==RDB_sock:
                sensor_responses = {}
                n_bytes = RDB_sock.recv_into(rdb_buf) # blocking?
                process_rdb_frame()
            elif s==sys.stdin:
                global steer_increase
                global acc_increase
                x=sys.stdin.read(1)[0]
                print x
                if x=='r':
                    steer_increase=-0.02
                elif x=='l':
                    steer_increase+=0.02
                elif x=='a':
                    acc_increase=1
                elif x=='d':
                    acc_increase=-1
           
    
if __name__ == '__main__':
    rdb_state()
