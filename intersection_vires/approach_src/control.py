import socket
import time
import numpy as np

import struct, collections
from ctypes import *

SCP_MAGIC_NO = 40108


class SCP_MSG_HDR_t(Structure):
    _pack_ = 4
    _fields_ = [
        ('magicNo', c_ushort),
        ('version', c_ushort),
        ('sender', c_char*64),
        ('receiver', c_char*64),
        ('dataSize', c_uint)]

def ReStart():
    SCP_PORT = 48179
    RDB_PORT = 48195
    RDB_CONTROL_PORT = 48190
    DEFAULT_BUFFER = 204800

    SCP_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    SCP_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    SCP_sock.connect(('127.0.0.1', SCP_PORT))
    scp_buf = bytearray(DEFAULT_BUFFER)
    scp_msg = SCP_MSG_HDR_t()
    scp_msg.magicNo = SCP_MAGIC_NO
    scp_msg.version = 1
    scp_msg.sender = "python_scp"
    scp_msg.receiver = "any"

    msg_text = "<SimCtrl><Stop/></SimCtrl>"
    scp_msg.dataSize = len(msg_text)
    SCP_sock.send(bytearray(scp_msg) + bytearray(msg_text))

    #msg_text = "<SimCtrl><LoadScenario filename=\"/home/cmu/Software/VTD.2.0/Data/Projects/Current/Scenarios/test.xml\"/></SimCtrl>"
    msg_text ="<SimCtrl><LoadScenario filename=\"/home/member/Documents/VTD/Data/Projects/Current/Scenarios/test.xml\"/></SimCtrl>"
    scp_msg.dataSize = len(msg_text)
    SCP_sock.send(bytearray(scp_msg) + bytearray(msg_text))

    msg_text = "<SimCtrl><Start mode=\"operation\"/></SimCtrl>"
    scp_msg.dataSize = len(msg_text)
    SCP_sock.send(bytearray(scp_msg)
                  + bytearray(msg_text))

    speed=0
    msg_text = "<EgoCtrl><Speed value=%lf/></EgoCtrl>"%speed
    scp_msg.dataSize = len(msg_text)
    SCP_sock.send(bytearray(scp_msg) + bytearray(msg_text))

    msg_text = "<Camera name=\"VIEW_CAMERA\" showOwner=\"true\">\
    <Frustum far=\"1500.000000\" fovHor=\"40.000000\" fovVert=\"30.000000\" near=\"1.000000\" offsetHor=\"0.000000\" offsetVert=\"0.000000\" />\
    <PosRelative player=\"AV\" dx=\"-30\" dy=\"0.000000\" dz=\"25\"/>\
    <ViewRelative dh=\"0.000000\" dp=\"0.6\" dr=\"0\" />\
    <Set target=\"AV\"/></Camera>"
    scp_msg.dataSize = len(msg_text)
    SCP_sock.send(bytearray(scp_msg) + bytearray(msg_text))


def SimStep(speed):
    SCP_PORT = 48179
    DEFAULT_BUFFER = 204800

    SCP_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    SCP_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    SCP_sock.connect(('127.0.0.1', SCP_PORT))
    scp_buf = bytearray(DEFAULT_BUFFER)
    scp_msg = SCP_MSG_HDR_t()
    scp_msg.magicNo = SCP_MAGIC_NO
    scp_msg.version = 1
    scp_msg.sender = "python_scp"
    scp_msg.receiver = "any"
    print("speed:",speed)

    msg_text = "<SimCtrl><Start/></SimCtrl>"
    scp_msg.dataSize = len(msg_text)
    SCP_sock.send(bytearray(scp_msg) + bytearray(msg_text))

    msg_text = "<EgoCtrl><Speed value=%lf/></EgoCtrl>"%speed
    scp_msg.dataSize = len(msg_text)
    SCP_sock.send(bytearray(scp_msg) + bytearray(msg_text))

    msg_text = "<SimCtrl><Step/></SimCtrl>"
    scp_msg.dataSize = len(msg_text)
    SCP_sock.send(bytearray(scp_msg) + bytearray(msg_text))


def SimStepAcceleration(acceleration):
    SCP_PORT = 48179
    DEFAULT_BUFFER = 204800

    SCP_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    SCP_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    SCP_sock.connect(('127.0.0.1', SCP_PORT))
    scp_buf = bytearray(DEFAULT_BUFFER)
    scp_msg = SCP_MSG_HDR_t()
    scp_msg.magicNo = SCP_MAGIC_NO
    scp_msg.version = 1
    scp_msg.sender = "python_scp"
    scp_msg.receiver = "any"
    print("acceleration:",acceleration)

    msg_text = "<SimCtrl><Start/></SimCtrl>"
    scp_msg.dataSize = len(msg_text)
    SCP_sock.send(bytearray(scp_msg) + bytearray(msg_text))

    msg_text = "<Dynamics><Driver maxAccelLong=\"30\"/></Dynamics>"
    scp_msg.dataSize = len(msg_text)
    SCP_sock.send(bytearray(scp_msg) + bytearray(msg_text))

    msg_text = "<Player><DriverBehavior desiredAcc=%lf/></Player>"%max(min(acceleration/300,1),-1)
    scp_msg.dataSize = len(msg_text)
    SCP_sock.send(bytearray(scp_msg) + bytearray(msg_text))

    msg_text = "<SimCtrl><Step/></SimCtrl>"
    scp_msg.dataSize = len(msg_text)
    SCP_sock.send(bytearray(scp_msg) + bytearray(msg_text))

if __name__ == '__main__':
    ReStart()
