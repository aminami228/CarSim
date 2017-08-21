import socket
from vires_types import SCP_MSG_HDR_t
import logging


class SCPComm(object):
    BUFFER = 204800
    SCP_MAGIC_NO = 40108

    def __init__(self):
        # type: () -> object
        self.scp_port = 48179

    def restart_scp(self, speed):
        scp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        scp_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        scp_sock.connect(('127.0.0.1', self.scp_port))
        scp_buf = bytearray(self.BUFFER)
        scp_msg = SCP_MSG_HDR_t()
        scp_msg.magicNo = self.SCP_MAGIC_NO
        scp_msg.version = 1
        scp_msg.sender = "python_scp"
        scp_msg.receiver = "any"

        msg_text = "<SimCtrl><Stop/></SimCtrl>"
        scp_msg.dataSize = len(msg_text)
        scp_sock.send(bytearray(scp_msg) + bytearray(msg_text))

        msg_text = "<SimCtrl><LoadScenario " \
                   "filename=\"/home/scotty/Software/VTD/Data/Projects/Current/Scenarios/test.xml\"/></SimCtrl>"
        scp_msg.dataSize = len(msg_text)
        scp_sock.send(bytearray(scp_msg) + bytearray(msg_text))

        # msg_text = "<SimCtrl><Start mode=\"operation\"/></SimCtrl>"
        msg_text = "<SimCtrl><Start mode=\"preparation\"/></SimCtrl>"
        scp_msg.dataSize = len(msg_text)
        scp_sock.send(bytearray(scp_msg) + bytearray(msg_text))

        msg_text = "<EgoCtrl><Speed value=%lf/></EgoCtrl>"%speed
        scp_msg.dataSize = len(msg_text)
        scp_sock.send(bytearray(scp_msg) + bytearray(msg_text))

        msg_text = "<Camera name=\"VIEW_CAMERA\" showOwner=\"true\">\
        <Frustum far=\"1500.000000\" fovHor=\"40.000000\" fovVert=\"30.000000\" near=\"1.000000\" offsetHor=\"0.000000\"" \
                   " offsetVert=\"0.000000\" />\
        <PosRelative player=\"AV\" dx=\"-30\" dy=\"0.000000\" dz=\"25\"/>\
        <ViewRelative dh=\"0.000000\" dp=\"0.6\" dr=\"0\" />\
        <Set target=\"AV\"/></Camera>"
        scp_msg.dataSize = len(msg_text)
        scp_sock.send(bytearray(scp_msg) + bytearray(msg_text))

    def scp_control(self, speed):
        scp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        scp_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        scp_sock.connect(('127.0.0.1', self.scp_port))
        scp_buf = bytearray(self.BUFFER)
        scp_msg = SCP_MSG_HDR_t()
        scp_msg.magicNo = self.SCP_MAGIC_NO
        scp_msg.version = 1
        scp_msg.sender = "python_scp"
        scp_msg.receiver = "any"
        logging.debug("speed:" + str(speed))

        msg_text = "<SimCtrl><Start/></SimCtrl>"
        scp_msg.dataSize = len(msg_text)
        scp_sock.send(bytearray(scp_msg) + bytearray(msg_text))

        msg_text = "<EgoCtrl><Speed value=%lf/></EgoCtrl>" % speed
        scp_msg.dataSize = len(msg_text)
        scp_sock.send(bytearray(scp_msg) + bytearray(msg_text))

        msg_text = "<SimCtrl><Step/></SimCtrl>"
        scp_msg.dataSize = len(msg_text)
        scp_sock.send(bytearray(scp_msg) + bytearray(msg_text))


if __name__ == '__main__':
    a = SCPComm()
    a.restart_scp(0.)
