import zmq
import json
import numpy as np
from struct import *

class Server():
    def __init__(self, port, obsize, num_of_flows):
        self.m_port = port
        self.m_num_of_flows = num_of_flows
        self.m_context = zmq.Context()
        self.m_socket = self.m_context.socket(zmq.REQ)
        self.m_obsize = obsize
        self.m_connected = False 
    
    def _recv(self):
        recv = self.m_socket.recv()
        return recv

    def communicate(self, action):
        if not self.m_connected:
            self.m_socket = self.m_context.socket(zmq.REQ)
            self.m_socket.connect("tcp://localhost:"+ str(self.m_port))
            self.m_socket.setsockopt(zmq.RCVTIMEO, 10000)
            print("NS3 Connected")
            self.m_connected = True
        while True:
            try:
                if action is None:
                    self.m_socket.send(b"reset");
                    recv = self._recv()
                else:
                    action_np = action.tolist()
                    dic = {}
                    dic["action"] = action_np
                    j = json.dumps(dic)
                    self.m_socket.send(j.encode())
                    recv = self._recv()
            except zmq.error.Again as e:
                self.m_socket.disconnect("tcp://localhost:" + str(self.m_port))
                self.m_connected = False
                print("time out disconnected")
                return None, None, None, True
            break
            #self.m_socket.disconnect("tcp://localhost:" + str(self.m_port))
        dict = json.loads(recv.decode("utf-8"))

        obs = dict["obs"]
        reward = dict["reward"]
        done = dict["done"]
        if done == 0:
            done = False
            x = np.array(obs)
        else:
            done = True
            x = np.zeros((self.m_num_of_flows, 6));
            self.m_socket.disconnect("tcp://localhost:" + str(self.m_port))
            print("NS3 DisConnected")
            self.m_connected = False

        return x, reward, done, False


    def byteToint(self, byte):
        result = int.from_bytes(byte, byteorder='big', signed=True)
        return result
