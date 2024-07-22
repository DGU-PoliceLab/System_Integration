import socket
from datetime import datetime
import time
from variable import get_debug_args
import numpy as np
import matplotlib.pyplot as plt
import cv2
from Utils.logger import get_logger
import time
import pymysql
import config
from Sensor import Rader, Thermal

class EdgeCam:
    def __init__(self, thermal_ip, thermal_port, rader_ip, rader_port, debug_args):
        self.thermal_ip = thermal_ip
        self.thermal_port = thermal_port
        self.rader_ip = rader_ip
        self.rader_port = rader_port
        self.thermal = Thermal(self.thermal_ip, self.thermal_port)
        self.rader = Rader(self.rader_ip, self.rader_port)
        self.logger = get_logger(name= '[EdgeCam]', console= True, file= False)
        self.data = {}

    def connect_rader(self):
        self.rader.connect()

    def disconnect_rader(self):
        self.rader.disconnect()

    def connect_thermal(self):
        self.thermal.connect()

    def disconnect_thermal(self):
        self.thermal.disconnect()

    def get_data(self, frame, tracks, detections):
        thermal_response, overlay_image = self.thermal.recevice(frame, detections)
        rader_response = self.rader.recevice(frame)
        self.logger.debug(thermal_response)
        self.logger.debug(rader_response)
        result = []
        for track in tracks:
            tid = track.track_id
            if tid not in self.data:
                self.data[tid] = {'tid': tid, 'temperature': None, 'breath': None, 'heart': None}
            x1, y1, x2, y2 = track.tlbr
            t_temp = []
            r_temp = []
            for td in thermal_response:
                pos = td['pos']
                if x1 <= pos[0] <= x2 and y1 <= pos[1] <= y2:
                    td['id'] = tid
                    td['score'] = abs((x1 + x2) / 2 - pos[0]) + abs((y1 + y2) / 2 - pos[1])
                    t_temp.append(td)
            for rd in rader_response:
                pos = rd['pos']
                if x1 <= pos[0] <= x2:
                    rd['id'] = tid
                    rd['score'] = abs((x1 + x2) / 2 - pos[0])
                    r_temp.append(rd)
            t_temp.sort(key= lambda x: x['score'])
            r_temp.sort(key= lambda x: x['score'])
            collect = {'tid': tid, 'temperature': None, 'breath': None, 'heart': None}
            if len(t_temp) > 0 and tid == t_temp[0]['id']:
                collect['temperature'] = td['temp']
            if len(r_temp) > 0 and tid == r_temp[0]['id']:
                collect['breath'] = rd['breath']
                collect['heart'] = rd['heart']
            if collect['temperature'] != None and collect['temperature'] != 0:
                self.data[tid]['temperature'] = collect['temperature']
            if collect['breath'] != None and collect['breath'] != 0:
                self.data[tid]['breath'] = collect['breath']
            if collect['heart'] != None and collect['heart'] != 0:
                self.data[tid]['heart'] = collect['heart']
            result.append(self.data[tid])
        return result, thermal_response, rader_response, overlay_image

