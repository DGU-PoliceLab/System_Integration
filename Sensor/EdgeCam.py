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
from Sensor import Rader, Thermal, CCTV
from Sensor import Rader, Thermal, CCTV

class EdgeCam:
    def __init__(self, thermal_ip, thermal_port, rader_ip, rader_port, debug_args):
        self.logger = get_logger(name= '[EdgeCam]', console= True, file= False)

        self.thermal_ip = thermal_ip
        self.thermal_port = thermal_port

        self.thermal = None
        if thermal_ip == None or thermal_port == None:
            self.thermal = Thermal.Thermal(self.thermal_ip, self.thermal_port, debug_args)
        self.thermal = None

        
        self.rader_ip = rader_ip
        self.rader_port = rader_port
        
        self.rader = Rader.Rader(self.rader_ip, self.rader_port, debug_args)
        self.cctv = CCTV.CCTV(debug_args)

        self.data = {}

    def get_cctv_info(self):
        return self.cctv.get_cctv_info()

    def connect_rader(self):
        self.rader.connect()

    def disconnect_rader(self):
        self.rader.disconnect()

    def connect_thermal(self):
        if self.thermal != None:
            self.thermal.connect()

    def disconnect_thermal(self):
        if self.thermal != None:
            self.thermal.disconnect()

    def get_data(self, frame, tracks, detections):
        thermal_response = []
        rader_response = []
        overlay_image = None

        if self.thermal != None:
            thermal_response, overlay_image = self.thermal.recevice(frame, detections)
            self.logger.debug(thermal_response)

    
        rader_response = self.rader.recevice(frame)
        self.logger.debug(rader_response)
        result = []
        for track in tracks:
            tid = track.track_id
            if tid not in self.data:
                self.data[tid] = {'tid': tid, 'temperature': None, 'breath': None, 'heart': None}
            x1, y1, x2, y2 = track.tlbr
            t_temp = []
            r_temp = []                       

            for rd in rader_response:
                pos = rd['pos']
                if x1 <= pos[0] <= x2:
                    rd['id'] = tid
                    rd['score'] = abs((x1 + x2) / 2 - pos[0])
                    r_temp.append(rd)
            r_temp.sort(key= lambda x: x['score'])
            collect = {'tid': tid, 'temperature': None, 'breath': None, 'heart': None}

            # if len(t_temp) > 0 and tid == t_temp[0]['id']:
            #     collect['temperature'] = td['temp']
            # if collect['temperature'] != None and collect['temperature'] != 0:
            #     self.data[tid]['temperature'] = collect['temperature']

            if len(r_temp) > 0 and tid == r_temp[0]['id']:
                collect['breath'] = rd['breath']
                collect['heart'] = rd['heart']               
            if collect['breath'] != None and collect['breath'] != 0:
                self.data[tid]['breath'] = collect['breath']
            if collect['heart'] != None and collect['heart'] != 0:
                self.data[tid]['heart'] = collect['heart']
            result.append(self.data[tid])
        return result, thermal_response, rader_response, overlay_image

