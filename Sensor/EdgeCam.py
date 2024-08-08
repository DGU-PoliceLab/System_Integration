from Utils.logger import get_logger
from Sensor import rader, thermal, cctv

class EdgeCam:
    def __init__(self, thermal_ip = None, thermal_port = None, rader_ip = None, rader_port = None, toilet_rader_ip = None, toilet_rader_port = None, debug_args = None):
        self.logger = get_logger(name= '[EdgeCam]', console= False, file= False)
        self.logger.info(f"thermal: {thermal_ip}:{thermal_port}, rader: {rader_ip}:{rader_port}")
        if thermal_ip is not None:
            self.thermal = thermal.Thermal(thermal_ip, thermal_port, debug_args)  
        else:
            self.thermal = None
        if rader_ip is not None:
            self.rader = rader.Rader(rader_ip, rader_port, debug_args)
        else:  
            self.rader = None
        if toilet_rader_ip is not None:
            self.toilet_rader = rader.Rader(toilet_rader_ip, toilet_rader_port, debug_args)
        else:  
            self.toilet_rader = None
        self.data = {}

    def connect_rader(self):
        if self.rader != None:
            self.rader.connect()

    def disconnect_rader(self):
        if self.rader != None:
            self.rader.disconnect()

    def connect_toilet_rader(self):
        if self.toilet_rader != None:
            self.toilet_rader.connect()

    def disconnect_toilet_rader(self):
        if self.toilet_rader != None:
            self.toilet_rader.disconnect()

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
            thermal_response, overlay_image = self.thermal.receive(frame, detections)
            self.logger.debug(thermal_response)
        if self.rader != None:
            rader_response = self.rader.receive(frame)
            self.logger.debug(rader_response)
        if self.toilet_rader != None:
            toilet_rader_response = self.toilet_rader.receive(frame)
            self.logger.debug(toilet_rader_response)
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
            collect = {'tid': tid, 'temperature': None, 'breath': None, 'heart': None, 'r_x': None, 'r_y': None}

            if len(r_temp) > 0 and tid == r_temp[0]['id']:
                collect['breath'] = rd['breath']
                collect['heart'] = rd['heart']     
                collect['r_x'] = rd['pos'][0]           
                collect['r_y'] = rd['pos'][1]           
            if collect['breath'] != None and collect['breath'] != 0:
                self.data[tid]['breath'] = collect['breath']
            if collect['heart'] != None and collect['heart'] != 0:
                self.data[tid]['heart'] = collect['heart']
            if collect['r_x'] != None and collect['r_x'] != 0:
                self.data[tid]['r_x'] = collect['r_x']
            if collect['r_y'] != None and collect['r_y'] != 0:
                self.data[tid]['r_y'] = collect['r_y']
            result.append(self.data[tid])
            self.logger.info(f"{result}")
        if len(toilet_rader_response) > 0:
            toilet_data = {'tid': -1, 'breath': None, 'heart': None}
            rd = toilet_rader_response[0]
            toilet_data['breath'] = rd['breath']
            toilet_data['heart'] = rd['heart'] 
            result.append(toilet_data)
            
        return result, thermal_response, rader_response, overlay_image

