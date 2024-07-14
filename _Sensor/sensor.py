from _Sensor.rader import Rader
from _Sensor.thermal import Thermal
from _Utils.logger import get_logger
from variable import get_debug_args

class Sensor():
    def __init__(self, thermal_ip, thermal_port, rader_ip, rader_port):
        self.thermal_ip = thermal_ip
        self.thermal_port = thermal_port
        self.rader_ip = rader_ip
        self.rader_port = rader_port
        self.thermal = Thermal(self.thermal_ip, self.thermal_port)
        self.rader = Rader(self.rader_ip, self.rader_port)
        self.logger = get_logger(name= '[SENSOR]', console= True, file= False)

    def connect_rader(self):
        self.rader.connect()

    def disconnect_rader(self):
        self.rader.disconnect()

    def connect_thermal(self):
        self.thermal.connect()

    def disconnect_thermal(self):
        self.thermal.disconnect()

    def get_data(self, frame, detections):
        thermal_response, overlay_image = self.thermal.recevice(frame, detections)
        rader_response = self.rader.recevice(frame)
        return thermal_response, rader_response, overlay_image