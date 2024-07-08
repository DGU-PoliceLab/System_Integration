from _Sensor.radar import Rader
from _Sensor.thermal import Thermal
from _Utils.logger import get_logger
from variable import get_debug_args

def Sensor(frame, detections):
    args = get_debug_args()
    thermal_info = {"ip": args.thermal_ip, "port": args.thermal_port}
    rader_info = {"ip": args.rader_ip, "port": args.rader_port}
    thermal_response = Thermal(thermal_info, frame, detections)
    rader_response = Rader(rader_info)
    print(thermal_response)
    print(rader_response)