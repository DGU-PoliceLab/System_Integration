import socket
from datetime import datetime
from Utils.logger import get_logger

class Rader():

    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.sock = None
        self.message = None
        self.buffer_size = 128
        self.vital_info = {}
        self.logger = get_logger(name= '[RADER]', console= True, file= False)
    
    def connect(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.ip, self.port))
        self.logger.info(f"Connect to rader({self.ip}:{self.port})")
        self.sock = sock

    def disconnect(self):
        message = b'\x33\x03\x04\xC3' # stop
        self.sock.send(message)
        self.logger.debug(f"Send stop message to rader({self.ip}:{self.port}), message: {message}")
        response = self.sock.recv(self.buffer_size)
        self.logger.debug(f"Response of stop message from rader({self.ip}:{self.port}), response: {response}")
        message = b'\x33\x03\x04\xC5' # reboot
        self.sock.send(message)
        self.logger.debug(f"Send reboot message to rader({self.ip}:{self.port}), message: {message}")
        response = self.sock.recv(self.buffer_size)
        self.logger.debug(f"Response of reboot message from rader({self.ip}:{self.port}), response: {response}")
        self.sock.close()
        self.logger.info(f"Disconnect to rader({self.ip}:{self.port})")
    
    def byte_to_hex(self, byte_data):
        hex_data = []
        for byte in byte_data:
            hex_byte = format(byte, '02x')
            hex_data.append(hex_byte)
        return hex_data
    
    def hex_to_int(self, hex_data):
        hex_str = ''.join(hex_data)
        int_data = int(hex_str, 16)
        return int_data
    
    def recv_raw_data(self):
        response = self.sock.recv(self.buffer_size)
        self.logger.debug(f"Response from rader({self.ip}:{self.port}): {response}")
        return response
    
    def manage_keys(self, track_keys, vital_keys):
        matching_keys = []
        for key in vital_keys:
            if key not in track_keys:
                del self.vital_info[key]
            else:
                matching_keys.append(key)
        return matching_keys
    
    def recevice(self, frame):
        try:
            h, w, _  = frame.shape
            raw_data = self.recv_raw_data()
            hex_data = self.byte_to_hex(raw_data)
            device_id = self.hex_to_int(hex_data[:2])
            result = []
            if device_id != 0:
                track_num_idx = 2
                track_num = self.hex_to_int(hex_data[track_num_idx])
                vital_num_idx = 2 + track_num * 5 if 2 + track_num * 5 < 28 else 28
                vital_num = self.hex_to_int(hex_data[vital_num_idx])
                track_data = hex_data[track_num_idx + 1: vital_num_idx]
                vital_data = hex_data[vital_num_idx + 1: vital_num_idx + 1 + vital_num * 23]
                self.logger.debug(f"device_id: {device_id} track_num: {track_num} vital_num: {vital_num}")
                track_info = {}
                for idx in range(0, len(track_data), 5):
                    track_id = self.hex_to_int(track_data[idx])
                    pos_x = self.hex_to_int(track_data[idx + 1: idx + 3]) / 10
                    pos_y = self.hex_to_int(track_data[idx + 3: idx + 5]) / 10
                    if -200 <= pos_x <= 200 and pos_y < 15:
                        conv_pos_x = (pos_x + 200) / 400 * w
                        track_info[track_id] = {"track_id":track_id, "pos_x":conv_pos_x, "pos_y":pos_y}
                self.logger.debug(f"TRACK: {track_info}")
                self.logger.debug(f"VITAL: {self.vital_info}")
                vital_info = {}
                for idx in range(0, len(vital_data), 23):
                    if idx + 11 < len(vital_data):
                        vital_id = self.hex_to_int(vital_data[idx])
                        vital_breath = self.hex_to_int(vital_data[idx + 1])
                        vital_heart = self.hex_to_int(vital_data[idx + 11])
                        vital_info[vital_id] = {"vital_id":vital_id, "vital_breath":vital_breath, "vital_heart":vital_heart}
                self.vital_info = vital_info
                track_keys = list(track_info.keys())
                vital_keys = list(self.vital_info.keys())
                matching_keys = self.manage_keys(track_keys, vital_keys)
                
                for key in matching_keys:
                    self.logger.info(f"Matching Data: {track_info[key]}, {self.vital_info[key]}")
                    result.append({'id': key, 'pos': (track_info[key]['pos_x'],track_info[key]['pos_y']), 'breath': self.vital_info[key]['vital_breath'], 'heart': self.vital_info[key]['vital_heart']})
            return result

        except Exception as e:
            self.logger.debug(f"Error occured in rader: {e}")
            pass