import socket
import numpy as np
import matplotlib.pyplot as plt
import cv2
from _Utils.logger import get_logger
from variable import get_thermal_args

class Thermal():

    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.sock = None
        self.buffer_size = 160640
        self.logger = get_logger(name= '[THERMAL]', console= True, file= False)

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.ip, self.port))
        self.logger.debug(f"Connect to thermal({self.ip}:{self.port})")

    def disconnect(self):
        self.sock.close()
        self.logger.debug(f"Disconnect to thermal({self.ip}:{self.port})")

    def parse_header(self, data):
        stx = data[0]
        total_length = int.from_bytes(data[1:5], byteorder='big')
        tx_sequence_number = data[5]
        length = int.from_bytes(data[6:10], byteorder='big')
        command = int.from_bytes(data[10:12], byteorder='big')
        frame_rate = data[12]
        resolution_x = int.from_bytes(data[13:15], byteorder='big')
        resolution_y = int.from_bytes(data[15:17], byteorder='big')
        vcm_temp_sensor = int.from_bytes(data[17:19], byteorder='big')
        temp_mcu = int.from_bytes(data[19:21], byteorder='big')
        temp_board = int.from_bytes(data[21:23], byteorder='big')
        return (stx, total_length, tx_sequence_number, length, command, frame_rate,
                resolution_x, resolution_y, vcm_temp_sensor, temp_mcu, temp_board)

    def recv_raw_data(self):
        response = b''
        while len(response) < self.buffer_size:
            packet = self.sock.recv(self.buffer_size)
            if not packet:
                break
            response += packet
        self.logger.debug(f"Response from thermal({self.ip}:{self.port}): {response}")
        return response
    
    def bytes_to_thermal_img(self, byte_data):
        header_size = 58
        calibration_size = 320 * 10 * 2
        image_size = 320 * 240 * 2

        img_start_idx = header_size + calibration_size
        img_end_idx = img_start_idx + image_size
        image_data = byte_data[img_start_idx:img_end_idx]
        thermal_img = np.frombuffer(image_data, dtype=np.uint16).reshape((240, 320))

        return thermal_img
    
    def get_temperature(self, rgb_img, thermal_img, detections):
        thermal_img_normalized = cv2.normalize(thermal_img, None, 0, 255, cv2.NORM_MINMAX)
        thermal_img_colored = cv2.applyColorMap(thermal_img_normalized.astype(np.uint8), cv2.COLORMAP_JET)

        def get_overlay_image(thermal_img_colored):
            args = get_thermal_args()
            thermal_img_colored = cv2.resize(thermal_img_colored, None, fx=args.scale_ratio, fy=args.scale_ratio, interpolation=cv2.INTER_LINEAR)
            rgb_height, rgb_width, _ = rgb_img.shape
            overlay_image = rgb_img.copy()
            x_offset = int((rgb_width - thermal_img_colored.shape[1]) * args.offset_x)
            y_offset = int((rgb_height - thermal_img_colored.shape[0]) * args.offset_y)        
            overlay_image[y_offset:y_offset+thermal_img_colored.shape[0], x_offset:x_offset+thermal_img_colored.shape[1]] = thermal_img_colored
            return overlay_image, x_offset, y_offset
        
        overlay_image, x_offset, y_offset = get_overlay_image(thermal_img_colored)

        temperature = []
        for i in range(detections.shape[0]):
            x1, y1, x2, y2 = detections[i][0:4]
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            thermal_x1 = max(0, int((x1 - x_offset) / 2.5))
            thermal_y1 = max(0, int((y1 - y_offset) / 2.5))
            thermal_x2 = min(thermal_img.shape[1], int((x2 - x_offset) / 2.5))
            thermal_y2 = min(thermal_img.shape[0], int((y2 - y_offset) / 2.5))

            if thermal_x2 > thermal_x1 and thermal_y2 > thermal_y1:
                pos_x = int((x1 + x2) / 2)
                pos_y = int((y1 + y2) / 2)
                thermal_box = thermal_img[thermal_y1:thermal_y2, thermal_x1:thermal_x2]
                avg_pixel_value = np.mean(thermal_box)
                skin_surface_temp = (avg_pixel_value - 5500) / 100
                self.logger.debug(f"Detection {i}: Average Pixel Value = {avg_pixel_value}, Skin Surface Temperature = {skin_surface_temp:.2f}°C")
                temperature.append({'id': i, 'pos': (pos_x, pos_y), 'temp': skin_surface_temp})

        # return temperature
        return temperature, overlay_image

    def recevice(self, frame, detections):
        self.connect()
        raw_data = self.recv_raw_data()
        header = self.parse_header(raw_data)
        rgb_img = frame
        thermal_img = self.bytes_to_thermal_img(raw_data)
        temperature, pos = self.get_temperature(rgb_img, thermal_img, detections)
        self.disconnect()
        return temperature, pos