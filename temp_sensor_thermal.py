import socket
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
from _Utils.logger import get_logger
from variable import get_debug_args

def parse_header(data):
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

def receive_thermal_data(sock, recv_size):
    data = b""
    while len(data) < recv_size:
        packet = sock.recv(recv_size - len(data))
        if not packet:
            break
        data += packet
    return data

def receive_thermal_image(ip, port, logger):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((ip, port))
    recv_size = 160640
    response = receive_thermal_data(sock, recv_size)
    thermal_image = None
    if response:
        (stx, total_length, tx_sequence_number, length, command, frame_rate,
        resolution_x, resolution_y, vcm_temp_sensor, temp_mcu, temp_board) = parse_header(response)
        logger.debug(f"STX: {stx}, Total Length: {total_length}, Tx Sequence Number: {tx_sequence_number}, Length: {length}, Command: {command}, Frame Rate: {frame_rate}, Resolution: {resolution_x}x{resolution_y}, VCM Temp Sensor: {vcm_temp_sensor}, Temp MCU: {temp_mcu}, Temp Board: {temp_board}")
        
        header_size = 58
        calibration_size = 320 * 10 * 2
        image_size = 320 * 240 * 2
        reserved_size = 320 - 58 - 1
        etx_size = 1
        image_data_start = header_size + calibration_size
        image_data_end = image_data_start + image_size
        image_data = response[image_data_start:image_data_end]

        if len(image_data) == image_size:
            thermal_image = np.frombuffer(image_data, dtype=np.uint16).reshape((240, 320))
        else:
            logger.warning("Error: Extracted image data size does not match the expected size.")

    sock.close()
    return thermal_image

def overlay_images(rgb_image, thermal_image, tracks, logger):
    thermal_image_normalized = cv2.normalize(thermal_image, None, 0, 255, cv2.NORM_MINMAX)
    thermal_image_colored = cv2.applyColorMap(thermal_image_normalized.astype(np.uint8), cv2.COLORMAP_JET)
    thermal_image_colored = cv2.resize(thermal_image_colored, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_LINEAR)

    rgb_height, rgb_width, _ = rgb_image.shape
        
    overlay_image = rgb_image.copy()
    x_offset = rgb_width - thermal_image_colored.shape[1] - 375
    y_offset = int((rgb_height - thermal_image_colored.shape[0]) * 0.89)
    
    overlay_image[y_offset:y_offset+thermal_image_colored.shape[0], x_offset:x_offset+thermal_image_colored.shape[1]] = thermal_image_colored
    temperature = []
    for i, track in enumerate(tracks):
        tlwh = track.tlwh
        tid = track.track_id
        x1 = int(tlwh[0])
        y1 = int(tlwh[1])
        x2 = int(tlwh[0] + tlwh[2])
        y2 = int(tlwh[1] + tlwh[3])
    # for i in range(tracks.shape[0]):
    #     x1, y1, x2, y2 = tracks[i][0:4]
    #     x1 = int(x1)
    #     y1 = int(y1)
    #     x2 = int(x2)
    #     y2 = int(y2)

        thermal_x1 = max(0, int((x1 - x_offset) / 2.5))
        thermal_y1 = max(0, int((y1 - y_offset) / 2.5))
        thermal_x2 = min(thermal_image.shape[1], int((x2 - x_offset) / 2.5))
        thermal_y2 = min(thermal_image.shape[0], int((y2 - y_offset) / 2.5))

        if thermal_x2 > thermal_x1 and thermal_y2 > thermal_y1:
            thermal_box = thermal_image[thermal_y1:thermal_y2, thermal_x1:thermal_x2]
            avg_pixel_value = np.mean(thermal_box)
            skin_surface_temp = (avg_pixel_value - 5500) / 100
            logger.info(f"Detection {i}: Average Pixel Value = {avg_pixel_value}, Skin Surface Temperature = {skin_surface_temp:.2f}°C")
            temperature.append({"id": i, "temp": skin_surface_temp})
    return temperature

def Thermal(thermal_ip, data_pipe, realtime_sensor_input_pipe):
    logger = get_logger(name= '[THERMAL]', console= False, file= False)
    data_pipe.send(True)
    while True:
        data = data_pipe.recv()
        tracks, meta_data, frame, num_frame = data
        # print(f"[THERMAL] thermal_queue size : {thermal_queue.qsize()}")
        rgb_image = frame
        thermal_image = receive_thermal_image(thermal_ip, 10603, logger)
        temperature = overlay_images(rgb_image, thermal_image, tracks, logger)
        realtime_sensor_input_pipe.send(temperature)
        # return temperature

