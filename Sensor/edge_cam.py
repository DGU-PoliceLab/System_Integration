# import socket
# from datetime import datetime
# import time
# from variable import get_debug_args
# import numpy as np
# import matplotlib.pyplot as plt
# import cv2
# from Utils.logger import get_logger
# from variable import get_root_args, get_sort_args, get_scale_args, get_debug_args
# import time
# import pymysql

# class EdgeCam:
#     def __init__(self):
#         if debug_args.debug == True:
#             # DB 연결 및 CCTV 정보 조회
#             source = debug_args.source
#             cctv_info = dict()
#             cctv_info['id'] = debug_args.cctv_id
#             cctv_info['ip'] = debug_args.cctv_ip
#             cctv_info['name'] = debug_args.cctv_name
#             thermal_info = dict()
#             thermal_info['ip'] = debug_args.thermal_ip
#             thermal_info['port'] = debug_args.thermal_port
#             rader_data = None
#             with open(debug_args.rader_data, 'r') as f:
#                 rader_data = json.load(f)
#         else:
#             # DB 연결 및 CCTV 정보 조회
#             try:
#                 conn = connect_db("mysql-pls")
#                 if conn.open:
#                     if dict_args['video_file'] != "":
#                         cctv_info = get_cctv_info(conn)
#                 else:
#                     logger.warning('RUN-CCTV Database connection is not open.')
#                     cctv_info = {'cctv_id': 404}
#             except Exception as e:
#                 logger.warning(f'Unable to connect to database, error: {e}')
#                 cctv_info = {'cctv_id': 404}

#             cctv_info = cctv_info[1]
#             source = cctv_info['ip']
#             Process(target=object_snapshot_control, args=(object_snapshot_control_queue,)).start()
    
#         cctv_info = dict()
#         cctv_info['id'] = debug_args.cctv_id
#         cctv_info['ip'] = debug_args.cctv_ip
#         cctv_info['name'] = debug_args.cctv_name

#         thermal_info = dict()
#         thermal_info['ip'] = debug_args.thermal_ip
#         thermal_info['port'] = debug_args.thermal_port

#         rader_data = None
#         pass

#     def getRGB(self):
#         pass
	
#     def getThermal(self):
#         pass
    
#     def getRadar(self):
#         pass
    
#     def getCompose(self):
#         radar_data = []
#         emotion_result_data = []

#         # 레이더와 이모션 결과를 정합해서 realtime_queue에 넣음
#         # def collect_realtime(radar_queue=None, emotion_queue=None, realtime_queue=None):
#             # assert radar_queue is None and emotion_queue is None:, 'At least one parameter is required(rader or emotion)'
#         while True:
#             if not radar_queue.empty():
#                 vital_id, heartbeat_rate, breath_rate, current_datetime, cctv_id = radar_queue.get()
#                 print(f"vital_id : {vital_id}, heartbeat_rate : {heartbeat_rate}, breath_rate : {breath_rate}, current_datetime : {current_datetime}, cctv_id : {cctv_id}")

#                 #     vital_id, heartbeat_rate, breath_rate, current_datetime, cctv_id = radar_queue.get()

#                 if not emotion_queue.empty() and radar_data is not None:
#                     print("Start Synchronizing")
#                     try:
#                         emotion_data = emotion_queue.get()
#                         print(f"emotion_data : {emotion_data}")
#                     except Exception as e:
#                         emotion_data = None
#                         print(e)                                                   
#                     # if radar_data is not None and emotion_data is not None:
#                         # matched_data = (emotion_data['current_datetime'], emotion_data['id'], rader_data[1], rader_data[2], emotion_data['mapped_emotion_results'][0], rader_data[3])
#             else:
#                 time.sleep(0.00001)
#         pass



#     # Get CCTV info (include rader, thermal) from database.
#     def get_device_info(conn: pymysql.connections.Connection):
#         cctv_info = []

#         with conn.cursor() as cur:
#             sql = "SELECT * FROM cctv"
#             cur.execute(sql)
#             response = cur.fetchall()
        
#         for row in response:
#             cctv_info.append({'cctv_id': row[0], 'cctv_ip': row[1], 'rader_ip': row[2], 'thermal_ip': row[3], 'cctv_name': row[4]})

#         return cctv_info
#         pass

# class Camera:
#     def __init__(self):
#         pass

# class Radar:
#     logger = get_logger(name = '[RADER]',console= False, file= False)

#     def __init__(self):
    
#         pass

#     def request(self, socket, message=None, buffer_size=128):
#         ret = {"status": "ERR"}
#         if message != None:
#             socket.send(message)
#         response = socket.recv(buffer_size)
#         timestemp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
#         ret = {"response": response, "timestemp":timestemp, "status": "OK"}
#         return ret
#         pass

#     def bytes_to_hex_list(self, byte_data):
#         hex_list = []
#         for byte in byte_data:
#             hex_byte = format(byte, '02x')
#             hex_list.append(hex_byte)
#         return hex_list
#         pass

#     def radar_start(self, radar_queue, Radar_IP, cctv_id):
#         try:
#             server_ip = Radar_IP
#             server_port = 5000
#             client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#             client_socket.connect((server_ip, server_port))
#             message = b'\x33\x04\x04\xC4'
#             last_time = time.time()

#             while True:
#                 try:
#                     response_dict = request(client_socket)
#                     hex_data = response_dict['response']
#                     hex_data = bytes_to_hex_list(hex_data)
#                     preprocess_data= hex_data
#                     preprocess_data = preprocess_data[:-2]
#                     preprocess_data = preprocess_data[7:]
#                     Track_Num = int(preprocess_data[0],16)
#                     Track_last_index = 1+Track_Num*5
#                     Track_list = preprocess_data[1:Track_last_index]
#                     Vital_list = preprocess_data[Track_last_index+1:]
#                     for i in range(0, len(Vital_list), 23):
#                         vital_id = int(Vital_list[i], 16)+1
#                         breath_rate_bytes = bytes.fromhex(Vital_list[i+1])
#                         heartbeat_rate_bytes = bytes.fromhex(Vital_list[i+10])
#                         breath_rate = int.from_bytes(breath_rate_bytes, byteorder='little', signed=False)
#                         heartbeat_rate = int.from_bytes(heartbeat_rate_bytes, byteorder='little', signed=False)
#                         if breath_rate != 0 and heartbeat_rate != 0:
#                             current_datetime = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
#                             # print(f"vital_id : {vital_id}")
#                             # print(f"breath_rate : {breath_rate}")
#                             # print(f"heartbeat_rate : {heartbeat_rate}")
#                             radar_queue.put((vital_id, heartbeat_rate, breath_rate, current_datetime, cctv_id))
#                             manage_queue_size(radar_queue, 500)
#                 except IndexError as e:
#                     # LOGGER.warning(f"IndexError occurred: {e}, skipping this loop iteration.")
#                     continue        
#         finally:
#             message = b'\x33\x03\x04\xC3' # stop
#             client_socket.send(message)
#             response = client_socket.recv(128)
#             message = b'\x33\x03\x04\xC5' # reboot
#             client_socket.send(message)
#             response = client_socket.recv(128)
#             client_socket.close()
#         pass

#     def draw(self, frame):
#         if num_frame < len(rader_data):
#             cur_rader_data = rader_data[num_frame]
#             vital_data = cur_rader_data["vital_info"]
#             target_data = []
#             for track in online_targets:
#                 tid = track.track_id
#                 x1, y1, x2, y2 = track.tlbr
#                 target_data.append({"id": tid, "range": [x1, x2, y1, y2]})

#             for vital in vital_data: 
#                 pos, depth = vital["pos"]
#                 heartbeat_rate = vital["heartbeat_rate"]
#                 breath_rate = vital["breath_rate"]
#                 offset = (int(pos) + 200) / 400 * int(w) # TODO 하드코딩 제거
#                 for target in target_data:
#                     tid = target["id"]
#                     pos_range = target["range"]
#                     if offset >= pos_range[0] and offset <= pos_range[1]:
#                         # logger.info(f"tid:{tid}, heartbeat_rate: {heartbeat_rate}, breath_rate: {breath_rate}")
#                         if debug_args.visualize:
#                             draw_frame = draw_vital.draw(draw_frame, int(pos_range[0]), int(pos_range[2]), heartbeat_rate, breath_rate)
#         pass

# class Thermal:
#     def __init__(self):
#         pass

#     def parse_header(self, data):
#         stx = data[0]
#         total_length = int.from_bytes(data[1:5], byteorder='big')
#         tx_sequence_number = data[5]
#         length = int.from_bytes(data[6:10], byteorder='big')
#         command = int.from_bytes(data[10:12], byteorder='big')
#         frame_rate = data[12]
#         resolution_x = int.from_bytes(data[13:15], byteorder='big')
#         resolution_y = int.from_bytes(data[15:17], byteorder='big')
#         vcm_temp_sensor = int.from_bytes(data[17:19], byteorder='big')
#         temp_mcu = int.from_bytes(data[19:21], byteorder='big')
#         temp_board = int.from_bytes(data[21:23], byteorder='big')
#         return (stx, total_length, tx_sequence_number, length, command, frame_rate,
#                 resolution_x, resolution_y, vcm_temp_sensor, temp_mcu, temp_board)
#         pass

#     def receive_thermal_data(self, sock, recv_size):
#         data = b""
#         while len(data) < recv_size:
#             packet = sock.recv(recv_size - len(data))
#             if not packet:
#                 break
#             data += packet
#         return data
#         pass

#     def receive_thermal_image(self, ip, port, logger):
#         sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         sock.connect((ip, port))
#         recv_size = 160640
#         response = receive_thermal_data(self, sock, recv_size)
#         thermal_image = None
#         if response:
#             (stx, total_length, tx_sequence_number, length, command, frame_rate,
#             resolution_x, resolution_y, vcm_temp_sensor, temp_mcu, temp_board) = parse_header(response)
#             logger.debug(f"STX: {stx}, Total Length: {total_length}, Tx Sequence Number: {tx_sequence_number}, Length: {length}, Command: {command}, Frame Rate: {frame_rate}, Resolution: {resolution_x}x{resolution_y}, VCM Temp Sensor: {vcm_temp_sensor}, Temp MCU: {temp_mcu}, Temp Board: {temp_board}")
            
#             header_size = 58
#             calibration_size = 320 * 10 * 2
#             image_size = 320 * 240 * 2
#             reserved_size = 320 - 58 - 1
#             etx_size = 1
#             image_data_start = header_size + calibration_size
#             image_data_end = image_data_start + image_size
#             image_data = response[image_data_start:image_data_end]

#             if len(image_data) == image_size:
#                 thermal_image = np.frombuffer(image_data, dtype=np.uint16).reshape((240, 320))
#             else:
#                 logger.warning("Error: Extracted image data size does not match the expected size.")

#         sock.close()
#         return thermal_image
#         pass

#     def overlay_images(self, rgb_image, thermal_image, detections, logger):
#         thermal_image_normalized = cv2.normalize(thermal_image, None, 0, 255, cv2.NORM_MINMAX)
#         thermal_image_colored = cv2.applyColorMap(thermal_image_normalized.astype(np.uint8), cv2.COLORMAP_JET)
#         thermal_image_colored = cv2.resize(thermal_image_colored, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_LINEAR)

#         rgb_height, rgb_width, _ = rgb_image.shape
            
#         overlay_image = rgb_image.copy()
#         x_offset = rgb_width - thermal_image_colored.shape[1] - 375
#         y_offset = int((rgb_height - thermal_image_colored.shape[0]) * 0.89)
        
#         overlay_image[y_offset:y_offset+thermal_image_colored.shape[0], x_offset:x_offset+thermal_image_colored.shape[1]] = thermal_image_colored
#         temperature = []
#         for i in range(detections.shape[0]):
#             x1, y1, x2, y2 = detections[i][0:4]
#             x1 = int(x1)
#             y1 = int(y1)
#             x2 = int(x2)
#             y2 = int(y2)

#             thermal_x1 = max(0, int((x1 - x_offset) / 2.5))
#             thermal_y1 = max(0, int((y1 - y_offset) / 2.5))
#             thermal_x2 = min(thermal_image.shape[1], int((x2 - x_offset) / 2.5))
#             thermal_y2 = min(thermal_image.shape[0], int((y2 - y_offset) / 2.5))

#             if thermal_x2 > thermal_x1 and thermal_y2 > thermal_y1:
#                 thermal_box = thermal_image[thermal_y1:thermal_y2, thermal_x1:thermal_x2]
#                 avg_pixel_value = np.mean(thermal_box)
#                 skin_surface_temp = (avg_pixel_value - 5500) / 100
#                 logger.info(f"Detection {i}: Average Pixel Value = {avg_pixel_value}, Skin Surface Temperature = {skin_surface_temp:.2f}°C")
#                 temperature.append({"id": i, "temp": skin_surface_temp})
#         return temperature
#         pass


#     def get_Thermal(self, hermal_info, frame, detections):
#         logger = get_logger(name= '[THERMAL]', console= True, file= False)
#         rgb_image = frame
#         thermal_image = receive_thermal_image(thermal_info.ip, thermal_info.port, logger)
#         temperature = overlay_images(rgb_image, thermal_image, detections, logger)
#         return temperature
#         pass
    
#     def draw(self, frame):
#         # temperature = Thermal(thermal_info, frame, face_detections)

#         pass