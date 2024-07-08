import socket
from datetime import datetime
from _Utils.logger import get_logger

def bytes_to_hex_list(byte_data):
    hex_list = []
    for byte in byte_data:
        hex_byte = format(byte, '02x')
        hex_list.append(hex_byte)
    return hex_list

def receive_rader_data(socket, message=None, buffer_size=128):
    if message != None:
        socket.send(message)
    response = socket.recv(buffer_size)
    return response

def Rader(rader_info):
    logger = get_logger(name = '[RADER]',console= True, file= False)
    try:
        ip = rader_info["ip"]
        port = rader_info["port"]
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((ip, port))
        message = b'\x33\x04\x04\xC4'
        try:
            response = receive_rader_data(sock)
            hex_data = bytes_to_hex_list(response)
            preprocess_data= hex_data
            preprocess_data = preprocess_data[:-2]
            preprocess_data = preprocess_data[7:]
            Track_Num = int(preprocess_data[0],16)
            Track_last_index = 1+Track_Num*5
            Track_list = preprocess_data[1:Track_last_index]
            Vital_list = preprocess_data[Track_last_index+1:]
            for i in range(0, len(Vital_list), 23):
                vital_id = int(Vital_list[i], 16)+1
                breath_rate_bytes = bytes.fromhex(Vital_list[i+1])
                heartbeat_rate_bytes = bytes.fromhex(Vital_list[i+10])
                breath_rate = int.from_bytes(breath_rate_bytes, byteorder='little', signed=False)
                heartbeat_rate = int.from_bytes(heartbeat_rate_bytes, byteorder='little', signed=False)
                if breath_rate != 0 and heartbeat_rate != 0:
                    current_datetime = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                    logger.debug((vital_id, heartbeat_rate, breath_rate, current_datetime, cctv_id))
                    # print(f"vital_id : {vital_id}")
                    # print(f"breath_rate : {breath_rate}")
                    # print(f"heartbeat_rate : {heartbeat_rate}")
                    # try:
                        # radar_queue.put((vital_id, heartbeat_rate, breath_rate, current_datetime, cctv_id))
                        # time.sleep(0.65)
                    # except Exception as e:
                        # print(f"Error in radar_queue.put: {e}")
        except Exception as e:
            logger.warning(f"IndexError occurred: {e}")
                   
    finally:
        message = b'\x33\x03\x04\xC3' # stop
        sock.send(message)
        response = sock.recv(128)
        message = b'\x33\x03\x04\xC5' # reboot
        sock.send(message)
        response = sock.recv(128)
        sock.close()