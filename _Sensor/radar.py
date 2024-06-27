import socket
from datetime import datetime
#from time import time
from queue import Queue
import time
from _Utils.logger import get_logger
from _Utils.pipeline import manage_queue_size

LOGGER = get_logger(name = '[RADER]',console= False, file= False)


def request(socket, message=None, buffer_size=128):
    ret = {"status": "ERR"}
    if message != None:
        socket.send(message)
    response = socket.recv(buffer_size)
    timestemp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    ret = {"response": response, "timestemp":timestemp, "status": "OK"}
    return ret

def bytes_to_hex_list(byte_data):
    hex_list = []
    for byte in byte_data:
        hex_byte = format(byte, '02x')
        hex_list.append(hex_byte)
    return hex_list

def radar_start(radar_queue, Radar_IP, cctv_id):
    try:
        server_ip = Radar_IP
        server_port = 5000
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((server_ip, server_port))
        message = b'\x33\x04\x04\xC4'
        last_time = time.time()
        while True:
            try:
                response_dict = request(client_socket)
                hex_data = response_dict['response']
                hex_data = bytes_to_hex_list(hex_data)
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
                        # print(f"vital_id : {vital_id}")
                        # print(f"breath_rate : {breath_rate}")
                        # print(f"heartbeat_rate : {heartbeat_rate}")
                        try:
                            radar_queue.put((vital_id, heartbeat_rate, breath_rate, current_datetime, cctv_id))
                            # time.sleep(0.65)
                        except Exception as e:
                            print(f"Error in radar_queue.put: {e}")
                        #print(f"radar_queue.qsize() : {radar_queue.qsize()}")
                        manage_queue_size(radar_queue, 500)
            except IndexError as e:
                # LOGGER.warning(f"IndexError occurred: {e}, skipping this loop iteration.")
                continue        
    finally:
        message = b'\x33\x03\x04\xC3' # stop
        client_socket.send(message)
        response = client_socket.recv(128)
        message = b'\x33\x03\x04\xC5' # reboot
        client_socket.send(message)
        response = client_socket.recv(128)
        client_socket.close()

if __name__== "__main__":
    radar_start()

