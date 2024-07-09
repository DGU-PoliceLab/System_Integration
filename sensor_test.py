import time
from datetime import datetime
import struct
import atexit
import socket
from _Sensor.sensor import Sensor
from variable import get_debug_args

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

def exit_handler(sock):
    message = b'\x33\x03\x04\xC3' # stop
    sock.send(message)
    response = sock.recv(128)
    message = b'\x33\x03\x04\xC5' # reboot
    sock.send(message)
    response = sock.recv(128)
    sock.close()

def rader_conn_test(rader_info):
    ip = rader_info["ip"]
    port = rader_info["port"]
    print(ip, port)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((ip, port))
    message = b'\x33\x04\x04\xC4'
    try:
        cnt = 0
        while cnt < 10:
            # print(cnt)
            response = receive_rader_data(sock)
            hex_data = bytes_to_hex_list(response)
            print(len(hex_data))
            # print(hex_data[2, 28])
            # print(hex_data[28, -1])
            # print(hex_data)
            # preprocess_data= hex_data
            # preprocess_data = preprocess_data[:-2]
            # preprocess_data = preprocess_data[7:]
            # track_num = int(preprocess_data[0],16)
            # track_last_index = 1+track_num*5
            # track_list = preprocess_data[1:track_last_index]
            # vital_list = preprocess_data[track_last_index+1:]
            # print("Track")
            # for i in range(0, len(track_list), 5):
            #     track_segment = bytes.fromhex(''.join(track_list[i:i+5]))
            #     (tid, pos_x, pos_y) = struct.unpack('=BhH', track_segment)
            #     pos_x /= 10.0
            #     pos_y /= 10.0
            #     if (pos_x <= 200 and pos_x >= -200) and pos_y < 20:
            #         print("tid:", tid, "x:", pos_x, "y:", pos_y)
            # print("Vital")
            # for i in range(0, len(vital_list), 23):
            #     if i + 10 < len(vital_list):
            #         vital_id = int(vital_list[i], 16)
            #         breath_rate_bytes = bytes.fromhex(vital_list[i+1])
            #         heartbeat_rate_bytes = bytes.fromhex(vital_list[i+10])
            #         breath_rate = int.from_bytes(breath_rate_bytes, byteorder='little', signed=False)
            #         heartbeat_rate = int.from_bytes(heartbeat_rate_bytes, byteorder='little', signed=False)
            #         if breath_rate != 0 and heartbeat_rate != 0:
            #             print("id:", vital_id, "heart:", heartbeat_rate, "breath", breath_rate)
            #         else:
            #             print("id:", vital_id, "heart:", heartbeat_rate, "breath", breath_rate)
            time.sleep(0.1)
            
            cnt += 1
    except Exception as e:
        print(f"IndexError occurred: {e}")
                   
    finally:
        exit_handler(sock)

def main():
    args = get_debug_args()
    thermal_info = {"ip": args.thermal_ip, "port": args.thermal_port}
    rader_info = {"ip": args.rader_ip, "port": args.rader_port}
    rader_conn_test(rader_info)

if __name__ == "__main__":
    main()