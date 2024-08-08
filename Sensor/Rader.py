
import socket
from datetime import datetime
from Utils.logger import get_logger
import struct

class Rader:
    def __init__(self, ip, port, debug_args):
        self.ip = ip
        self.port = port
        self.sock = None
        self.buffer_size = 144  # 기본 패킷 크기 설정, 필요 시 조정 가능
        self.logger = get_logger(name='[RADER]', console=False, file=False)
    
    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.ip, self.port))
        self.logger.info(f"Connect to rader({self.ip}:{self.port})")

    def disconnect(self):
        message = b'\x33\x03\x04\xC3'  # stop
        self.sock.send(message)
        self.logger.debug(f"Send stop message to rader({self.ip}:{self.port}), message: {message}")
        response = self.sock.recv(self.buffer_size)
        self.logger.debug(f"Response of stop message from rader({self.ip}:{self.port}), response: {response}")
        message = b'\x33\x03\x04\xC5'  # reboot
        self.sock.send(message)
        self.logger.debug(f"Send reboot message to rader({self.ip}:{self.port}), message: {message}")
        response = self.sock.recv(self.buffer_size)
        self.logger.debug(f"Response of reboot message from rader({self.ip}:{self.port}), response: {response}")
        self.sock.close()
        self.logger.info(f"Disconnect to rader({self.ip}:{self.port})")

    def _calculate_checksum(self, data):
        checksum = 0
        for byte in data[:-1]: # 마지막 바이트는 체크섬이므로 제외
            checksum += byte
        checksum = ~checksum & 0xFF # 비트 반전 후 하위 8비트만 취함
        return checksum

    def _parse_packet(self, data, packet_number):
        if len(data) < 8: # 최소 패킷 길이 (SOP, Command, Length, Frame End, Checksum 포함)
            # print("  Invalid packet length")
            return None, None

        sop = data[0]
        command = data[1]
        length = (data[2] << 8) | data[3]
        
        if len(data) < length:
            # print("  Incomplete packet received")
            return None, None
        
        ack = data[4]
        device_id = (data[5] << 8) | data[6]
        track_num = data[7]

        # print(f"Data {packet_number}: {data.hex()}")
        # print(f"  SOP: 0x{sop:02X}")
        # print(f"  Command: 0x{command:02X}")
        # print(f"  Length: {length}")
        # print(f"  ACK: {ack}")
        # print(f"  Device ID: {device_id}")
        # print(f"  Track Num: {track_num}")

        track_info = []
        index = 8  # Payload 시작 인덱스
        for _ in range(track_num):
            track_id = data[index]
            position_x = struct.unpack('<h', data[index + 1:index + 3])[0]  # 리틀 엔디안 부호 있는 16비트 정수
            position_y = struct.unpack('<H', data[index + 3:index + 5])[0]  # 리틀 엔디안 부호 없는 16비트 정수
            track_info.append((track_id, position_x, position_y))
            # print(f"  Track ID: {track_id}")
            # print(f"  position_x: {position_x}")
            # print(f"  position_y: {position_y}")
            index += 5  # 각 트랙 데이터 크기

        vital_info = []
        if len(data) > index:
            vital_num = data[index]
            index += 1
            for _ in range(vital_num):
                if index + 23 <= len(data):
                    vital_id = data[index]
                    br_est = data[index + 1]
                    hr_est = data[index + 10]
                    
                    if 6 <= br_est <= 36 and 30 <= hr_est <= 120:
                        vital_info.append((vital_id, br_est, hr_est))
                        # print(f"  Vital ID: {vital_id}")
                        # print(f"  br_est: {br_est}")
                        # print(f"  hr_est: {hr_est}")
                    
                    index += 23  # 각 Vital 데이터 크기

        # frame_end = data[-2]
        # checksum = data[-1]
        # calculated_checksum = self._calculate_checksum(data[:-1])
        # print(f"  Frame End: 0x{frame_end:02X}")
        # print(f"  Checksum: 0x{checksum:02X} (Calculated: 0x{calculated_checksum:02X})")
        # if checksum != calculated_checksum:
        #     print("  Checksum mismatch!")
        # else:
        #     print("  Checksum valid.")

        return track_info, vital_info

    def receive(self, frame):
        try:
            h, w, _ = frame.shape
            # print(f"Frame dimensions: height={h}, width={w}")  # 프레임 정보 출력
            result = []
            br_est_sums = {}
            hr_est_sums = {}
            valid_vital_counts = {}
            data_counter = 1

            while True:
                header = self.sock.recv(4)
                if not header:
                    # print("No header received, exiting loop")
                    break
                
                if len(header) < 4:
                    # print("Incomplete header received, exiting loop")
                    break

                length = (header[2] << 8) | header[3]
                if length <= 0 or length > 1024:  # 임의의 최대 길이 제한 설정
                    # print(f"Invalid length received: {length}, exiting loop")
                    break

                data = header + self.sock.recv(length - 4)
                if not data:
                    # print("No data received, exiting loop")
                    break

                tracks, vitals = self._parse_packet(data, data_counter)
                if not tracks or not vitals:
                    continue
                
                track_info = {}
                for track_id, pos_x, pos_y in tracks:
                    if -200 <= pos_x <= 200 and 0 <= pos_y <= 200:
                        conv_pos_x = (pos_x + 200) / 400 * w
                        conv_pos_y = pos_y / 10.0
                        track_info[track_id] = {"track_id": track_id, "pos_x": conv_pos_x, "pos_y": conv_pos_y}
                        # print(f"Track Data: ID={track_id}, X={pos_x}, Y={pos_y}")
                        self.logger.info(f"TRACK: {track_info}")

                for vital_id, br_est, hr_est in vitals:
                    if vital_id not in track_info:
                        continue  # 트랙 정보가 없으면 무시

                    if vital_id not in br_est_sums:
                        br_est_sums[vital_id] = 0
                        hr_est_sums[vital_id] = 0
                        valid_vital_counts[vital_id] = 0

                    br_est_sums[vital_id] += br_est
                    hr_est_sums[vital_id] += hr_est
                    valid_vital_counts[vital_id] += 1

                    if valid_vital_counts[vital_id] == 10:
                        br_est_avg = br_est_sums[vital_id] / 10
                        hr_est_avg = hr_est_sums[vital_id] / 10
                        # print(f"Track ID: {vital_id}, X: {track_info[vital_id]['pos_x']}, Y: {track_info[vital_id]['pos_y']}, Average br_est: {br_est_avg}, Average hr_est: {hr_est_avg}")
                        self.logger.info(f"Matching Data: {track_info[vital_id]}, vital_breath: {br_est_avg}, vital_heart: {hr_est_avg}")
                        
                        # 초기화
                        br_est_sums[vital_id] = 0
                        hr_est_sums[vital_id] = 0
                        valid_vital_counts[vital_id] = 0

                        # 기존 데이터 제거 후 추가
                        result = [r for r in result if r['id'] != vital_id]
                        result.append({
                            'id': vital_id,
                            'pos': (track_info[vital_id]['pos_x'], track_info[vital_id]['pos_y']),
                            'breath': br_est_avg,
                            'heart': hr_est_avg,
                        })
                        # print(f"10 frames aggregated data: {result}")

                    # vital_info = {"vital_id": vital_id, "vital_breath": br_est, "vital_heart": hr_est}
                    # print(f"Vital Data: ID={vital_id}, Breath={br_est}, Heart={hr_est}")

                data_counter += 1
                if data_counter == 12:
                    break

                # # 주기적으로 결과 출력
                # if data_counter % 10 == 0:
                #     print(f"Intermediate result after {data_counter} packets: {result}")

            self.logger.info(f"result: {result}")
            return result

        except Exception as e:
            self.logger.debug(f"Error occurred in rader: {e}")
            import traceback
            print(traceback.format_exc())
            return result 

    def recv_all(self, length):
        """ 수신 길이만큼 데이터를 수신 """
        data = b''
        while len(data) < length:
            packet = self.sock.recv(length - len(data))
            if not packet:
                return None
            data += packet
        return data


