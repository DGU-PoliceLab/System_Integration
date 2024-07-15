import socket
import pickle
import time
from Utils.logger import get_logger
from Utils._time import process_time_check
import sys

INPUT_PORTS = [20001, 20002, 20003, 20004, 20005]
OUTPUT_PORTS = 19999

def get_sock(port=None):
    """
    gen socket object
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    if port is not None:
        sock.bind(('localhost', port))
    return sock

def socket_distributor(target=INPUT_PORTS):
    """
    Client(RUN) -> Server(20000)(HAR) -> Servers(20001, 20002, 20003, 20004, 20005)
    """
    logger = get_logger(name= '[DISTRIBUTOR]', console= True, file= True)
    distributor = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    distributor.bind(('localhost', 20000))
    while True:
        serialized_data, addr = distributor.recvfrom(65536)
        if serialized_data:
            logger.info(f"Data send to servers")
            for port in target:
                logger.debug(f"Send to Port {port}")
                sock.sendto(serialized_data, ('localhost', port))
        else:
            time.sleep(0.0001)

def socket_collector():
    """
    Clients(HAR) -> Server(19999) -> DB or MQ
    """
    logger = get_logger(name= '[COLLECTOR]', console= True, file= True)
    collector = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    collector.bind(('localhost', 19999))
    while True:
        serialized_data, addr = collector.recvfrom(65536)
        if serialized_data:
            data = pickle.loads(serialized_data)
            logger.info(data)
            # 여기에 이벤트 발생시 처리하는 함수 추가
        else:
            time.sleep(0.0001)

@process_time_check
def socket_provider(sock, port, data):
    """
    Data -> Serialization -> Send -> Receive
    (           CLIENT           )  ( SERVER )
    """
    serialized_data = pickle.dumps(data)
    sock.sendto(serialized_data, ('localhost', port))

@process_time_check
def socket_consumer(sock):
    """
    Serialized Data -> Send -> Receive -> Deserialization
    (         SERVER       )  (          CLIENT          )
    """
    serialized_data, addr = sock.recvfrom(65536)
    print(sys.getsizeof(serialized_data))
    data = pickle.loads(serialized_data)
    return data