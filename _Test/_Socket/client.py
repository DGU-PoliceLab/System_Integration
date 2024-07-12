import socket
import threading
import json
from _Utils.logger import get_logger

SOURCE = "_temp"

class ClientSocket:
    def __init__(self, ip, port):
        self.logger = get_logger(name= '[SOCKETCLIENT]', console= True, file= True)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.data = None
        self.cond = threading.Condition()
        self.receiveThread = threading.Thread(target=self.receive)
        try:
            self.sock.connect((ip, port))
            self.receiveThread.start()
        except Exception as e:
            self.logger.error(f"Error occured in __init__, error: {e}")

    def receive(self):
        while True:
            try:
                byte_data = self.sock.recv(1024)
                self.logger.info(f'Server : {byte_data}')
                if len(byte_data) > 0:
                    str_data = str(byte_data)
                    # str_data = byte_data.decode('utf-8')
                    data = eval(str_data)
                    with self.cond:
                        self.data = data
            except Exception as e:
                self.logger.error(f"Error occured in receive(), error: {e}")
                # self.sock.close()
                break

    def send(self, data):
        try:
            str_data = str(data)
            # str_data = json.dumps(data)
            byte_data = str_data.encode('utf-8')
            self.logger.info(f'send, data: {data}, str: {str_data}, byte: {byte_data}')
            self.sock.sendall(byte_data)
        except Exception as e:
            self.logger.error(f"Error occured in send(), error: {e}")

    def get(self):
        return self.data

def main():
    TCP_IP = 'localhost'
    TCP_PORT = 9000
    client = ClientSocket(TCP_IP, TCP_PORT)
    while True:
        s = input('>>>')
        if s != "":
            client.send(s)
        else:
            break

if __name__ == "__main__":
    main()