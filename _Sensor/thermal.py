import socket
import threading
import base64
import numpy

OUTPUT = None

class ServerSocket:

    def __init__(self, ip, port):
        self.TCP_IP = ip
        self.TCP_PORT = port
        self.socketOpen()
        self.receiveThread = threading.Thread(target=self.receiveImages)
        self.receiveThread.start()

    def socketClose(self):
        self.sock.close()
        print(u'Server socket [ TCP_IP: ' + self.TCP_IP + ', TCP_PORT: ' + str(self.TCP_PORT) + ' ] is close')

    def socketOpen(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((self.TCP_IP, self.TCP_PORT))
        self.sock.listen(1)
        print(u'Server socket [ TCP_IP: ' + self.TCP_IP + ', TCP_PORT: ' + str(self.TCP_PORT) + ' ] is open')
        self.conn, self.addr = self.sock.accept()
        print(u'Server socket [ TCP_IP: ' + self.TCP_IP + ', TCP_PORT: ' + str(self.TCP_PORT) + ' ] is connected with client')

    def receiveImages(self):
        global OUTPUT
        try:
            while True:
                stringData = self.recvall(self.conn, 1228800)
                data = numpy.frombuffer(base64.b64decode(stringData), numpy.uint8)
                frame = data.reshape(480, 640, 3) # 세로 해상도, 가로 해상도, RGB값
                temp = self.get_temp(frame)
                OUTPUT.put(frame)
                print(frame)
        except Exception as e:
            print(e)
            self.receiveThread = threading.Thread(target=self.receiveImages)
            self.receiveThread.start()

    def get_temp(frame):
        _temp = 0
        _max = 0
        _max_pixel = []
        for row in frame:
            for pixel in row:
                _cur = pixel[0] + pixel[1] + pixel[2]
                if _cur > _max:
                    _max = _cur
                    _max_pixel = pixel
        print(_max_pixel)

    def recvall(self, sock, count):
        buf = b''
        while count:
            newbuf = sock.recv(count)
            if not newbuf: return None
            buf += newbuf
            count -= len(newbuf)
        return buf

def thermal_start(queue):
    global OUTPUT
    OUTPUT = queue
    server = ServerSocket('0.0.0.0', 5100)

if __name__ == "__main__":
    thermal_start()