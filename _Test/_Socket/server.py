import socket
import threading
from multiprocessing import Process
from _Utils.logger import get_logger

class ServerSocket:

    def __init__(self, ip, port):
        self.logger = get_logger(name= '[SOCKETSERVER]', console= True, file= True)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((ip, port))
        self.clients = []
        self.data = None
        self.sock.listen()
        self.receive()

    def broadcast(self, data):
        for client in self.clients:
            client.send(data)

    def handle(self, client):
        while True:
            try:
                data = client.recv(65536)
                self.logger.debug(data.encoded('utf-8'))
                self.broadcast(data)
            except:
                self.logger.debug(f"Disconnected {client}")
                self.clients.remove(client)
                client.close()
                break

    def receive(self):
        while True:
            client, address = self.sock.accept()
            self.logger.debug(f"Connected with {address} #{client}")
            self.clients.append(client)
            thread = threading.Thread(target=self.handle, args=(client,))
            thread.start()

def main():
    server = ServerSocket('localhost', 9001)

if __name__ == "__main__":
    main()