import sys
import socket
import pickle
from threading import Thread

def run_socket_server(port=20000):
    server = SocketServer()
    server.start(port)

def socket_server_thread(port):
    thread = Thread(target=run_socket_server, args=(port,))
    thread.setDaemon(True)
    thread.start()

class SocketServer:

    def __init__(self):
        self.__socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.__connections = list()
        self.__names = list()
        print("server created")

    def __client_thread(self, client_id):
        connection = self.__connections[client_id]
        client = self.__names[client_id]
        print(f"Client {client_id} connected")
        cnt = 0
        while True:
            try:
                buffer = connection.recv(65536)
                obj = pickle.loads(buffer)
                if obj['type'] == 'broadcast':
                    self.__broadcast(obj['sender_id'], obj['data'])
                cnt += 1

            except Exception as e:
                print(f"Error in client {client_id}: {e}")
                print(self.__connections)
                self.__connections[client_id].close()
                self.__connections[client_id] = None
                self.__names[client_id] = None

    def __broadcast(self, client_id=0, data=None):
        buffer = pickle.dumps(data)
        for i in range(1, len(self.__connections)):
            if client_id != i:
                self.__connections[i].send(buffer)

    def __wait_connect(self, connection):
        try:
            buffer = connection.recv(65536)
            obj = pickle.loads(buffer)
            if obj['type'] == 'connect':
                self.__connections.append(connection)
                self.__names.append(obj['name'])
                connection.send(pickle.dumps({
                    'id': len(self.__connections) - 1
                }))
                thread = Thread(target=self.__client_thread, args=(len(self.__connections) - 1,))
                thread.setDaemon(True)
                thread.start()
            else:
                print("Invalid connection")
        except Exception as e:
            print(f"Error in connection: {e}")          

    def start(self, port = 20000, max_connection = 10):
        self.__socket.bind(('localhost', port))
        self.__socket.listen(max_connection)
        print("Server is starting")
        self.__connections.clear()
        self.__names.clear()
        self.__connections.append(None)
        self.__names.append("Server")

        while True:
            connection, address = self.__socket.accept()
            print(f"New connection from {address}")
            thread = Thread(target=self.__wait_connect, args=(connection,))
            thread.setDaemon(True)
            thread.start()

class SocketProvider:
    def __init__(self):
        self.__socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.__id = None
        self.__name = None
        self.__is_connected = False

    def __connect(self, name):
        self.__socket.send(pickle.dumps({
            'type': 'connect',
            'name': name
        }))
        try:
            buffer = self.__socket.recv(65536)
            obj = pickle.loads(buffer)
            if obj['id']:
                self.__name = name
                self.__id = obj['id']
                self.__is_connected = True
                print(f"Connected to server with id {self.__id}")
            else:
                print("Connection failed")
        except Exception as e:
            print(f"Error in connection: {e}")

    def start(self, target=20000):
        self.__socket.connect(('localhost', target))
        self.__connect("Provider")

    def send(self, data):
        buffer = pickle.dumps({
            'type': 'broadcast',
            'sender_id': self.__id,
            'data': data})
        print(sys.getsizeof(buffer))
        self.__socket.send(buffer)
        
class SocketConsumer:
    def __init__(self):
        self.__socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.__id = None
        self.__name = None
        self.__is_connected = False
        self.__data = None

    def __connect(self, name):
        self.__socket.send(pickle.dumps({
            'type': 'connect',
            'name': name
        }))
        try:
            buffer = self.__socket.recv(65536)
            obj = pickle.loads(buffer)
            if obj['id']:
                self.__name = name
                self.__id = obj['id']
                self.__is_connected = True

                thread = Thread(target=self.__receive_thread)
                thread.setDaemon(True)
                thread.start()

                print(f"Connected to server with id {self.__id}")
            else:
                print("Connection failed")
        except Exception as e:
            print(f"Error in connection: {e}")

    def __receive_thread(self):
        while self.__is_connected:
            try:
                buffer = self.__socket.recv(65536)
                obj = pickle.loads(buffer)
                self.__data = obj
            except Exception as e:
                print(f"Error in connection: {e}")

    def start(self, target=20000):
        self.__socket.connect(('localhost', target))
        self.__connect("Consumer")

    def get(self):
        return self.__data