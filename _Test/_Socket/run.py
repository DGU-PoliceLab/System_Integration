from multiprocessing import Process
from _Socket.server import ServerSocket

def input_socket():
    input_socket = ServerSocket('localhost', 9001)

def output_socket():
    output_socket = ServerSocket('localhost', 9002)

def socket_process():
    input_process = Process(target=input_socket).start()
    # output_process = Process(target=output_socket).start(); output_process.join()

if __name__ == '__main__':
    socket_process()