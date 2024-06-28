import time

def sub_proces1(pipe):
    time.sleep(10)
    pipe.send(True)
    while True:
        print("1", pipe.recv())

def sub_proces2(pipe):
    time.sleep(1)
    pipe.send(True)
    while True:
        print("2", pipe.recv())