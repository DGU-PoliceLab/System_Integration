from sub import sub_proces1, sub_proces2
from multiprocessing import Process, Pipe
import time

pipe_input_1, pipe_output_1 = Pipe()
pipe_input_2, pipe_output_2 = Pipe()

process1 = Process(target=sub_proces1, args=(pipe_output_1,))
process2 = Process(target=sub_proces2, args=(pipe_output_2,))

process1.start()
process2.start()

while True:
    print("wating for sub process 1 to start...")
    if pipe_input_1.recv():
        print("sub process 1 started")
        break
    else:
        time.sleep(0.1)

while True:
    print("wating for sub process 2 to start...")
    if pipe_input_2.recv():
        print("sub process 2 started")
        break
    else:
        time.sleep(0.1)

while True:
    pipe_input_1.send("Hello from main")
    pipe_input_2.send("Hello from main")
    time.sleep(1)