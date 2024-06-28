from sub import sub_process1, sub_process2
from multiprocessing import Process, Pipe
import time 

main_input_pipe, sub_input_pipe = Pipe()

cnt = 0

sub_process1 = Process(target=sub_process1, args=(sub_input_pipe,))
sub_process1.start()

while True:
    main_input_pipe.send(cnt)
    print(f"main process send data: {cnt}")
    cnt += 1
    time.sleep(0.1)
