def sub_process1(pipe):
    p = pipe.recv()
    print(f"sub process1 recv data: {p}")

def sub_process2(pipe):
    while True:
        p = pipe.recv()
        print(f"sub process2 recv data: {p}")