import time
from multiprocessing import Process, Pipe, Queue
from _Utils.logger import get_logger

LOGGER = get_logger(name= '[EVENT]', console= True, file= True)

def collect_evnet(pipe):
    while True:
        event = pipe.recv()
        if event is not None:
            LOGGER.info(f"Event: {event}")
        else:
            time.sleep(0.0001)