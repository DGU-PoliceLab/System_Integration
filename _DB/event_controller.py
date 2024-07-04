import time
from multiprocessing import Process, Pipe, Queue
from _Utils.logger import get_logger

def collect_evnet(pipe):
    logger = get_logger(name= '[EVENT]', console= True, file= True)
    while True:
        event = pipe.recv()
        if event is not None:
            logger.info(f"Event: {event}")
        else:
            time.sleep(0.0001)