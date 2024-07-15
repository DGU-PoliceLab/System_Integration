import time
from multiprocessing import Process, Pipe, Queue
from _Utils.logger import get_logger
##############################################################################
from threading import Thread
from _DB.db_controller import insert_event,connect_db
from _DB.mq_controller import connect_mq
from queue import Queue
##############################################################################

def collect_evnet(pipe):
    logger = get_logger(name= '[EVENT]', console= True, file= True)
    ##############################################################################
    conn = connect_db()
    mq_conn = connect_mq()
    event_queue = Queue()
    insert_db_thread = Thread(target=insert_event, args=(event_queue, conn, mq_conn))
    insert_db_thread.start()
    while True:
        event = pipe.recv()
        if event is not None:
            event_queue.put(event)
            # print(event)
            # print(f"event_queue.size: {event_queue.qsize()}")
            # logger.info(f"Event: {event}")
    ############################################################################## 
        else:
            time.sleep(0.0001)