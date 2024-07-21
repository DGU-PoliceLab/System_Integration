import time
from multiprocessing import Process, Pipe, Queue
from Utils.logger import get_logger
from threading import Thread
from EventHandler.db_controller import insert_event,connect_db
from EventHandler.mq_controller import connect_mq
from queue import Queue

def collect_evnet(pipe):
    logger = get_logger(name= '[EVENT]', console= True, file= True)
    conn = connect_db()
    mq_conn = connect_mq()
    event_queue = Queue()
    insert_db_thread = Thread(target=insert_event, args=(event_queue, conn, mq_conn))
    insert_db_thread.start()
    while True:
        event = pipe.recv()
        if event is not None:
            event_queue.put(event)
            # logger.info(f"event_queue.size: {event_queue.qsize()}")
        else:
            time.sleep(0.0001)