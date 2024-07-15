import time
from multiprocessing import Process, Pipe, Queue
from Utils.logger import get_logger
from threading import Thread
from queue import Queue
from variable import get_debug_args, get_arg
import pymysql
import os
import pika
import json
import datetime
import copy
import cv2

db_config = {
    "host": "172.30.1.43",
    "port": 3306,
    "user": "root",
    "password": "mhncity@364",
    "database": "mysql-pls",
    "charset": "utf8"
}
mq_config = {
    "host": '172.30.1.43',
    "port": 9090,
    "user": 'mhncity',
    "password": 'mhncity@364',
    "exchange": 'event_exchange'
}

logger = get_logger(name= '[EVENT]', console= True, file= True)

class EventHandler:
    def __init__(self):
        # self.db_conn = self.connect_db(config=db_config)
        pass

    def connect_db(self, config, db_name=None):
        if db_name == None:
            db_name = config["database"]

        conn = pymysql.connect(host=config["host"],
                                port=config["port"],
                                user=config["user"],
                                password=config["password"],
                                database=db_name,
                                charset=config["charset"])
        return conn
    


    def update(self, event_pipe):
        def collect_evnet(pipe):
            debug_args = get_debug_args()

            if debug_args.debug == True:
                while True:
                    event = pipe.recv()
                    if event is not None:
                        logger.info(f"Event: {event}")
                    else:
                        time.sleep(0.0001)
            else:
                db_conn = self.db_conn
                mq_conn = self.mq_conn

                event_queue = Queue()
                insert_db_thread = Thread(target=insert_event, args=(event_queue, db_conn, mq_conn))
                insert_db_thread.start()
                while True:
                    event = pipe.recv()
                    if event is not None:
                        event_queue.put(event)
                        print(f"event_queue.size: {event_queue.qsize()}")
                    else:
                        time.sleep(0.0001)
            pass
        collect_evnet(event_pipe)



class DBUtil:
    def __init__(self):
        self.connect = self.connect_db()
        self.connect_pls_temp = self.connect_db(db_name="mysql-pls") # 스냅샷 쪽 인물 데이터 실시간 갱신 할 때 쓰이는 듯
        pass


        pass


class MQUtil:
    def __init__(self):
        self.connect = self.connect_mq()
        pass

    def connect_mq(self):
        credentials = pika.PlainCredentials(username=CONFIG['user'], password=CONFIG['password'])
        connection_params = pika.ConnectionParameters(host=CONFIG['host'], credentials=credentials)
        connection = pika.BlockingConnection(connection_params)
        channel = connection.channel()

        if channel.is_open:
            return channel
        else:
            return None
        pass

    def publish_message(self, message):  ## TODO! 예외처리 방식의 문제가 있음
        assert self.connect is not None, 'FUNCTION publish_message : channel object does not exist'

        try:
            self.connect.basic_publish(exchange=CONFIG['exchange'], body=json.dumps(message), routing_key='')

        except Exception as e:
            logger.warning(f'FUNCTION publish_message : Error occurred during message publishing, error: {e}')
        pass


class meg_handler:
        
    def __init__(self):

        pass

    

# call by run
def object_snapshot_control(data_pipe):

    NAS_PATH = get_arg('root', 'nas_path')
    MAX_PERSON_NUM = get_arg('root','max_person_num')

    conn = connect_db("mysql-pls")
    mq_conn = connect_mq()
    if conn.open:
        pass
    else:
        logger.info("FUNCTION object_snapshot_control : Database connection failed")
    
    print("스냅샷 프로세스 시작")
    body_cutting_frames = {}
    previous_tid_count = 0  # 이전 루프의 tid 개수를 저장하는 변수
    first_run = True  # 첫 실행 여부를 확인하는 변수 추가
    total_db_insert_datas = []  # 모든 db_insert_datas를 모으는 리스트 추가
    data_pipe.send(True)
    last_update_time = time.time()  # Initialize the last update time

    while True:
        data = data_pipe.recv()
        tracks, meta_data, frame, num_frame = data
        tracking_time = meta_data['current_datetime']
        cctv_id = meta_data['cctv_id']
        cctv_name = meta_data['cctv_name']

        save_path = os.path.join(NAS_PATH, str(cctv_id))
        if save_path != '':
            os.makedirs(save_path, exist_ok=True)

        for i, track in enumerate(tracks):
            tlwh = track.tlwh
            tid = track.track_id
            margin = 30
            x1 = int(tlwh[0])
            y1 = int(tlwh[1])
            x2 = int(tlwh[0] + tlwh[2])
            y2 = int(tlwh[1] + tlwh[3])
            body_cutting_frame = copy.deepcopy(frame[y1:y2, x1:x2]) # 새로운 객체별 몸통 bbox 추론
            body_cutting_frames[tid] = body_cutting_frame
            if body_cutting_frames[tid] is None:
                logger.warning(f"Frame is Empty(body_cutting_frames[{tid}])")
                
            people_thumbnail_location_link = None
            people_thumbnail_location_link = str(tid) + "_" + str(tracking_time)
            file_path = os.path.join(save_path, f"ID_{people_thumbnail_location_link}")
            tracking_time_obj = datetime.strptime(tracking_time, "%Y-%m-%d_%H:%M:%S")
            formatted_tracking_time = tracking_time_obj.strftime("%y%m%d_%H%M%S")
            people_name_material = formatted_tracking_time
            people_name = people_name_material

            db_insert_file_path = os.path.join(str(cctv_id), f"ID_{people_thumbnail_location_link}.jpg")
            try:
                db_insert_datas = [cctv_id, tid, people_name, db_insert_file_path]
                total_db_insert_datas.append(db_insert_datas) 
            except Exception as e:
                logger.warning(f"Error occurred while sending to the database, error: {e}")

            try:
                if body_cutting_frames[tid].size > 0:
                    cv2.imwrite(f"{file_path}.jpg", body_cutting_frames[tid])
                    logger.info(f"Save complete(OBJECT #{tid}).")
                else:
                    logger.warning(f"Snapshot is empty(OBJECT #{tid}).")
                    x1 = max(0, int(tlwh[0]) - margin)
                    y1 = max(0, int(tlwh[1]) - margin)
                    x2 = min(frame.shape[1], int(tlwh[0] + tlwh[2]) + margin)
                    y2 = min(frame.shape[0], int(tlwh[1] + tlwh[3]) + margin)
                    body_cutting_frame = copy.deepcopy(frame[y1:y2, x1:x2])
                    body_cutting_frames[tid] = body_cutting_frame
                    cv2.imwrite(f"{file_path}.jpg", body_cutting_frames[tid])
                    logger.info(f"Save Edited snapshot(OBJECT #{tid}).")
            except Exception as e:
                logger.warning(f"Error occurred while saving OBJECT #{tid} from CCTV #{cctv_id}, error: {e}")  
   
        if total_db_insert_datas:
            current_time = time.time()
            if  current_time - last_update_time >= 10:  # Check if 10 seconds have passed
                insert_snapshot(total_db_insert_datas, conn, mq_conn)
                last_update_time = current_time  # Update the last update time
            total_db_insert_datas.clear()
    pass


def insert_realtime(queue, conn):
    import queue as QueueModule
    QUEUE_EMPTY_INTERVAL = 1

    ### 쓰는 곳 ###
    # insert_realtime_thread = Thread(target=insert_realtime, args=(realtime_status_queue, realtime_status_conn),daemon=False).start()

    # assert conn is None, 'conn object does not exist'
    assert conn is not None, 'FUNCTION insert_realtime : conn object does not exist'
    
    try:
        print("insert_realtime 실행")
        while True:
            try:
                data = queue.get_nowait()
            except QueueModule.Empty:  
                time.sleep(QUEUE_EMPTY_INTERVAL)
                continue

            cctv_id, people_id, heartbeat_rate, breath_rate, body_temperature, emotion, radar_datetime = data
            cur = conn.cursor()
            sql_bring_people_id = 'SELECT people_id FROM people WHERE cctv_id = %s AND people_color_num = %s'

            cur.execute(sql_bring_people_id, (cctv_id, people_id))

            people_table_id = cur.fetchone()
            if people_table_id:
                people_table_id = people_table_id[0]
                print(f"people_table_id : {people_table_id}")

                ## 열화상 값 더미 였던 것을 변경
                sql = "INSERT INTO realtime_status (cctv_id, people_id, realtime_heartbeat_rate, realtime_breathe_rate, realtime_body_temperature, realtime_emotion, realtime_datetime) VALUES (%s, %s, %s, %s, %s, %s, %s)"
                # TODO : Get actual body temperature
                # body_temperature = round(random.uniform(36.4, 37.3), 1)

                body_temperature = round(body_temperature, 2)
                cur.execute(sql, (cctv_id, people_table_id, heartbeat_rate, breath_rate, body_temperature, emotion, radar_datetime))
                conn.commit()
                logger.info("Real-time data insertion complete.")
            else:
                logger.warning("'people_id' that satisfies the condition does not exist.")

    except Exception as e:
        logger.warning(f"Error occurred during real-time data insertion, error: {e}")
        conn.rollback()
    pass


def insert_snapshot(data_list, conn, mq_conn):
    assert conn is not None, 'FUNCTION insert_snapshot : conn object does not exist'
    assert mq_conn is not None, 'FUNCTION insert_snapshot : mq_conn object does not exist'
    try:
        print("insert_snapshot 실행")
        with conn.cursor() as cur:
            # Delete existing data
            delete_sql = "DELETE FROM people WHERE cctv_id = %s"
            cur.execute(delete_sql, (data_list[0][0],))
            for data in data_list:


                sql = """
                INSERT INTO people  
                (cctv_id, people_name, people_color_num, people_thumbnail_location) 
                VALUES (%s,%s,%s,%s)
                """

                values = [data[0], data[2], data[1], data[3]]
                cur.execute(sql, values)
        conn.commit()
        try:                
            message = {"alert_type": "people"}            
            publish_message(message, mq_conn)
        except Exception as e:
            logger.warning(f"Error occurred on message queue, error: {e}")
    except Exception as e:
        logger.warning(f"Error occurred during insert data into database, error: {e}")
    pass


def insert_snapshot(data_list, conn, mq_conn):
    assert conn is not None, 'FUNCTION insert_snapshot : conn object does not exist'
    assert mq_conn is not None, 'FUNCTION insert_snapshot : mq_conn object does not exist'
    try:
        print("insert_snapshot 실행")
        with conn.cursor() as cur:
            # Delete existing data
            delete_sql = "DELETE FROM people WHERE cctv_id = %s"
            cur.execute(delete_sql, (data_list[0][0],))
            for data in data_list:

                sql = """
                INSERT INTO people  
                (cctv_id, people_name, people_color_num, people_thumbnail_location) 
                VALUES (%s,%s,%s,%s)
                """
                # DB에 삽입할 값 설정
                values = [data[0], data[2], data[1], data[3]]
                cur.execute(sql, values)
        conn.commit()
        try:                
            message = {"alert_type": "people"}            
            publish_message(message, mq_conn)
        except Exception as e:
            logger.warning(f"Error occurred on message queue, error: {e}")
    except Exception as e:
        logger.warning(f"Error occurred during insert data into database, error: {e}")
    pass


def delete_snapshot(tid,conn, mq_conn):
    assert conn is not None, 'FUNCTION delete_snapshot : conn object does not exist'
    assert mq_conn is not None, 'FUNCTION delete_snapshot : mq_conn object does not exist'
    try:
        with conn.cursor() as cursor:
            sql = "DELETE FROM people WHERE people_color_num = %s"
            cursor.execute(sql, (tid,))
        conn.commit()

        logger.info(f"Deleted object with 객체 ID: {tid} from the database.")
        logger.info(f"Object #{tid} detele complete.")

        try:                
            message = {"alert_type": "people"}      # 스냅샷 새로고침        
            publish_message(message, mq_conn)
        except Exception as e:
            logger.warning(f"FUNCTION delete_snapshot : Error occurred on message queue, error: {e}")

    except Exception as e:
        logger.warning(f"FUNCTION delete_snapshot : Error occurred during delete(OBJECT #tid), error: {e}")
    pass



def insert_event(event_queue, conn, mq_conn):
    assert conn is not None, 'conn object does not exist'
    
    # event insert delay code start
    LAST_EVENT_TIME = {"falldown": None, "selfharm": None, "violence": None, "longterm_status": None}
    DELAY_TIME = get_arg('root', 'event_delay')

    def str_to_second(time_str):
        tl = list(map(int, time_str.split(":")))
        tn = tl[0] * 3600 + tl[1] * 60 + tl[2]
        return tn

    def check_delay(event, cur_time):
        last_time = LAST_EVENT_TIME[event]
        if last_time == None:
            update(event, cur_time)
            return False
        else:
            diff = cur_time - last_time
            if diff > DELAY_TIME:
                return True
            else:
                return False

    def update(event, cur_time):
        LAST_EVENT_TIME[event] = cur_time
    # event insert delay code end
        
    while True:
        event = event_queue.get()
        if event is not None:
            try:             
                ##############################임시. 원래 모델에서 제공되야될 정보      
                cctv_id = event['cctv_id']
                event_type = event['action']
                track_id = event['id']
                event_location = event['location']
                current_datetime = event['current_datetime']
                event_date = str(current_datetime)[:10]
                event_time = str(current_datetime)[11:19]
                event_start_datetime = str(current_datetime)[:19]

                event_end_datetime = event_start_datetime                
                event_start = event_start_datetime
                event_end = event_end_datetime

                event_clip_directory = "As soon as"
                ##############################

                logger.info(f"[EVENT DETECT] - cctv_id : {cctv_id}, event_type : {event_type}, event_location : {event_location}, track_id : {track_id}, event_date : {event_date}, event_time : {event_time}, event_clip_directory : {event_clip_directory}, event_start : {event_start}, event_end : {event_end}")
                event_cur_time = str_to_second(event_time) 

                if check_delay(event_type, event_cur_time):
                    try:
                        cur = conn.cursor()

                        sql_bring_people_id = "SELECT people_color_num FROM people WHERE people_color_num = %s AND cctv_id = %s"

                        cur.execute(sql_bring_people_id, (track_id, cctv_id))
                        people_table_id = cur.fetchone()
                        if people_table_id:
                            people_table_id = people_table_id[0]
                        else:
                            logger.warn("'[insert_realtime] : people_id' that satisfies the condition does not exist.")
                            
                        DB_insert_sql = """
                        INSERT INTO event 
                        (cctv_id, event_type, event_location, event_detection_people, 
                        event_date, event_time, event_clip_directory, 
                        event_start, event_end) 
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """
                        values = [cctv_id, event_type, event_location, people_table_id, event_date, event_time, event_clip_directory, event_start, event_end]
                        cur.execute(DB_insert_sql, values)
                        logger.info("Event insertion complete.")

                        try:
                            event_id = cur.lastrowid
                            try:
                                message = {"alert_type": "event", "event_id": event_id}  # 이벤트 알람. 이벤트 새로고침.
                                publish_message(message, mq_conn)
                            except Exception as e:
                                logger.warning(f"Can not find event_id, {e}")
                                pass
                        except Exception as e:
                            logger.warning(f"Error occurred in message Queue, error, {e}")
                    except Exception as e:
                        logger.warning(f"Error occurred during event data insertion, error: {e}")
                    finally:
                        conn.commit()
                        update(event_type, event_cur_time) # event insert delay code use here
            except Exception as e:
                logger.warning(f"Error occurred in insert_event loop, error: {e}")
    pass
                
