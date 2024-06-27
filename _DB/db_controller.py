import pymysql
import random
import queue as QueueModule
import time
from _DB.config import db_config as CONFIG
from _DB.mq_controller import publish_message
from _Utils.logger import get_logger
from variable import get_arg

QUEUE_EMPTY_INTERVAL = 1
HEART_RATE_THRESHOLD = 100
BREATH_RATE_THRESHOLD = 100
EMOTION_THRESHOLD = 2
LOGGER = get_logger(name = '[DB]')

def connect_db(db = None):
    if db != None:
        conn = pymysql.connect(host=CONFIG["host"],
                            port=CONFIG["port"],
                            user=CONFIG["user"],
                            password=CONFIG["password"],
                            database=db,
                            charset=CONFIG["charset"])
        return conn
    else:
        conn = pymysql.connect(host=CONFIG["host"],
                            port=CONFIG["port"],
                            user=CONFIG["user"],
                            password=CONFIG["password"],
                            database=CONFIG["database"],
                            charset=CONFIG["charset"])
        return conn



def insert_realtime(queue, conn):
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
            cctv_id, people_id, heartbeat_rate, breath_rate, emotion, radar_datetime = data
            cur = conn.cursor()
            sql_bring_people_id = 'SELECT people_id FROM people WHERE cctv_id = %s AND people_color_num = %s'
            cur.execute(sql_bring_people_id, (cctv_id, people_id))
            people_table_id = cur.fetchone()
            if people_table_id:
                people_table_id = people_table_id[0]
                print(f"people_table_id : {people_table_id}")
                sql = "INSERT INTO realtime_status (cctv_id, people_id, realtime_heartbeat_rate, realtime_breathe_rate, realtime_body_temperature, realtime_emotion, realtime_datetime) VALUES (%s, %s, %s, %s, %s, %s, %s)"
                # TODO : Get actual body temperature
                body_temperature = round(random.uniform(36.4, 37.3), 1)
                cur.execute(sql, (cctv_id, people_table_id, heartbeat_rate, breath_rate, body_temperature, emotion, radar_datetime))
                conn.commit()
                LOGGER.info("Real-time data insertion complete.")
            else:
                LOGGER.warning("'people_id' that satisfies the condition does not exist.")
    except Exception as e:
        LOGGER.warning(f"Error occurred during real-time data insertion, error: {e}")
        conn.rollback()

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
            LOGGER.warning(f"Error occurred on message queue, error: {e}")
    except Exception as e:
        LOGGER.warning(f"Error occurred during insert data into database, error: {e}")

def update_snapshot(data_list, conn, mq_conn):
    assert conn is not None, 'FUNCTION update_snapshot : conn object does not exist'
    assert mq_conn is not None, 'FUNCTION update_snapshot : mq_conn object does not exist'
    try:
        # print("update_snapshot 실행")
        with conn.cursor() as cur:
            # 외래 키 제약 조건 비활성화
            # cur.execute("SET FOREIGN_KEY_CHECKS = 0")
            # Delete existing data
            delete_sql = "DELETE FROM people WHERE cctv_id = %s"
            cur.execute(delete_sql, (data_list[0][0],))
            
            for data in data_list:
                
                # Insert new data if data[1] is not None or null
                # cctv_id, tid, people_name, db_insert_file_path 순서
                if data[1] is not None:
                    insert_sql = """
                    INSERT INTO people  
                    (cctv_id, people_name, people_color_num, people_thumbnail_location) 
                    VALUES (%s,%s,%s,%s)
                    """
                    values = [data[0], data[2], data[1], data[3]]
                    cur.execute(insert_sql, values)

            # 외래 키 제약 조건 다시 활성화
            # cur.execute("SET FOREIGN_KEY_CHECKS = 1")
        conn.commit()
        
        try:                
            message = {"alert_type": "people"}            
            publish_message(message, mq_conn)
        except Exception as e:
            LOGGER.warning(f"FUNCTION update_snapshot : Error occurred on message queue, error: {e}")
    except Exception as e:
        LOGGER.warning(f"FUNCTION update_snapshot : Error occurred during insert data into database, error: {e}")

def delete_snapshot(tid,conn, mq_conn):
    assert conn is not None, 'FUNCTION delete_snapshot : conn object does not exist'
    assert mq_conn is not None, 'FUNCTION delete_snapshot : mq_conn object does not exist'
    try:
        with conn.cursor() as cursor:
            sql = "DELETE FROM people WHERE people_color_num = %s"
            cursor.execute(sql, (tid,))
        conn.commit()
        print(f"Deleted object with 객체 ID: {tid} from the database.")
        LOGGER.info(f"Object #{tid} detele complete.")
        try:                
            message = {"alert_type": "people"}            
            publish_message(message, mq_conn)
        except Exception as e:
            LOGGER.warning(f"FUNCTION delete_snapshot : Error occurred on message queue, error: {e}")
    except Exception as e:
        LOGGER.warning(f"FUNCTION delete_snapshot : Error occurred during delete(OBJECT #tid), error: {e}")

def insert_event(queue, conn, mq_conn):
    assert conn is not None, 'conn object does not exist'
    """
    이상행동 타입(심박수: heartrate, 체온: temperature, 감정: emotion, 장시간고정자세: longterm_status, 폭행: violence, 쓰러짐: falldown, 자해: selfharm)	
    event_detection_people : people table -> people_id
    """
    
    # event insert delay code start
    LAST_EVENT_TIME = {"falldown": None, "selfharm": None}
    DELAY_TIME = get_arg('root', 'event_delay')

    def str_to_second(time_str):
        tl = list(map(int, time_str.split(":")))
        tn = tl[0] * 3600 + tl[1] * 60 + tl[2]
        return tn

    def check(event, cur_time):
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
        try:
            if queue.empty():
                time.sleep(QUEUE_EMPTY_INTERVAL)
                continue

            # cctv_id, event_type, event_location, track_id, event_date, event_time, event_clip_directory, event_start, event_end = queue.get()
            cctv_id, event_type, event_location, track_id, event_date, event_time, event_clip_directory, event_start, event_end = queue.get()
            LOGGER.info(f"[EVENT DETECT] - cctv_id : {cctv_id}, event_type : {event_type}, event_location : {event_location}, track_id : {track_id}, event_date : {event_date}, event_time : {event_time}, event_clip_directory : {event_clip_directory}, event_start : {event_start}, event_end : {event_end}")
            event_cur_time = str_to_second(event_time) # event insert delay code use here
            if check(event_type, event_cur_time): # event insert delay code use here
                try:
                    cur = conn.cursor()
                    sql_bring_people_id = """
                    SELECT people_id 
                    FROM people 
                    WHERE people_color_num = %s AND cctv_id = %s
                    """
                    print(f"track_id : {track_id}") 
                    print(f"cctv_id : {cctv_id}")
                    cur.execute(sql_bring_people_id, (track_id, cctv_id))
                    people_table_id = cur.fetchone()
                    if people_table_id:
                        people_table_id = people_table_id[0]
                    else:
                        LOGGER.warn("'[insert_realtime] : people_id' that satisfies the condition does not exist.")
                        
                    DB_insert_sql = """
                    INSERT INTO event 
                    (cctv_id, event_type, event_location, event_detection_people, 
                    event_date, event_time, event_clip_directory, 
                    event_start, event_end) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    values = [cctv_id, event_type, event_location, people_table_id, event_date, event_time, event_clip_directory, event_start, event_end]
                    cur.execute(DB_insert_sql, values)
                    LOGGER.info("Event insertion complete.")

                    try:
                        event_id = cur.lastrowid
                        message = {"alert_type": "event", "event_id": event_id}
                        publish_message(message, mq_conn)
                    except Exception as e:
                        LOGGER.warning(f"Error occurred in message Queue, error, {e}")
                except Exception as e:
                    LOGGER.warning(f"Error occurred during event data insertion, error: {e}")
                finally:
                    conn.commit()
                    update(event_type, event_cur_time) # event insert delay code use here
        except Exception as e:
            LOGGER.warning(f"Error occurred in insert_event loop, error: {e}")

# def insert_event(queue, conn, mq_conn):
#     print(f"[insert_event] : queue id : {id(queue)}")
#     """
#     이상행동 타입(심박수: heartrate, 체온: temperature, 감정: emotion, 장시간고정자세: longterm_status, 폭행: violence, 쓰러짐: falldown, 자해: selfharm)	
#     event_detection_people : people table -> people_id
#     """
#     assert conn is not None, 'conn object does not exist'
    
#     while not queue.empty():
#         (cctv_id, event_type, event_location, track_id, event_date, event_time, event_clip_directory, event_start, event_end) = queue.get()
#         print(f"insert_event 함수 실행")
#         try:
#             cur = conn.cursor()
#             sql = """
#             SELECT people_id 
#             FROM people 
#             WHERE people_color_num = %s AND cctv_id = %s
#             """
#             cur.execute(sql_bring_people_id, (track_id, cctv_id))
#             people_table_id = cur.fetchone()
#             if people_table_id:
#                 people_table_id = people_table_id[0]
#             else:
#                 LOGGER.warn("'people_id' that satisfies the condition does not exist.")
                
#             sql = """
#             INSERT INTO event 
#             (cctv_id, event_type, event_location, event_detection_people, 
#             event_date, event_time, event_clip_directory, 
#             event_start, event_end) 
#             VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
#             """
#             values = [cctv_id, event_type, event_location, people_table_id, event_date, event_time, event_clip_directory, event_start, event_end]
#             cur.execute(DB_insert_sql, values)
#             LOGGER.info("Event insertion complete.")

#             try:
#                 event_id = cur.lastrowid
#                 message = {"alert_type": "event", "event_id": event_id}
#                 publish_message(message, mq_conn)
#             except Exception as e:
#                 LOGGER.warning(f"Error occurred in message Queue, error, {e}")
#         except Exception as e:
#             LOGGER.warning(f"Error occurred during event data insertion, error: {e}")
#         finally:
#             conn.commit()

def assign_conn_process(conn):
    assert conn is None, 'conn object does not exist'
    realtime_db_process = Process(target=exectue_sql(conn))
    realtime_db_process.start()

def exectue_sql(conn):
    assert conn is None, 'conn object does not exist'
    cur = conn.cursor()
    cur.execute("SELECT * FROM realtime_status")
    rows = cur.fetchall()
    for row in rows:
        LOGGER.info(row)