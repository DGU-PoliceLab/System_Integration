import time
from multiprocessing import Process, Pipe, Queue
from Utils.logger import get_logger
from threading import Thread
from queue import Queue
from variable import get_debug_args, get_arg
import pymysql
import config
import os
import pika
import json
import datetime
import copy
import cv2
import config

class EventHandler:
    def __init__(self, is_debug=False):
        self.mq_logger = get_logger(name= '[MQ]', console= True, file= True)
        self.db_logger = get_logger(name = '[DB]', console=False, file=False)
        self.event_logger = get_logger(name= '[EVENT]', console= True, file= True)

        if is_debug:
            self.db_conn = self.connect_db()
        else:
            self.db_conn = self.connect_db(config=config.db_config)

        
    def connect_db(self, config=None):
        if config != None:
            conn = pymysql.connect(host=config["host"],
                                port=config["port"],
                                user=config["user"],
                                password=config["password"],
                                database=config["database"],
                                charset=config["charset"])
            return conn
        else:
            return None
    
    def connect_mq(self):
        credentials = pika.PlainCredentials(username=config.mq_config['user'], password=config.mq_config['password'])
        connection_params = pika.ConnectionParameters(host=config.mq_config['host'], credentials=credentials)
        connection = pika.BlockingConnection(connection_params)
        channel = connection.channel()
        if channel.is_open:
            print("MQ 연결")
        else:
            self.mq_logger.error("Failed to open MQ channel.")
        return channel

    def update(self, pipe, is_debug):
        
        if is_debug == False:
            import config
            conn = self.connect_db(config.db_config)
            mq_conn = self.connect_mq(config.mq_config)

            event_queue = Queue()
            insert_db_thread = Thread(target=self.insert_event, args=(event_queue, conn, mq_conn))
            insert_db_thread.start()
            while True:
                event = pipe.recv()
                if event is not None:
                    event_queue.put(event)
                    # logger.info(f"event_queue.size: {event_queue.qsize()}")
                else:
                    time.sleep(0.0001)
        else:
            event_queue = Queue()
            while True:
                event = pipe.recv()
                if event is not None:
                    event_queue.put(event)
                    self.event_logger.info(f"event_queue.size: {event_queue.qsize()}")
                else:
                    time.sleep(0.0001)

    #region MQ
    def publish_message_mq(self, message, channel):
        assert channel is not None, 'FUNCTION publish_message : channel object does not exist'
        self.mq_logger.info("publish_message 실행")
        try:
            channel.basic_publish(exchange=config.mq_config['exchange'], body=json.dumps(message), routing_key='')

        except Exception as e:
            self.mq_logger.warning(f'FUNCTION publish_message : Error occurred during message publishing, error: {e}')

    def update(self, event_pipe):
        pass
    #endregion MQ

    #region DB query
    def insert_realtime(self, body_data_list, conn):
        # assert conn is None, 'conn object does not exist'
        assert conn is not None, 'FUNCTION insert_realtime : conn object does not exist'
        try:
            cur = conn.cursor()
            for data in body_data_list:            
                cctv_id, people_id, heartbeat_rate, breath_rate, body_temperature, emotion, radar_datetime = data
                cur = conn.cursor()
                sql_bring_people_id = 'SELECT people_id FROM people WHERE cctv_id = %s AND people_color_num = %s'
                cur.execute(sql_bring_people_id, (cctv_id, people_id))
                people_table_id = cur.fetchone()
                if people_table_id:
                    people_table_id = people_table_id[0]
                    print(f"people_table_id : {people_table_id}")
                    sql = "INSERT INTO realtime_status (cctv_id, people_id, realtime_heartbeat_rate, realtime_breathe_rate, realtime_body_temperature, realtime_emotion, realtime_datetime) VALUES (%s, %s, %s, %s, %s, %s, %s)"
                    cur.execute(sql, (cctv_id, people_table_id, heartbeat_rate, breath_rate, body_temperature, emotion, radar_datetime))               
                    
                    self.db_logger.info("Real-time data insertion complete.")
                else:
                    self.db_logger.warning("'people_id' that satisfies the condition does not exist.")
            conn.commit()
                
            self.db_logger.info("Real-time data insertion complete.")

        except Exception as e:
            self.db_logger.warning(f"Error occurred during real-time data insertion, error: {e}")
            conn.rollback()

    def insert_snapshot(self, data_list, conn, mq_conn):
        assert conn is not None, 'FUNCTION insert_snapshot : conn object does not exist'
        assert mq_conn is not None, 'FUNCTION insert_snapshot : mq_conn object does not exist'     
        self.db_logger.info(f"insert_snapshot")
        self.db_logger.info(f"data: {data_list}")
        
        try:
            with conn.cursor() as cur:
                # Delete existing data
                if len(data_list) > 0:
                    delete_sql = "DELETE FROM people WHERE cctv_id = %s"
                    cur.execute(delete_sql, (data_list[0][0],))
                for data in data_list:
                    sql = """
                    INSERT INTO people  
                    (people_id, cctv_id, people_name, people_color_num, people_thumbnail_location) 
                    VALUES (%s, %s,%s,%s,%s)
                    """
                    # print(f"")
                    # DB에 삽입할 값 설정
                    values = [data[1], data[0], data[2], data[1], data[3]]
                    cur.execute(sql, values)
                conn.commit()           
            
            try:                
                message = {"alert_type": "people"}            
                self.publish_message_mq(message, mq_conn)
            except Exception as e:
                self.db_logger.warning(f"Error occurred on message queue, error: {e}")
        except Exception as e:
            self.db_logger.warning(f"Error occurred during insert data into database, error: {e}")



    def update_snapshot(self, data_list, conn, mq_conn):
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
                self.publish_message_mq(message, mq_conn)
            except Exception as e:
                self.db_logger.warning(f"FUNCTION update_snapshot : Error occurred on message queue, error: {e}")
        except Exception as e:
            self.db_logger.warning(f"FUNCTION update_snapshot : Error occurred during insert data into database, error: {e}")

    def delete_snapshot(self, tid,conn, mq_conn):
        assert conn is not None, 'FUNCTION delete_snapshot : conn object does not exist'
        assert mq_conn is not None, 'FUNCTION delete_snapshot : mq_conn object does not exist'
        try:
            with conn.cursor() as cursor:
                sql = "DELETE FROM people WHERE people_color_num = %s"
                cursor.execute(sql, (tid,))
            conn.commit()
            print(f"Deleted object with 객체 ID: {tid} from the database.")
            self.db_logger.info(f"Object #{tid} detele complete.")
            try:                
                message = {"alert_type": "people"}            
                self.publish_message_mq(message, mq_conn)
            except Exception as e:
                self.db_logger.warning(f"FUNCTION delete_snapshot : Error occurred on message queue, error: {e}")
        except Exception as e:
            self.db_logger.warning(f"FUNCTION delete_snapshot : Error occurred during delete(OBJECT #tid), error: {e}")

    def insert_event(self, event_queue, conn, mq_conn):
        assert conn is not None, 'conn object does not exist'
        assert mq_conn is not None, 'mq_conn object does not exist'
        """
        이상행동 타입(심박수: heartrate, 체온: temperature, 감정: emotion, 장시간고정자세: longterm_status, 폭행: violence, 쓰러짐: falldown, 자해: selfharm)	
        event_detection_people : people table -> people_id
        """
        
        # event insert delay code start
        LAST_EVENT_TIME = {"falldown": None, "selfharm": None, "longterm_status": None, "violence": None}
        DELAY_TIME = get_arg('root', 'event_delay')

        def str_to_second(time_str):
            tl = list(map(int, time_str.split(":")))
            tn = tl[0] * 3600 + tl[1] * 60 + tl[2]
            return tn

        def check(event, cur_time):
            last_time = LAST_EVENT_TIME[event]
            if last_time == None:
                return True
            else:
                diff = cur_time - last_time
                if diff > DELAY_TIME:
                    return True
                else:
                    return False

        def update(event, cur_time):
            LAST_EVENT_TIME[event] = cur_time
        # event insert delay code end
        from datetime import datetime

        while True:
            event = event_queue.get()
            print(event)
            if event is not None:
                event_type = event['action']
                if event_type == "emotion":                                    
                    combine_list = event['combine_list']                                
                    snapshot_data_list = []
                    body_data_list = []                
                    for emotion_data in combine_list: 
                        cctv_id = emotion_data['cctv_id']
                        track_id = emotion_data['id']

                        emotion_index = emotion_data['emotion_index']
                        db_insert_file_path = emotion_data['db_insert_file_path'] #TODO rename

                        combine_dict = emotion_data['combine_dict']               
                        body_temperature = combine_dict['temperature']
                        breath_rate = combine_dict['breath']
                        heartbeat_rate = combine_dict['heart']
                        
                        people_name = "Dummy"   #TODO rename
                        people_id = track_id         #TODO rename
                        people_color_num = track_id  #TODO rename
                        
                        # realtime_datetime = event['current_datetime']
                        realtime_datetime = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

                        s_data = [cctv_id, track_id, people_name, db_insert_file_path]                  
                        snapshot_data_list.append(s_data)
                        b_data = [cctv_id, people_id, heartbeat_rate, breath_rate, body_temperature, emotion_index, realtime_datetime]
                        body_data_list.append(b_data)            
                    print(f"insert_snapshot 호출 전 snapshot_data_list : {snapshot_data_list}")
                    self.insert_snapshot(snapshot_data_list, conn, mq_conn)  
                    self.insert_realtime(body_data_list, conn)  
                    continue
                try:
                    cctv_id = event['cctv_id']
                    event_type = event['action']
                    track_id = event['id']       
                    event_location = event['location']
                    current_datetime = event['current_datetime']
                    event_date = copy.deepcopy(str(current_datetime)[:10])
                    event_time = copy.deepcopy(str(current_datetime)[11:19])
                    event_start_datetime = copy.deepcopy(str(current_datetime)[:19])
                    event_end_datetime = event_start_datetime # 이 부분은 추후에 event_start_datetime + 2~4초 정도로 수정할 예정
                    event_clip_directory = "As soon as"
                    event_start = event_start_datetime
                    event_end = event_end_datetime

                    self.db_logger.info(f"[EVENT DETECT] - cctv_id : {cctv_id}, event_type : {event_type}, event_location : {event_location}, track_id : {track_id}, event_date : {event_date}, event_time : {event_time}, event_clip_directory : {event_clip_directory}, event_start : {event_start}, event_end : {event_end}")
                    event_cur_time = str_to_second(event_time) # event insert delay code use here
                    if check(event_type, event_cur_time): # event insert delay code use here
                        
                        try:
                            cur = conn.cursor()
                            # sql_bring_people_id = """
                            # SELECT people_id 
                            # FROM people 
                            # WHERE people_color_num = %s AND cctv_id = %s
                            # """
                            sql_bring_people_id = "SELECT people_color_num FROM people WHERE people_color_num = %s AND cctv_id = %s"
                            # print(f"track_id : {track_id}") 
                            # print(f"cctv_id : {cctv_id}")
                            cur.execute(sql_bring_people_id, (track_id, cctv_id))
                            people_table_id = cur.fetchone()
                            if people_table_id:
                                people_table_id = people_table_id[0]
                            else:
                                self.db_logger.warn("'[INSERT_EVENT] : people_id' that satisfies the condition does not exist.")
                                
                            DB_insert_sql = """
                            INSERT INTO event 
                            (cctv_id, event_type, event_location, event_detection_people, 
                            event_date, event_time, event_clip_directory, 
                            event_start, event_end) 
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """
                            values = [cctv_id, event_type, event_location, people_table_id, event_date, event_time, event_clip_directory, event_start, event_end]
                            print(values)
                            cur.execute(DB_insert_sql, values)
                            self.db_logger.info("Event insertion complete.")

                            try:
                                event_id = cur.lastrowid
                                self.db_logger.info(f"lastrowid로 가져온 event_id : {event_id}")
                                try:
                                    message = {"alert_type": "event", "event_id": event_id}
                                    self.publish_message_mq(message, mq_conn)
                                except Exception as e:
                                    self.db_logger.warning(f"Can not find event_id, {e}")
                                    pass
                            except Exception as e:
                                self.db_logger.warning(f"Error occurred in message Queue, error, {e}")
                        except Exception as e:
                            self.db_logger.warning(f"Error occurred during event data insertion, error: {e}")
                        finally:
                            conn.commit()
                            update(event_type, event_cur_time) # event insert delay code use here
                except Exception as e:
                    self.db_logger.warning(f"Error occurred in insert_event loop, error: {e}")
                    pass
    #endregion DB query