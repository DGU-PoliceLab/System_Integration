import time
from multiprocessing import Process, Pipe, Queue
from Utils.logger import get_logger
from threading import Thread
from queue import Queue
from variable import get_debug_args, get_arg
import pymysql
import config
import requests
import pika
import json
import datetime
import copy
import config

ENDPOINT = "https://was:40000"



            
class EventHandler:
    def __init__(self, is_debug=False):
        self.mq_logger = get_logger(name= '[MQ]', console= True, file= True)
        self.db_logger = get_logger(name = '[DB]', console=False, file=False)
        self.event_logger = get_logger(name= '[EVENT]', console= True, file= True)
        
        self.LAST_EVENT_TIME = {"falldown": None, "selfharm": None, "longterm_status": None, "violence": None}
        self.DELAY_TIME = get_arg('root', 'event_delay')

        # if is_debug:
        #     self.db_conn = self.connect_db()
        # else:
        #     self.db_conn = self.connect_db(config=config.db_config)

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
            pass
        else:
            self.mq_logger.error("Failed to open MQ channel.")
        return channel

    def update(self, pipe):        
        # if is_debug == False:
        import config
        # conn = self.connect_db(config.db_config)
        # mq_conn = self.connect_mq(config.mq_config)

        event_queue = Queue()
        insert_db_thread = Thread(target=self.insert_event, args=(event_queue))
        insert_db_thread.start()
        while True:
            event = pipe.recv()
            
            if event is not None:
                # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                # print(event)
                event_queue.put(event)
                # logger.info(f"event_queue.size: {event_queue.qsize()}")
            else:
                time.sleep(0.0001)



    #region MQ
    def publish_message_mq(self, message, channel):
        assert channel is not None, 'FUNCTION publish_message : channel object does not exist'
        self.mq_logger.info("publish_message 실행")
        try:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(message)
            # channel.basic_publish(exchange=config.mq_config['exchange'], body=json.dumps(message), routing_key='')
            # url = ENDPOINT + "/message/send"
            # response = requests.post(url, {"key": "event", "message": {"event": ""}})

        except Exception as e:
            self.mq_logger.warning(f'FUNCTION publish_message : Error occurred during message publishing, error: {e}')
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
                    values = [data[1], data[0], data[2], data[1], data[3]]
                    cur.execute(sql, values)
                conn.commit()                      
            message = {"alert_type": "people"}            
            self.publish_message_mq(message, mq_conn)
        except Exception as e:
            self.db_logger.warning(f"Error occurred during insert data into database, error: {e}")

    def insert_event(self, event_queue):
        # assert conn is not None, 'conn object does not exist'
        # assert mq_conn is not None, 'mq_conn object does not exist'
        
        def str_to_second(time_str):
            tl = list(map(int, time_str.split(":")))
            tn = tl[0] * 3600 + tl[1] * 60 + tl[2]
            return tn

        def check(event, cur_time):
            last_time = self.LAST_EVENT_TIME[event]
            if last_time == None:
                return True
            else:
                diff = cur_time - last_time
                if diff > self.DELAY_TIME:
                    return True
                else:
                    return False

        # event insert delay code end
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
                        
                        # realtime_datetime = event['current_datetime']
                        realtime_datetime = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

                        s_data = [cctv_id, track_id, people_name, db_insert_file_path]                  
                        snapshot_data_list.append(s_data)
                        b_data = [cctv_id, people_id, heartbeat_rate, breath_rate, body_temperature, emotion_index, realtime_datetime]
                        body_data_list.append(b_data)            
                    #####
                    # self.insert_snapshot(snapshot_data_list, conn, mq_conn)  
                    # self.insert_realtime(body_data_list, conn)  
                    ####
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
                        ####
                        # cur = conn.cursor()
                        ####

                        sql_bring_people_id = "SELECT people_color_num FROM people WHERE people_color_num = %s AND cctv_id = %s"
                        # ####
                        # cur.execute(sql_bring_people_id, (track_id, cctv_id))
                        # people_table_id = cur.fetchone()
                        # ####

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
                        ####
                        # cur.execute(DB_insert_sql, values)
                        # self.db_logger.info("Event insertion complete.")
                        # # event_id = cur.lastrowid
                        # self.db_logger.info(f"lastrowid로 가져온 event_id : {event_id}")
                        # message = {"alert_type": "event", "event_id": event_id}
                        ####
                        # self.publish_message_mq(message, mq_conn)
                        # conn.commit()
                        ####
                        self.LAST_EVENT_TIME[event] = event_cur_time
                except Exception as e:
                    self.db_logger.warning(f"Error occurred in insert_event loop, error: {e}")
                    pass
    #endregion DB query