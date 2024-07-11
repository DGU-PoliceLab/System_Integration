from _Utils.logger import get_logger
import time
import pymysql

radar_data = []
emotion_result_data = []

LOGGER = get_logger(name = '[LOGGER]')

def manage_queue_size(queue, size):
    while queue.qsize() > size:
        queue.get()

def sychronize_data(radar_data, emotion_data, thermal_data, realtime_queue):
    for thermal in thermal_data:
        if ((thermal['id']+1) == emotion_data['id']):
            matched_data = (emotion_data['cctv_id'], emotion_data['id'], radar_data[1], radar_data[2], thermal['temp'], emotion_data['mapped_emotion_results'][0], radar_data[3])
            realtime_queue.put(matched_data)
            print(f"realtime_queue.qsize() : {realtime_queue.qsize()}")
            LOGGER.info(f'Matched data added: {matched_data}')

def collect_realtime(data_pipe, emotion_queue=None, radar_queue=None, realtime_queue=None):
    logger = get_logger(name= '[COLLECT_REALTIME]', console= True, file= False)
    data_pipe.send(True)
    while True:
        thermal_data = data_pipe.recv() # [{"id": i, "temp": skin_surface_temp}] 이런 형태의 값이 들어옴 (열화상 센서 값)
        if not radar_queue.empty():
            vital_id, heartbeat_rate, breath_rate, current_datetime, cctv_id = radar_queue.get()
            radar_data = [vital_id, heartbeat_rate, breath_rate, current_datetime, cctv_id]
            if not emotion_queue.empty() and radar_data is not None:
                try:
                    emotion_data = emotion_queue.get()
                    # print(f"emotion_data : {emotion_data}")
                    # print(f"thermal_data : {thermal_data}")
                except Exception as e:
                    emotion_data = None
                    thermal_data = None
                    print("Get Radar Error!!")
                    print(e)
                
                print(emotion_data)
                if radar_data is not None and emotion_data is not None and thermal_data is not None:
                    # 이 아래에 있는 함수 호출이 안됨. 락걸림.
                    sychronize_data(radar_data, emotion_data, thermal_data, realtime_queue) 
        else:
            time.sleep(0.00001)


# 레이더와 이모션 결과를 정합해서 realtime_queue에 넣음
# def collect_realtime(radar_queue=None, emotion_queue=None, realtime_queue=None):
#     # assert radar_queue is None and emotion_queue is None:, 'At least one parameter is required(rader or emotion)'
#     print('collect_realtime 진입')
#     while True:
#         if not radar_queue.empty():
#             vital_id, heartbeat_rate, breath_rate, current_datetime, cctv_id = radar_queue.get()
#             radar_data = [vital_id, heartbeat_rate, breath_rate, current_datetime, cctv_id]
#             # print(f"vital_id : {vital_id}, heartbeat_rate : {heartbeat_rate}, breath_rate : {breath_rate}, current_datetime : {current_datetime}, cctv_id : {cctv_id}")
#             # print(f"radar_queue.qsize() : {radar_queue.qsize()}")
#             # try:
#             #     vital_id, heartbeat_rate, breath_rate, current_datetime, cctv_id = radar_queue.get()
#             #     print("rader_data")
#             #     print(vital_id, heartbeat_rate, breath_rate, current_datetime, cctv_id)
#             # except Exception as e:
#             #     print("Get Radar Error!!")
#             #     print(e)
#             #     radar_data = None

#             if not emotion_queue.empty() and radar_data is not None:
#                 print("Start Synchronizing")
#                 try:
#                     emotion_data = emotion_queue.get()
#                     print(f"emotion_data : {emotion_data}")
#                 except Exception as e:
#                     emotion_data = None
#                     print("Get Radar Error!!")
#                     print(e)
                
#                 print(emotion_data)
#                 if radar_data is not None and emotion_data is not None:
#                     # 이 아래에 있는 함수 호출이 안됨. 락걸림.
#                     sychronize_data(radar_data, emotion_data, realtime_queue) 
#         else:
#             time.sleep(0.00001)


def get_cctv_info(conn: pymysql.connections.Connection):
    """
    Get CCTV info (include rader, thermal) from database.
    """
    cctv_info = []
    with conn.cursor() as cur:
        sql = "SELECT * FROM cctv"
        cur.execute(sql)
        response = cur.fetchall()
    for row in response:
        cctv_info.append({'cctv_id': row[0], 'cctv_ip': row[1], 'rader_ip': row[2], 'thermal_ip': row[3], 'cctv_name': row[4]})
    return cctv_info