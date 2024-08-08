from Utils.logger import get_logger
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
        try:
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
                    
                    matched_data = (emotion_data['cctv_id'], emotion_data['id'], radar_data[1], radar_data[2], thermal['temp'], emotion_data['mapped_emotion_results'][0], radar_data[3])
                    realtime_queue.put(matched_data)
                
            else:
                time.sleep(0.00001)
        except:
            print("pipeline.py pipe 오류 발생")
            time.sleep(0.00001)

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