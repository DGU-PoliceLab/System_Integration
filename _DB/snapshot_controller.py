import os
from datetime import datetime
from pytz import timezone
import copy
import cv2
import time  # 추가
from sklearn.metrics.pairwise import cosine_similarity

from _DB.db_controller import connect_db, insert_snapshot, delete_snapshot, update_snapshot
from _DB.mq_controller import connect_mq
from _Utils.logger import get_logger
from variable import get_arg

NAS_PATH = get_arg('root', 'nas_path')
LOGGER = get_logger(name = '[SNAPSHOT]', console = True, file = False)
MAX_PERSON_NUM = get_arg('root','max_person_num')


def create_folder(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
            LOGGER.info(f"Folder created at: {path}")
        # else:
            # LOGGER.warning(f"Folder already exists at: {path}")
    except Exception as e:
        LOGGER.warning(f"Error occurred during create folder: {e}")

def calculate_histogram(image):
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def check_similar(image1, image2, threshold=0.8):
    """
    Check if two images are similar based on cosine similarity of their histograms.
    """
    hist1 = calculate_histogram(image1)
    hist2 = calculate_histogram(image2)
    similarity = cosine_similarity([hist1], [hist2])[0][0]
    return similarity > threshold

def object_snapshot_control(data_pipe):
    
    conn = connect_db("mysql-pls")
    mq_conn = connect_mq()
    if conn.open:
        pass
    else:
        LOGGER.info("FUNCTION object_snapshot_control : Database connection failed")
        print(f"FUNCTION object_snapshot_control : Database connection failed")

    
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
            create_folder(save_path)


        for i, track in enumerate(tracks):
            # print(f"i : {i}, track : {track}")
            tlwh = track.tlwh
            tid = track.track_id
            a = 30
            x1 = int(tlwh[0])
            y1 = int(tlwh[1])
            x2 = int(tlwh[0] + tlwh[2])
            y2 = int(tlwh[1] + tlwh[3])
            body_cutting_frame = copy.deepcopy(frame[y1:y2, x1:x2]) # 새로운 객체별 몸통 bbox 추론
            body_cutting_frames[tid] = body_cutting_frame
            if body_cutting_frames[tid] is None:
                LOGGER.warning(f"Frame is Empty(body_cutting_frames[{tid}])")
                
            people_thumbnail_location_link = None
            # current_datetime = tracking_time.strftime("%Y-%m-%d_%H:%M:%S")
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
                LOGGER.warning(f"Error occurred while sending to the database, error: {e}")

            try:
                if body_cutting_frames[tid].size > 0:
                    cv2.imwrite(f"{file_path}.jpg", body_cutting_frames[tid])
                    LOGGER.info(f"Save complete(OBJECT #{tid}).")
                else:
                    LOGGER.warning(f"Snapshot is empty(OBJECT #{tid}).")
                    x1 = max(0, int(tlwh[0]) - a)
                    y1 = max(0, int(tlwh[1]) - a)
                    x2 = min(frame.shape[1], int(tlwh[0] + tlwh[2]) + a)
                    y2 = min(frame.shape[0], int(tlwh[1] + tlwh[3]) + a)
                    body_cutting_frame = copy.deepcopy(frame[y1:y2, x1:x2])
                    body_cutting_frames[tid] = body_cutting_frame
                    cv2.imwrite(f"{file_path}.jpg", body_cutting_frames[tid])
                    LOGGER.info(f"Save Edited snapshot(OBJECT #{tid}).")
            except Exception as e:
                LOGGER.warning(f"Error occurred while saving OBJECT #{tid} from CCTV #{cctv_id}, error: {e}")  

    

        # 데이터베이스에 한꺼번에 업데이트
        if total_db_insert_datas:
            current_time = time.time()
            if  current_time - last_update_time >= 10:  # Check if 10 seconds have passed
                # LOGGER.info(f"[snapshot_controller] - first_run : {first_run}, total_db_insert_datas : {total_db_insert_datas}")
                insert_snapshot(total_db_insert_datas, conn, mq_conn)
                last_update_time = current_time  # Update the last update time
            total_db_insert_datas.clear()  # 리스트 초기화
            