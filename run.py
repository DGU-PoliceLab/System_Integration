import sys
sys.path.insert(0, "/System_Integration/")
sys.path.insert(0, "/System_Integration/_PoseEstimation/mmlab")
sys.path.insert(0, "/System_Integration/_Tracker")
sys.path.insert(0, "/System_Integration/_HAR")
sys.path.insert(0, "/System_Integration/_MOT")

import copy

import threading
from threading import Thread
from queue import Queue
from multiprocessing import Process, Pipe
import multiprocessing
from datetime import datetime
from pytz import timezone
import time
import cv2
import numpy as np

from _Tracker.BoTSORT.tracker.bot_sort import BoTSORT

import pymysql
from _DB.db_controller import connect_db, insert_event, insert_realtime
from _DB.mq_controller import connect_mq
from _DB.snapshot_controller import object_snapshot_control
from _Utils.logger import get_logger
from _Utils.concat_frame_to_video import save_vid_clip
from _Utils.head_bbox import *
from _Utils.pipeline import *
import _Utils.draw_bbox_skeleton as draw_bbox_skeleton 
# from _Utils.socket_udp import get_sock, socket_distributor, socket_collector, socket_provider
from _Utils.socket_tcp import socket_server_thread, SocketProvider

from _Sensor.radar import radar_start

from _HAR.PLASS.selfharm import Selfharm
from _HAR.CSDC.falldown import Falldown
from _HAR.HRI.emotion import Emotion 
from variable import get_root_args, get_sort_args

from rtmo import get_model
import pickle

# variable로 빼기
TIME_ZONE = timezone('Asia/Seoul')
DEBUG_MODE = True

def main():
    logger = get_logger(name= '[RUN]', console= True, file= True)
    # update by hyunsu, kim
    # :소캣 객체를 생성
    sock = get_sock()

    args = get_root_args()
    dict_args = vars(args)
    bot_sort_args = get_sort_args()
    bot_sort_args.ablation = False
    bot_sort_args.mot20 = not bot_sort_args.fuse_score


    object_snapshot_control_queue = multiprocessing.Queue()
    realtime_status_queue = Queue()

    if DEBUG_MODE == True:
        # DB 연결 및 CCTV 정보 조회
        source = "_Input/videos/mhn_demo_1.mp4" 
        mq_conn = None
        realtime_status_conn = None
        cctv_info = dict()
        cctv_info['cctv_ip'] = -1
        cctv_info['cctv_id'] = -1
        cctv_info['cctv_name'] = -1
    else:
        try:
            conn = connect_db("mysql-pls")
            if conn.open:
                if dict_args['video_file'] != "":
                    cctv_info = get_cctv_info(conn)
            else:
                logger.warning('RUN-CCTV Database connection is not open.')
                cctv_info = {'cctv_id': 404}
        except Exception as e:
            logger.warning(f'Unable to connect to database, error: {e}')
            cctv_info = {'cctv_id': 404}

        mq_conn = connect_mq()
        realtime_status_conn = connect_db("mysql-pls")
        cctv_info = cctv_info[1]
        source = cctv_info['cctv_ip']
        multiprocessing.Process(target=object_snapshot_control, args=(object_snapshot_control_queue,)).start()

    logger.info(f"cctv_info : {cctv_info}")
    
    # update by hyunsu, kim
    # :소캣통신을 위한 쓰레드 정의
    # distributor_thread = Thread(target=socket_distributor, daemon=False).start()
    # collector_thread = Thread(target=socket_collector, daemon=False).start()
    socket_server_thread(20000)

    inferencer, init_args, call_args, display_alias = get_model()

    now = datetime.now(TIME_ZONE)
    timestamp = str(now).replace(" ", "").replace(":",";")
    cap = cv2.VideoCapture(source)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', 'v', '4')
    fps = 30
    out = cv2.VideoWriter(f'/System_Integration/_Output/video_clip_{timestamp}.mp4', fourcc, fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))) 
    num_frame = 0

    selfharm_thread = Thread(target=Selfharm, daemon=False).start()
    socket_provider = SocketProvider()
    socket_provider.start(target=20000)

    if cap.get(cv2.CAP_PROP_FPS):
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    tracker = BoTSORT(bot_sort_args, fps)
    
    import _MOT.face_detection as face_detection
    face_detector = face_detection.build_detector('RetinaNetResNet50', confidence_threshold=.5, nms_iou_threshold=.3)

    if cap.isOpened():
        while True:
            ret, frame = cap.read()
            drawed_frame = None
            if ret:
                current_datetime = datetime.now(TIME_ZONE)
                detections = []
                skeletons = []
                
                temp_call_args = copy.deepcopy(call_args)
                temp_call_args['inputs'] = frame
                preprocessing_time = time.time()               

                for _ in inferencer(**temp_call_args):

                    pred = _['predictions'][0]
                   
                    l_p = len(pred)
                    logger.info(f'frame #{num_frame} pose_results- {l_p} person detect!')
                    n_person = 1
                    pred.sort(key = lambda x: x['bbox'][0][0])
                    for p in pred:
                        keypoints = p['keypoints']
                        keypoints_scores = p['keypoint_scores']

                        detection = [*p['bbox'][0], p['bbox_score']]

                        detections.append(detection)
                        skeletons.append([a + [b] for a, b in zip(keypoints, keypoints_scores)])
                        n_person += 1   
                detections = np.array(detections, dtype=np.float32)
                skeletons = np.array(skeletons, dtype=np.float32)
                online_targets = tracker.update(detections, skeletons, frame)
                
                num_frame += 1
                tracks = online_targets
                meta_data = {'cctv_id': cctv_info['cctv_id'], 'current_datetime': current_datetime, 'cctv_name': cctv_info['cctv_name'], 'num_frame':num_frame, 'frame_size': [w, h]}
                # logger.warning(num_frame)
                input_data = [tracks, meta_data]
                # update by hyunsu, kim
                # :생성된 데이터를 소캣 통신으로 보내는 부분 (이 과정에서 데이터는 직렬화되어 전송됨)
                socket_provider(sock, 20000, input_data)
                process = round(float(time.time() - preprocessing_time),4)                
                logger.info(f"전처리 처리 시간 정보 : frame : {num_frame} time : {process} ")
            else:
                break
    else:
        logger.error('video error')
    out.release()
    cap.release()

if __name__ == '__main__':
    main()