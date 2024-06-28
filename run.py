import sys
sys.path.insert(0, "/System_Integration/")
sys.path.insert(0, "/System_Integration/_PoseEstimation/mmlab")
sys.path.insert(0, "/System_Integration/_Tracker")
sys.path.insert(0, "/System_Integration/_HAR")
sys.path.insert(0, "/System_Integration/_MOT")
import copy
from threading import Thread
from multiprocessing import Process, Queue, Pipe
import cv2
import torch
import numpy as np
import time
from datetime import datetime
from pytz import timezone

from _Tracker.BoTSORT.tracker.bot_sort import BoTSORT
from _DB.db_controller import connect_db, insert_event, insert_realtime
from _DB.mq_controller import connect_mq
from _DB.snapshot_controller import object_snapshot_control
from _Utils.logger import get_logger
from _Utils.concat_frame_to_video import save_vid_clip
from _Utils.head_bbox import *
from _Utils.pipeline import *
from _Utils._time import process_time_check
import _Utils.draw_bbox_skeleton as draw_bbox_skeleton 
# from _Utils._port import kill_process
# from _Utils.socket_udp import get_sock, socket_distributor, socket_collector, socket_provider
# from _Utils.socket_tcp import socket_server_thread, SocketProvider, SocketConsumer
from _Sensor.radar import radar_start
from _HAR.PLASS.selfharm import Selfharm
from _HAR.CSDC.falldown import Falldown
from _HAR.HRI.emotion import Emotion 
from variable import get_root_args, get_sort_args

from rtmo import get_model
import pickle

TIME_ZONE = timezone('Asia/Seoul')
DEBUG_MODE = True

def main():
    # 출력 로그 설정
    logger = get_logger(name= '[RUN]', console= False, file= True)

    # 루트 인자 및 기타 인자 설정
    args = get_root_args()
    dict_args = vars(args)
    bot_sort_args = get_sort_args()
    bot_sort_args.ablation = False
    bot_sort_args.mot20 = not bot_sort_args.fuse_score

    torch.cuda.empty_cache()
    torch.multiprocessing.set_start_method('spawn')
    object_snapshot_control_queue = Queue()
    # 프로세스간 데이터 전달을 위한 파이프라인 생성
    selfharm_input_pipe, selfharm_output_pipe = Pipe()

    # 기능별 프로세스 생성
    selfharm_process = Process(target=Selfharm, args=(selfharm_output_pipe,))
    selfharm_process.start()

    # 디버그 모드
    if DEBUG_MODE == True:
        # DB 연결 및 CCTV 정보 조회
        source = "_Input/videos/mhn_demo_2.mp4" 
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

        cctv_info = cctv_info[1]
        source = cctv_info['cctv_ip']
        Process(target=object_snapshot_control, args=(object_snapshot_control_queue,)).start()
    # CCTV 정보 출력
    logger.info(f"cctv_info : {cctv_info}")
    
    # # 소켓 통신을 위한 소켓 객체 생성
    # try:
    #     socket_server_thread(20000) # 입력 소켓
    # except:
    #     kill_process(20000)
    #     socket_server_thread(20000)
    # try:
    #     socket_server_thread(20001) # 출력 소켓
    # except:
    #     kill_process(20001)
    #     socket_server_thread(20001) # 출력 소켓

    # 사람 감지 및 추적을 위한 모델 로드
    inferencer, init_args, call_args, display_alias = get_model()

    # 동영상 관련 설정
    now = datetime.now(TIME_ZONE)
    timestamp = str(now).replace(" ", "").replace(":",";")
    cap = cv2.VideoCapture(source)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', 'v', '4')
    fps = 30
    out = cv2.VideoWriter(f'/System_Integration/_Output/video_clip_{timestamp}.mp4', fourcc, fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))) 
    num_frame = 0
    if cap.get(cv2.CAP_PROP_FPS):
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    tracker = BoTSORT(bot_sort_args, fps)

    # # 입력 소캣에 데이터 공급을 위한 소캣 객체 생성
    # socket_provider = SocketProvider()
    # socket_provider.start(target=20000)

    # # 출력 소캣에 데이터를 소비를 위한 소캣 객체 생성
    # socket_consumer = SocketConsumer()
    # socket_consumer.start(target=20001)
    
    import _MOT.face_detection as face_detection
    face_detector = face_detection.build_detector('RetinaNetResNet50', confidence_threshold=.5, nms_iou_threshold=.3)

    # _HAR 모듈 실행 대기
    while True:
        selfharm_ready = selfharm_input_pipe.recv()
        if selfharm_ready:
            break
        else:
            time.sleep(0.1)


    if cap.isOpened():
        while True:
            ret, frame = cap.read()
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
                input_data = [tracks, meta_data]
                # socket_provider.send(input_data)
                
                # 모듈로 데이터 전송
                selfharm_input_pipe.send(input_data)

                time.sleep(0.0001)
            else:
                selfharm_process.join()
                break
    else:
        logger.error('video error')
    out.release()
    cap.release()

if __name__ == '__main__':
    main()