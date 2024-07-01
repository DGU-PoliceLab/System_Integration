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
from _DB.event_controller import collect_evnet
from _DB.snapshot_controller import object_snapshot_control
from _Utils.logger import get_logger
from _Utils.concat_frame_to_video import save_vid_clip
from _Utils.head_bbox import *
from _Utils.pipeline import *
from _Utils._time import process_time_check
import _Utils.draw_bbox_skeleton as draw_bbox_skeleton 
import _MOT.face_detection as face_detection
from _Sensor.radar import radar_start
from _HAR.PLASS.selfharm import Selfharm
from _HAR.CSDC.falldown import Falldown
from _HAR.HRI.emotion import Emotion 
from _HAR.MHNCITY.violence import Violence
from variable import get_root_args, get_sort_args
from rtmo import get_model

DEBUG_MODE = True

# 출력 로그 설정
LOGGER = get_logger(name= '[RUN]', console= True, file= True)

def wait_subprocess_ready(name, pipe):
    while True:
        LOGGER.info(f'wating for {name} process to ready...')
        if pipe.recv():
            LOGGER.info(f'{name} process ready')
            break
        else:
            time.sleep(0.1)

def main():
    # 루트 인자 및 기타 인자 설정
    args = get_root_args()
    dict_args = vars(args)
    bot_sort_args = get_sort_args()
    bot_sort_args.ablation = False
    bot_sort_args.mot20 = not bot_sort_args.fuse_score

    # 멀티프로세스 환경 torch, cuda 설정
    torch.multiprocessing.set_start_method('spawn')
    
    object_snapshot_control_queue = Queue()
    
    # 프로세스간 데이터 전달을 위한 파이프라인 생성
    selfharm_input_pipe, selfharm_output_pipe = Pipe()
    falldown_input_pipe, falldown_output_pipe = Pipe()
    emotion_input_pipe, emotion_output_pipe = Pipe()
    violence_input_pipe, violence_output_pipe = Pipe()

    # 이벤트 처리를 위한 수집을 위한 파이프라인 생성
    event_input_pipe, event_output_pipe = Pipe()

    # 모듈별 프로세스 생성
    selfharm_process = Process(target=Selfharm, args=(selfharm_output_pipe, event_input_pipe,))
    falldown_process = Process(target=Falldown, args=(falldown_output_pipe, event_input_pipe,))
    emotion_process = Process(target=Emotion, args=(emotion_output_pipe, event_input_pipe,))
    violence_process = Process(target=Violence, args=(violence_output_pipe, event_input_pipe,))
    
    # 이벤트 프로세스 생성
    event_process = Process(target=collect_evnet, args=(event_output_pipe,))

    # 모듈별 프로세스 시작
    # selfharm_process.start()
    # falldown_process.start()
    # emotion_process.start()
    # violence_process.start()

    # 이벤트 프로세스 시작
    event_process.start()

    # 디버그 모드
    if DEBUG_MODE == True:
        # DB 연결 및 CCTV 정보 조회
        source = "_Input/videos/mhn_demo_1.mp4" 
        cctv_info = dict()
        cctv_info['cctv_ip'] = -1
        cctv_info['cctv_id'] = -1
        cctv_info['cctv_name'] = -1
    else:
        # DB 연결 및 CCTV 정보 조회
        try:
            conn = connect_db("mysql-pls")
            if conn.open:
                if dict_args['video_file'] != "":
                    cctv_info = get_cctv_info(conn)
            else:
                LOGGER.warning('RUN-CCTV Database connection is not open.')
                cctv_info = {'cctv_id': 404}
        except Exception as e:
            LOGGER.warning(f'Unable to connect to database, error: {e}')
            cctv_info = {'cctv_id': 404}

        cctv_info = cctv_info[1]
        source = cctv_info['cctv_ip']
        Process(target=object_snapshot_control, args=(object_snapshot_control_queue,)).start()
    # CCTV 정보 출력
    LOGGER.info(f"cctv_info : {cctv_info}")

    # 사람 감지 및 추적을 위한 모델 로드
    inferencer, init_args, call_args, display_alias = get_model()

    # 동영상 관련 설정
    now = datetime.now()
    timestamp = str(now).replace(" ", "").replace(":",";")
    cap = cv2.VideoCapture(source)
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    fps = 30
    num_frame = 0
    if cap.get(cv2.CAP_PROP_FPS):
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    tracker = BoTSORT(bot_sort_args, fps)

    # 디버그(시각화, 동영상 저장) 
    if args.debug_visualize:    
        out = cv2.VideoWriter(f'/System_Integration/_Output/video_clip_{timestamp}.avi', fourcc, fps, (int(w), int(h))) 

    # _HAR 모듈 실행 대기
    # wait_subprocess_ready("Selfharm", selfharm_input_pipe)
    # wait_subprocess_ready("Falldown", falldown_input_pipe)
    # wait_subprocess_ready("Emotion", emotion_input_pipe)
    # wait_subprocess_ready("Violence", violence_input_pipe)

    # 사람 감지 및 추적
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            draw_frame = frame.copy()
            current_datetime = datetime.now()
            detections = []
            skeletons = []
            temp_call_args = copy.deepcopy(call_args)
            temp_call_args['inputs'] = frame         

            for _ in inferencer(**temp_call_args):
                pred = _['predictions'][0]
                l_p = len(pred)
                LOGGER.info(f'frame #{num_frame} pose_results- {l_p} person detect!')
                n_person = 1
                pred.sort(key = lambda x: x['bbox'][0][0])

                for p in pred:
                    keypoints = p['keypoints']
                    keypoints_scores = p['keypoint_scores']
                    detection = [*p['bbox'][0], p['bbox_score']]
                    detections.append(detection)
                    skeletons.append([a + [b] for a, b in zip(keypoints, keypoints_scores)])
                    if args.debug_visualize:
                        draw_frame = draw_bbox_skeleton.draw(draw_frame, n_person, detection, keypoints)
                    n_person += 1   
            
            detections = np.array(detections, dtype=np.float32)
            skeletons = np.array(skeletons, dtype=np.float32)
            online_targets = tracker.update(detections, skeletons, frame)
            num_frame += 1
            tracks = online_targets # 모듈로 전달할 감지 결과
            meta_data = {'cctv_id': cctv_info['cctv_id'], 'current_datetime': current_datetime, 'cctv_name': cctv_info['cctv_name'], 'num_frame':num_frame, 'frame_size': (int(w), int(h))} # 모듈로 전달할 메타데이터
            input_data = [tracks, meta_data] # 모듈로 전달할 데이터
            e_input_data = [frame, meta_data]
            
            # 모듈로 데이터 전송
            # selfharm_input_pipe.send(input_data)
            # falldown_input_pipe.send(input_data)
            # emotion_input_pipe.send(e_input_data)
            # violence_input_pipe.send(input_data)

            if args.debug_visualize:
                out.write(draw_frame)
        else:
            break
    if args.debug_visualize:
        out.release()
    cap.release()
    event_process.kill()

if __name__ == '__main__':
    main()