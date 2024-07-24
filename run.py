import sys
sys.path.insert(0, "/System_Integration/")
sys.path.insert(0, "/System_Integration/PoseEstimation/mmlab")
sys.path.insert(0, "/System_Integration/Tracker")
sys.path.insert(0, "/System_Integration/HAR")
sys.path.insert(0, "/System_Integration/MOT")
import os
import copy
from multiprocessing import Process, Pipe
import cv2
import torch
import numpy as np
import time
import json
from datetime import datetime
import atexit
from Tracker.BoTSORT.tracker.bot_sort import BoTSORT
from Sensor.EdgeCam import EdgeCam
from Utils.logger import get_logger
from Utils.head_bbox import *
from Utils.pipeline import *
import Utils.draw_bbox_skeleton as draw_bbox_skeleton 
import MOT.face_detection as face_detection
from HAR.PLASS.selfharm import Selfharm
from HAR.CSDC.falldown import Falldown
from HAR.HRI.emotion import Emotion
from HAR.MHNCITY.violence.violence import Violence
from variable import get_root_args, get_sort_args, get_scale_args, get_debug_args, get_rader_args, get_thermal_args
import PoseEstimation.mmlab.rtmo as rtmo
from EventHandler import EventHandler

#TODO
# sensor 관련은 EdgeCam class을 통해서 사용하도록
# EventHandler로 일단 내용 정리(내부적으론 DB와 MQ 둘다 사용)
# 각 탐지 모듈 코드 최적화
# 시간적 성능 측정 방법 

#
# variable 프리셋 만들기
# args 싱글톤
# 자해 모듈 트레커 사용하도록
# 낙상/쓰러짐 모듈 tid
# 트레커 reid 기능

def main():
    # 출력 로그 설정
    logger = get_logger(name= '[RUN]', console= False, file= False)
    # 루트 인자 및 기타 인자 설정
    args = get_root_args()
    dict_args = vars(args)
    bot_sort_args = get_sort_args()
    bot_sort_args.ablation = False
    bot_sort_args.mot20 = not bot_sort_args.fuse_score
    rader_args = get_rader_args()
    thermal_args = get_thermal_args()
    debug_args = get_debug_args()
    scale_args = get_scale_args()

    def check_args():
        if debug_args.debug == False:
            logger.info("Unsupported arguments")
            logger.info("debug_args.debug == False")
            exit()
        pass
    check_args()

    torch.multiprocessing.set_start_method('spawn') # See "https://tutorials.pytorch.kr/intermediate/dist_tuto.html"
    
    evnt_handler = EventHandler.EventHandler(is_debug=debug_args.debug)
        
    # 이벤트 처리를 위한 수집을 위한 파이프라인 생성
    event_input_pipe, event_output_pipe = Pipe()
    
    # 이벤트 프로세스
    event_process = Process(target=evnt_handler.update, args=(event_output_pipe, debug_args.debug))
    event_process.start()
    
    process_list = []
    if 'selfharm' in args.modules:
        selfharm_pipe_list = []
        for _ in range(scale_args.selfharm):
            selfharm_input_pipe, selfharm_output_pipe = Pipe()
            selfharm_pipe_list.append((selfharm_input_pipe, selfharm_output_pipe))
        for i in range(scale_args.selfharm):
            selfharm_process = Process(target=Selfharm, args=(selfharm_pipe_list[i][1], event_input_pipe,), name=f"Selfharm_Process_{i}")
            process_list.append(selfharm_process)
            selfharm_process.start()

    if 'falldown' in args.modules:
        falldown_pipe_list = []
        for _ in range(scale_args.falldown):
            falldown_input_pipe, falldown_output_pipe = Pipe()
            falldown_pipe_list.append((falldown_input_pipe, falldown_output_pipe))
        for i in range(scale_args.falldown):
            falldown_process = Process(target=Falldown, args=(falldown_pipe_list[i][1], event_input_pipe,), name=f"Falldown_Process_{i}")
            process_list.append(falldown_process)
            falldown_process.start()

    if 'emotion' in args.modules:
        emotion_pipe_list = []
        for _ in range(scale_args.emotion):
            emotion_input_pipe, emotion_output_pipe = Pipe()
            emotion_pipe_list.append((emotion_input_pipe, emotion_output_pipe))
        for i in range(scale_args.emotion):
            emotion_process = Process(target=Emotion, args=(emotion_pipe_list[i][1], event_input_pipe,), name=f"Emotion_Process_{i}")
            process_list.append(emotion_process)
            emotion_process.start()

    if 'violence' in args.modules:
        violence_pipe_list = []
        for _ in range(scale_args.violence):
            violence_input_pipe, violence_output_pipe = Pipe()
            violence_pipe_list.append((violence_input_pipe, violence_output_pipe))
        for i in range(scale_args.violence):
            violence_process = Process(target=Violence, args=(violence_pipe_list[i][1], event_input_pipe,), name=f"Violence_Process_{i}")
            process_list.append(violence_process)
            violence_process.start()
    
    # 자세 추정 모델
    inferencer, init_args, call_args, display_alias = rtmo.get_model()
    # 얼굴 감지 모델 로드
    face_detector = face_detection.build_detector('RetinaNetResNet50', confidence_threshold=.5, nms_iou_threshold=.3)

    # 센서 관련 설정
    sensor = EdgeCam(debug_args.thermal_ip, debug_args.thermal_port, debug_args.rader_ip, debug_args.rader_port, debug_args=debug_args)
    cctv_info = sensor.get_cctv_info()
    cctv_source = cctv_info['source']
    
    # 동영상 관련 설정
    now = datetime.now()
    timestamp = str(now).replace(" ", "").replace(":", "-").replace(".", "-")
    cap = cv2.VideoCapture(cctv_source)
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    fps = 30
    num_frame = 0
    if cap.get(cv2.CAP_PROP_FPS):
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    tracker = BoTSORT(bot_sort_args, fps)

    if debug_args.debug == False: #TODO TEMP 해당 로직은 클래스 내부로 정리되어야함
        # 열화상 센서 연결
        if thermal_args.use_thermal and not thermal_args.use_reconnect:
            sensor.connect_thermal()
        # 레이더 센서 연결
        if rader_args.use_rader:
            sensor.connect_rader()

    # 디버그(시각화, 동영상 저장) 
    if debug_args.visualize:
        output_path = f"{debug_args.output}/{timestamp}"
        os.mkdir(output_path)
        filepath = debug_args.source
        filename = os.path.basename(filepath) 
        out = cv2.VideoWriter(os.path.join(output_path, filename + ".mp4"), fourcc, fps, (int(w), int(h)))

    # HAR 모듈 실행 대기
    def wait_subprocess_ready(name, pipe, logger):
        while True:
            logger.info(f'wating for {name} process to ready...')
            if pipe.recv():
                logger.info(f'{name} process ready')
                break
            else:
                time.sleep(0.1)
    if 'selfharm' in args.modules:
        for i in range(scale_args.selfharm):
            wait_subprocess_ready("Selfharm", selfharm_pipe_list[i][0], logger)
    if 'falldown' in args.modules:
        for i in range(scale_args.falldown):
            wait_subprocess_ready("Falldown", falldown_pipe_list[i][0], logger)
    if 'emotion' in args.modules:
        for i in range(scale_args.emotion):
            wait_subprocess_ready("Emotion", emotion_pipe_list[i][0], logger)
    if 'violence' in args.modules:
        for i in range(scale_args.violence):
            wait_subprocess_ready("Violence", violence_pipe_list[i][0], logger)

    # 종료 함수
    def shutdown():
        if 'selfharm' in args.modules:
            for p in selfharm_pipe_list:
                p[0].send("end_flag")
        if 'falldown' in args.modules:
            for p in falldown_pipe_list:
                p[0].send("end_flag")
        if 'emotion' in args.modules:
            for p in emotion_pipe_list:
                p[0].send("end_flag")
        if 'violence' in args.modules:
            for p in violence_pipe_list:
                p[0].send("end_flag")
        event_process.kill()
    atexit.register(shutdown)
    
    # 사람 감지 및 추적
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            v_frame = frame.copy()
            current_datetime = datetime.now()
            detections = []
            skeletons = []
            temp_call_args = copy.deepcopy(call_args)
            temp_call_args['inputs'] = frame         

            for _ in inferencer(**temp_call_args):
                pred = _['predictions'][0]
                l_p = len(pred)
                logger.info(f'frame #{num_frame} pose_results- {l_p} person detect!')
                pred.sort(key = lambda x: x['bbox'][0][0])

                for p in pred:
                    keypoints = p['keypoints']
                    keypoints_scores = p['keypoint_scores']
                    detection = [*p['bbox'][0], p['bbox_score']]  # TODO box score 계산 방식을 스켈레톤 포인트 중 제일 낮은 socre를 사용하도록 고치기                  
                    # 스켈레톤 포인트에 음수가 있는 경우 제외 TODO 탐지 결과 동일한지 확인 필요
                    conditions = [(x < 0 or y < 0) for x, y in keypoints]
                    from itertools import compress
                    invalid_skeletons = list(compress(keypoints, conditions))
                    if invalid_skeletons:
                        continue
                    detections.append(detection)
                    skeletons.append([a + [b] for a, b in zip(keypoints, keypoints_scores)])
            detections = np.array(detections, dtype=np.float32)
            skeletons = np.array(skeletons, dtype=np.float32)
            tracks = tracker.update(detections, skeletons, frame)
            if num_frame % fps == 0:
                face_detections = face_detector.detect(frame)

            meta_data = {'cctv_id': cctv_info['cctv_id'], 'current_datetime': current_datetime, 'fps': int(fps), 'timestamp': timestamp,
                         'cctv_name': cctv_info['cctv_name'], 'num_frame':num_frame, 'frame_size': (int(w), int(h))} 

            # if debug_args.visualize: # TODO 시각화 코드를 따로 빼내야함
            for i, track in enumerate(tracks):
                skeletons = track.skeletons
                detection = track.tlbr
                tid = track.track_id
                v_frame = draw_bbox_skeleton.draw(v_frame, tid, detection, skeletons[-1])
            meta_data['v_frame'] = v_frame

            combine_data = None # TODO 더미 데이터 넣을 것
            emotion_interval = fps * 3  # 따로 파라미터로 빼던가 해야함  TODO
            if debug_args.debug == False:  # 디버그에서도 지원하도록 해야함 TODO
                if num_frame % emotion_interval == 0:
                    combine_data, thermal_data, rader_data, overlay_image = sensor.get_data(frame, tracks, face_detections)
                    logger.info(combine_data)
                           
            # 모듈로 데이터 전송
            if 'selfharm' in args.modules and 0 < scale_args.selfharm:
                selfharm_pipe_list[num_frame % scale_args.selfharm][0].send([tracks, meta_data])
            if 'falldown' in args.modules and 0 < scale_args.falldown:
                falldown_pipe_list[num_frame % scale_args.falldown][0].send([tracks, meta_data])
            if num_frame % emotion_interval == 0:
                if 'emotion' in args.modules and 0 < scale_args.emotion:
                    emotion_pipe_list[num_frame % scale_args.emotion][0].send([tracks, meta_data, face_detections, frame, combine_data])
            if 'violence' in args.modules and 0 < scale_args.violence:
                violence_pipe_list[num_frame % scale_args.violence][0].send([tracks, meta_data])                            
            num_frame += 1
        else:
            break
    if debug_args.visualize:
        out.release()
    cap.release()

    if debug_args.debug == False:    #TODO 해당 로직은 클래스 내부에 있어야함
        if thermal_args.use_thermal and not thermal_args.use_reconnect:
            sensor.disconnect_thermal()
        if rader_args.use_rader:
            sensor.disconnect_rader()
    logger.warning("Main process end.")

if __name__ == '__main__':
    main()