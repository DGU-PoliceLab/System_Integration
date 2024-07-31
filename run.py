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
import atexit
from Tracker.BoTSORT.tracker.bot_sort import BoTSORT
from Sensor.edgecam import EdgeCam
from Service.was import readActiveCctvList
from Utils.logger import get_logger
from Utils.head_bbox import *
from Utils.pipeline import *
import Utils.draw_bbox_skeleton as draw_bbox_skeleton 
import MOT.face_detection as face_detection
from HAR.PLASS.selfharm import Selfharm
from HAR.CSDC.falldown import Falldown
from HAR.HRI.emotion import Emotion
from HAR.MHNCITY.violence.violence import Violence
from Event.handler import update
from variable import get_root_args, get_sort_args, get_scale_args, get_debug_args, get_rader_args, get_thermal_args
import PoseEstimation.mmlab.rtmo as rtmo

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

    torch.multiprocessing.set_start_method('spawn')
        
    # 이벤트 처리를 위한 수집을 위한 파이프라인 생성
    event_input_pipe, event_output_pipe = Pipe()
    
    # 이벤트 프로세스
    event_process = Process(target=update, args=(event_output_pipe,))
    event_process.start()
    
    process_list = []
    # 자해 모듈 설정
    if 'selfharm' in args.modules:
        selfharm_pipe_list = []
        for _ in range(scale_args.selfharm):
            selfharm_input_pipe, selfharm_output_pipe = Pipe()
            selfharm_pipe_list.append((selfharm_input_pipe, selfharm_output_pipe))
        for i in range(scale_args.selfharm):
            selfharm_process = Process(target=Selfharm, args=(selfharm_pipe_list[i][1], event_input_pipe,), name=f"Selfharm_Process_{i}")
            process_list.append(selfharm_process)
            selfharm_process.start()

    # 낙상 모듈 설정
    if 'falldown' in args.modules:
        falldown_pipe_list = []
        for _ in range(scale_args.falldown):
            falldown_input_pipe, falldown_output_pipe = Pipe()
            falldown_pipe_list.append((falldown_input_pipe, falldown_output_pipe))
        for i in range(scale_args.falldown):
            falldown_process = Process(target=Falldown, args=(falldown_pipe_list[i][1], event_input_pipe,), name=f"Falldown_Process_{i}")
            process_list.append(falldown_process)
            falldown_process.start()

    # 감정 모듈 설정
    if 'emotion' in args.modules:
        emotion_pipe_list = []
        for _ in range(scale_args.emotion):
            emotion_input_pipe, emotion_output_pipe = Pipe()
            emotion_pipe_list.append((emotion_input_pipe, emotion_output_pipe))
        for i in range(scale_args.emotion):
            emotion_process = Process(target=Emotion, args=(emotion_pipe_list[i][1], event_input_pipe,), name=f"Emotion_Process_{i}")
            process_list.append(emotion_process)
            emotion_process.start()

    # 폭행 모듈 설정
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

    # CCTV 정보 받아오기
    cctv_data = readActiveCctvList(debug_args.debug)
    cctv_info = cctv_data[0]
    # logger 옵션 상관없이 출력
    print(f"cctv_info >>> {cctv_info}")

    # 센서 관련 설정
    sensor = EdgeCam(cctv_info['thermal_ip'], cctv_info['thermal_port'], cctv_info['rader_ip'], cctv_info['rader_port'], debug_args=debug_args)
    sensor.connect_rader()
    sensor.connect_thermal()

    # 동영상 관련 설정
    from datetime import datetime
    now = datetime.now()
    timestamp = str(now).replace(" ", "").replace(":", "-").replace(".", "-")
    cap = cv2.VideoCapture(cctv_info['cctv_ip'])
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    fps = 30
    num_frame = 0
    if cap.get(cv2.CAP_PROP_FPS):
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    tracker = BoTSORT(bot_sort_args, fps)

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
                    detection = [*p['bbox'][0], p['bbox_score']]        
                    
                    detections.append(detection)
                    skeletons.append([a + [b] for a, b in zip(keypoints, keypoints_scores)])
            detections = np.array(detections, dtype=np.float32)
            skeletons = np.array(skeletons, dtype=np.float32)
            tracks = tracker.update(detections, skeletons, frame)
            if num_frame % fps == 0:
                face_detections = face_detector.detect(frame)

            meta_data = {'cctv_id': cctv_info['cctv_id'],
                        'cctv_name': cctv_info['cctv_name'],
                        'cctv_ip': cctv_info['cctv_ip'],
                        'location_id': cctv_info['location_id'],
                        'location_name': cctv_info['location_name'],
                        'current_datetime': current_datetime,
                        'timestamp': timestamp,
                        'fps': int(fps),
                        'num_frame':num_frame,                       
                        'frame_size': (int(w), int(h))} 

            if debug_args.visualize:
                for i, track in enumerate(tracks):
                    skeletons = track.skeletons
                    detection = track.tlbr
                    tid = track.track_id
                    v_frame = draw_bbox_skeleton.draw(v_frame, tid, detection, skeletons[-1])
                meta_data['v_frame'] = v_frame

            combine_data = None
            emotion_interval = fps * 3
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
                    meta_data['frame']= frame
                    emotion_pipe_list[num_frame % scale_args.emotion][0].send([tracks, meta_data, face_detections, frame, combine_data])
            
            if num_frame % emotion_interval == 0:
                print(combine_data)
                if 'violence' in args.modules and 0 < scale_args.violence:
                    violence_pipe_list[num_frame % scale_args.violence][0].send([tracks, meta_data, combine_data])                            
            num_frame += 1
        else:
            break
    if debug_args.visualize:
        out.release()
    cap.release()
    sensor.disconnect_rader()
    sensor.disconnect_thermal()

    logger.warning("Main process end.")

if __name__ == '__main__':
    main()