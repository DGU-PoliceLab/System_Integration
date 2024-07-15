import sys
sys.path.insert(0, "/workspace/policelab-git/System_Integration/")
sys.path.insert(0, "/workspace/policelab-git/System_Integration/PoseEstimation/mmlab")
sys.path.insert(0, "/workspace/policelab-git/System_Integration/Tracker")
sys.path.insert(0, "/workspace/policelab-git/System_Integration/HAR")
sys.path.insert(0, "/workspace/policelab-git/System_Integration/MOT")
import os
import signal
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
# from EventHandler.EventHandler import EventHandler
from Sensor.sensor import Sensor
# from Sensor.edge_cam import EdgeCam, Radar, Thermal, Camera
from Utils.logger import get_logger
from Utils.head_bbox import *
from Utils.pipeline import *
import Utils.draw_bbox_skeleton as draw_bbox_skeleton 
import Utils.draw_vital as draw_vital
import MOT.face_detection as face_detection
from HAR.PLASS.selfharm import Selfharm
from HAR.CSDC.falldown import Falldown
from HAR.HRI.emotion import Emotion
from HAR.MHNCITY.violence.violence import Violence
from variable import get_root_args, get_sort_args, get_scale_args, get_debug_args, get_rader_args, get_thermal_args
from rtmo import get_model

from DB.db_controller import connect_db, insert_event, insert_realtime
from DB.mq_controller import connect_mq
from DB.event_controller import collect_evnet
from DB.snapshot_controller import object_snapshot_control

def main():
    # 출력 로그 설정
    logger = get_logger(name= '[RUN]', console= True, file= False)
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
        # if debug_args.debug == False:
        #     logger.info("Unsupported arguments")
        #     logger.info("debug_args.debug == False")
        #     exit()
        pass
    check_args()

    torch.multiprocessing.set_start_method('spawn') # See "https://tutorials.pytorch.kr/intermediate/dist_tuto.html"
    
    # 이벤트 처리를 위한 수집을 위한 파이프라인 생성
    event_input_pipe, event_output_pipe = Pipe()
    # 이벤트 프로세스
    event_process = Process(target=collect_evnet, args=(event_output_pipe,))
    event_process.start()


    # 센서 데이터 + 감정 = 실시간 객체 정보 수집을 위한 파이프라인 생성
    # realtime_sensor_input_pipe, realtime_sensor_output_pipe = Pipe()
    # 센서 데이터 + 감정 = 실시간 객체 정보 수집을 위한 프로세스
    # realtime_sensor_process = Process(target=collect_realtime, args=(realtime_sensor_output_pipe,))
    # realtime_sensor_process.start()

    # matched_data = (emotion_data['cctv_id'], emotion_data['id'], radar_data[1], radar_data[2], thermal['temp'], emotion_data['mapped_emotion_results'][0], radar_data[3])
    # realtime_queue.put(matched_data)
    # insert_realtime_thread = Thread(target=insert_realtime, args=(realtime_status_queue, realtime_status_conn),daemon=False).start()
    
      
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

    # 디버그 모드
    if debug_args.debug == True:
        source = debug_args.source
        cctv_info = dict()
        cctv_info['cctv_id'] = debug_args.cctv_id
        cctv_info['ip'] = debug_args.cctv_ip
        cctv_info['cctv_name'] = debug_args.cctv_name
        thermal_info = dict()
        thermal_info['ip'] = debug_args.thermal_ip
        thermal_info['port'] = debug_args.thermal_port
        rader_data = None
        with open(debug_args.rader_data, 'r') as f:
            rader_data = json.load(f)
        pass
    else:
        try:
            conn = connect_db("mysql-pls")
            if conn.open:
                if dict_args['video_file'] != "":
                    cctv_info = get_cctv_info(conn)
                    cctv_info = cctv_info[1]
                    source = cctv_info['cctv_ip']
            else:
                logger.warning('RUN-CCTV Database connection is not open.')
                cctv_info = {'cctv_id': 404}
        except Exception as e:
            logger.warning(f'Unable to connect to database, error: {e}')
            cctv_info = {'cctv_id': 404}

    # 사람 감지 및 추적을 위한 모델 로드
    inferencer, init_args, call_args, display_alias = get_model()
    # 얼굴 감지 모델 로드
    face_detector = face_detection.build_detector('RetinaNetResNet50', confidence_threshold=.5, nms_iou_threshold=.3)

    # 동영상 관련 설정
    now = datetime.now()
    timestamp = str(now).replace(" ", "").replace(":", "-").replace(".", "-")
    cap = cv2.VideoCapture(source)
    fourcc = cv2.VideoWriter_fourcc('M','P','4','V')
    fps = 30
    num_frame = 0
    if cap.get(cv2.CAP_PROP_FPS):
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    tracker = BoTSORT(bot_sort_args, fps)

    # 센서 관련 설정
    sensor = Sensor(debug_args.thermal_ip, debug_args.thermal_port, debug_args.rader_ip, debug_args.rader_port)
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

    def wait_subprocess_ready(name, pipe, logger):
        while True:
            logger.info(f'wating for {name} process to ready...')
            if pipe.recv():
                logger.info(f'{name} process ready')
                break
            else:
                time.sleep(0.1)

    # HAR 모듈 실행 대기
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
    def shut_down(sig=None, frame=None):
        logger.warning("Terminated by user input.")
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
        os._exit()

    # atexit.register(shut_down)
    signal.signal(signal.SIGINT, shut_down)
    
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

            meta_data = {'cctv_id': cctv_info['cctv_id'], 'current_datetime': current_datetime, 'fps': int(fps), 'timestamp': timestamp,
                         'cctv_name': cctv_info['cctv_name'], 'num_frame':num_frame, 'frame_size': (int(w), int(h))} 

            # if debug_args.visualize:
            for i, track in enumerate(tracks):
                skeletons = track.skeletons
                detection = track.tlbr
                tid = track.track_id
                v_frame = draw_bbox_skeleton.draw(v_frame, tid, detection, skeletons[-1])
            meta_data['v_frame'] = v_frame
 
            if num_frame % fps == 0:
                combine_data, thermal_data, rader_data, overlay_image = sensor.get_data(frame, tracks, face_detections)
                logger.info(combine_data)
                           
            # 모듈로 데이터 전송
            if 'selfharm' in args.modules and 0 < scale_args.selfharm:
                selfharm_pipe_list[num_frame % scale_args.selfharm][0].send([tracks, meta_data])
            if 'falldown' in args.modules and 0 < scale_args.falldown:
                falldown_pipe_list[num_frame % scale_args.falldown][0].send([tracks, meta_data])
            if num_frame % fps == 0:
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
    if thermal_args.use_thermal and not thermal_args.use_reconnect:
        sensor.disconnect_thermal()
    if rader_args.use_rader:
        sensor.disconnect_rader()
    shut_down()

    logger.warning("Main process end.")

if __name__ == '__main__':
    main()