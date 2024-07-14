import sys
sys.path.insert(0, "/System_Integration/")
sys.path.insert(0, "/System_Integration/_PoseEstimation/mmlab")
sys.path.insert(0, "/System_Integration/_Tracker")
sys.path.insert(0, "/System_Integration/_HAR")
sys.path.insert(0, "/System_Integration/_MOT")
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

from _Tracker.BoTSORT.tracker.bot_sort import BoTSORT
from EventHandler.EventHandler import EventHandler
from _Sensor.sensor import Sensor
# from Sensor.EdgeCam import EdgeCam, Radar, Thermal, Camera
from _Utils.logger import get_logger
from _Utils.head_bbox import *
from _Utils.pipeline import *
import _Utils.draw_bbox_skeleton as draw_bbox_skeleton 
import _Utils.draw_vital as draw_vital
import _MOT.face_detection as face_detection
from _HAR.PLASS.selfharm import Selfharm
from _HAR.CSDC.falldown import Falldown
from _HAR.HRI.emotion import Emotion
from _HAR.MHNCITY.violence.violence import Violence
from variable import get_root_args, get_sort_args, get_scale_args, get_debug_args
from rtmo import get_model

def main():
    # 출력 로그 설정
    logger = get_logger(name= '[RUN]', console= True, file= True)
    # 루트 인자 및 기타 인자 설정
    args = get_root_args()
    dict_args = vars(args)
    bot_sort_args = get_sort_args()
    bot_sort_args.ablation = False
    bot_sort_args.mot20 = not bot_sort_args.fuse_score
    debug_args = get_debug_args()
    scale_args = get_scale_args()

    def check_args():
        if debug_args.debug == False:
            logger.info("Unsupported arguments")
            logger.info("debug_args.debug == False")
            exit()
    check_args()

    torch.multiprocessing.set_start_method('spawn') # See "https://tutorials.pytorch.kr/intermediate/dist_tuto.html"
    event_input_pipe, event_output_pipe = Pipe()
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

    event_handler = EventHandler()   
    event_process = Process(target=event_handler.update, args=(event_output_pipe, ))
    event_process.start()

    # 디버그 모드
    if debug_args.debug == True:
        source = debug_args.source
        cctv_info = dict()
        cctv_info['id'] = debug_args.cctv_id
        cctv_info['ip'] = debug_args.cctv_ip
        cctv_info['name'] = debug_args.cctv_name
        thermal_info = dict()
        thermal_info['ip'] = debug_args.thermal_ip
        thermal_info['port'] = debug_args.thermal_port
        rader_data = None
        with open(debug_args.rader_data, 'r') as f:
            rader_data = json.load(f)
    else:
        # try:
        #     conn = connect_db("mysql-pls")
        #     if conn.open:
        #         if dict_args['video_file'] != "":
        #             cctv_info = get_cctv_info(conn)
        #     else:
        #         logger.warning('RUN-CCTV Database connection is not open.')
        #         cctv_info = {'cctv_id': 404}
        # except Exception as e:
        #     logger.warning(f'Unable to connect to database, error: {e}')
        #     cctv_info = {'cctv_id': 404}

        # cctv_info = cctv_info[1]
        # source = cctv_info['ip']
        # Process(target=object_snapshot_control, args=(object_snapshot_control_queue,)).start()
        pass # 서버 상의 DB 연동 코드를 봐야 알 수 있음

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

    sensor = Sensor(debug_args.thermal_ip, debug_args.thermal_port, debug_args.rader_ip, debug_args.rader_port)
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

    # _HAR 모듈 실행 대기
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
    def shut_down():
        print("ShutDown!")
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
    atexit.register(shut_down)
    
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

            meta_data = {'cctv_id': cctv_info['id'], 'current_datetime': current_datetime, 'fps': int(fps), 'timestamp': timestamp,
                         'cctv_name': cctv_info['name'], 'num_frame':num_frame, 'frame_size': (int(w), int(h))} 

            if debug_args.visualize:
                for i, track in enumerate(tracks):
                    skeletons = track.skeletons
                    detection = track.tlbr
                    tid = track.track_id
                    v_frame = draw_bbox_skeleton.draw(v_frame, tid, detection, skeletons[-1])
                meta_data['v_frame'] = v_frame
            
            # 모듈로 데이터 전송
            if 'selfharm' in args.modules and 0 < scale_args.selfharm:
                selfharm_pipe_list[num_frame % scale_args.selfharm][0].send([tracks, meta_data])
            if 'falldown' in args.modules and 0 < scale_args.falldown:
                falldown_pipe_list[num_frame % scale_args.falldown][0].send([tracks, meta_data])
            if num_frame % fps == 0:
                if 'emotion' in args.modules and 0 < scale_args.emotion:
                    emotion_pipe_list[num_frame % scale_args.emotion][0].send([tracks, meta_data, face_detections, frame])
            if 'violence' in args.modules and 0 < scale_args.violence:
                violence_pipe_list[num_frame % scale_args.violence][0].send([tracks, meta_data])

            combine_data, thermal_data, rader_data, overlay_image = sensor.get_data(frame, tracks, face_detections)
            logger.info(combine_data)
            if debug_args.visualize:
                red = (0, 0, 255)
                blue = (255, 0, 0)
                for td in thermal_data:
                    pos = td['pos']
                    cv2.line(draw_frame, (int(pos[0]), int(pos[1])), (int(pos[0]), int(pos[1])), red, 20)
                    cv2.putText(draw_frame, str(td['temp']), (int(pos[0]), int(pos[1]) + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, red, 2)

                for rd in rader_data:
                    pos = rd['pos']
                    cv2.line(draw_frame, (int(pos[0]), int(h/2)), (int(pos[0]), int(h/2)), blue, 20)
                    cv2.putText(draw_frame, str(rd['breath']), (int(pos[0]), int(h/2) + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, blue, 2)
                    cv2.putText(draw_frame, str(rd['heart']), (int(pos[0]), int(h/2) + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, blue, 2)

                cv2.imwrite("/System_Integration/_Output/overlay_image.png", overlay_image)
                out.write(draw_frame)
            num_frame += 1
        else:
            break
    if debug_args.visualize:
        out.release()
    cap.release()
    sensor.disconnect_rader()
    shut_down()

    logger.warning("Main process end.")

if __name__ == '__main__':
    main()