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

from EventHandler.EventHandler import EventHandler
from Sensor.EdgeCam import EdgeCam, Radar, Thermal, Camera

from _Tracker.BoTSORT.tracker.bot_sort import BoTSORT
from _Utils.logger import get_logger
from _Utils.head_bbox import *
import _Utils.draw_bbox_skeleton as draw_bbox_skeleton 
import _MOT.face_detection as face_detection
from _HAR.PLASS.selfharm import Selfharm
from _HAR.CSDC.falldown import Falldown
from _HAR.HRI.emotion import Emotion
from _HAR.MHNCITY.violence.violence import Violence

from variable import get_root_args, get_sort_args, get_scale_args, get_debug_args
import rtmo

logger = get_logger(name= '[RUN]', console= True, file= True)

def main():
    #region init() 
    args = get_root_args() # args['root'] 형태가 더 바람직함 #TODO 
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
        pass

    # 자세 추정 모델
    inferencer, init_args, call_args, display_alias = rtmo.get_model()
    # 얼굴 감지 모델
    face_detector = face_detection.build_detector('RetinaNetResNet50', confidence_threshold=.5, nms_iou_threshold=.3)

    # 동영상 관련 설정
    now = datetime.now()
    timestamp = str(now).replace(" ", "").replace(":", "-").replace(".", "-")
    cap = cv2.VideoCapture(source)
    fourcc = cv2.VideoWriter_fourcc('M','P','4','V')
    fps = None
    num_frame = 0

    if cap.get(cv2.CAP_PROP_FPS):
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    tracker = BoTSORT(bot_sort_args, fps)

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

    def shutdown():
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
    atexit.register(shutdown)
    #endregion init()

    #region update()
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
            
            meta_data = {'cctv_id': cctv_info['id'], 'current_datetime': current_datetime, 'fps': int(fps), 'timestamp': timestamp,
                         'cctv_name': cctv_info['name'], 'num_frame':num_frame, 'frame_size': (int(w), int(h))} 
            
            #'timestamp': timestamp 용도 상, 'dir_name' 혹은 'path'를 전달하는게 맞음. timestamp가 내부적으로 왜 필요한지 유추가 어려우니까.
            
            if debug_args.visualize:
                for i, track in enumerate(tracks):
                    skeletons = track.skeletons
                    detection = track.tlbr
                    tid = track.track_id
                    v_frame = draw_bbox_skeleton.draw(v_frame, tid, detection, skeletons[-1])
                meta_data['v_frame'] = v_frame
            
            if 'selfharm' in args.modules and 0 < scale_args.selfharm:
                selfharm_pipe_list[num_frame % scale_args.selfharm][0].send([tracks, meta_data])
            if 'falldown' in args.modules and 0 < scale_args.falldown:
                falldown_pipe_list[num_frame % scale_args.falldown][0].send([tracks, meta_data])
            if num_frame % fps == 0:
                if 'emotion' in args.modules and 0 < scale_args.emotion:
                    emotion_pipe_list[num_frame % scale_args.emotion][0].send([tracks, meta_data, face_detections, frame])
            if 'violence' in args.modules and 0 < scale_args.violence:
                violence_pipe_list[num_frame % scale_args.violence][0].send([tracks, meta_data])

            if debug_args.visualize:
                out.write(v_frame)

            num_frame += 1
        else:
            break
    #endregion update()

    #region shutdown()
    cap.release()
    if debug_args.visualize:
        out.release()
    logger.warning("Main process end.")
    #shutdown()
    #endregion shutdown()
    
if __name__ == '__main__':
    main()

    #TODO 'falldown' 같은 action 이름은 따로 관리되어야함, 잠재적으로 버그가 생길 여지가 많음"
    #"end_flag"