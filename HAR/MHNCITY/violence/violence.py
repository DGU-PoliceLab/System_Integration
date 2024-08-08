import argparse
import sys
sys.path.insert(0, '/System_Integration/HAR/MHNCITY/violence')
from model import TemporalDynamicGCN, evaluate_frames
import numpy as np
import time
from Utils.logger import get_logger
from collections import deque
import pickle
import onnx
import onnxruntime

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_1', type=str, default='HAR/MHNCITY/models/model_epoch_270.pth', help='fight 1 model checkpoint path')
    parser.add_argument('--model_2', type=str, default='HAR/MHNCITY/models/model_epoch_397.pth', help='fight 1 model checkpoint path')
    parser.add_argument('--threshhold', type=float, default=0.65, help='Violence threshhold')
    parser.add_argument('--window_size', type=int, default=12, help='Window size')
    parser.add_argument('--num_frames', type=int, default=5, help='num frames')
    parser.add_argument('--num_persons', type=int, default=5, help='num persons')
    parser.add_argument('--num_keypoints', type=int, default=17, help='num keypoints')
    parser.add_argument('--device', type=str, default='cuda', help='CPU/CUDA device option')
    args = parser.parse_args()
    return args

MAX_LEN = 300
N = 10
THRESHOLD = 0.2
event_deque = deque(maxlen=MAX_LEN)
def check_violence(confidence=0, num_detected_people=0):
    if num_detected_people < 2:
        return False
    
    if 0.95 < confidence:
        event_deque.append('normal')
    elif THRESHOLD < confidence:
        event_deque.append('violence')
    else:
        event_deque.append('normal')

    conter = 0
    for evt in event_deque:
        if evt == "violence":
            conter += 1    
    if N <= conter:
        event_deque.clear()
        return True
    return False

def adjust_max(max_score_1, max_score_2):
    score_gap = abs(max_score_1 - max_score_2)
    score_1 = max_score_1 * 0.6 if max_score_1 > 0.7 and score_gap > 0.4 else max_score_1
    score_2 = max_score_2 * 0.6 if max_score_2 > 0.7 and score_gap > 0.4 else max_score_2
    max_avg_score = (score_1 + score_2) / 2
    return max_avg_score

def adjust_mean(mean_score_1, mean_score_2):
    score_gap = abs(mean_score_1 - mean_score_2)
    score_1 = mean_score_1 * 0.6 if mean_score_1 > 0.7 and score_gap > 0.4 else mean_score_1
    score_2 = mean_score_2 * 0.6 if mean_score_2 > 0.7 and score_gap > 0.4 else mean_score_2
    mean_avg_score = (score_1 + score_2) / 2
    return mean_avg_score

def pad_keypoints(keypoints, num_persons, num_keypoints):
    # Pad with zeros if less than 'num_persons'
    if keypoints.shape[0] < num_persons:
        padding = ((0, num_persons - keypoints.shape[0]), (0, 0), (0, 0))
        #print ('padding', padding)
        keypoints = np.pad(keypoints, padding, mode='constant')
    return keypoints[:num_persons, :num_keypoints, :2]  # Select first 'num_persons' and 'num_keypoints'

def Violence(data_pipe, event_pipe):
    logger = get_logger(name="[MhnCity.Violence]", console=False, file=False)
    args = parse_args()
    from variable import get_debug_args
    debug_args = get_debug_args()

    seq=0
    ort_session_270 = onnxruntime.InferenceSession("/System_Integration/HAR/MHNCITY/violence/models/model_gcn270.onnx")
    ort_session_397 = onnxruntime.InferenceSession("/System_Integration/HAR/MHNCITY/violence/models/model_gcn397.onnx")
    ort_270_inputs = ort_session_270.get_inputs()[0].name
    ort_270_outputs = ort_session_270.get_outputs()[0].name
    ort_397_inputs = ort_session_397.get_inputs()[0].name
    ort_397_outputs = ort_session_397.get_outputs()[0].name

    all_batch_keypoints = []
    current_batch_keypoints = []
    distances = []
    current_frame_count = 0
    data_pipe.send(True)

    from Utils._visualize import Visualizer
    if debug_args.visualize:
            visualizer = Visualizer("violence")
            init_flag = True

    while True:
        try:
            data = data_pipe.recv()
            if data:
                if data == "end_flag":
                    logger.warning("Violence process end.")
                    if debug_args.visualize:    
                        visualizer.merge_img_to_video()
                    break
                tracks, meta_data = data
                frame_skeletons = []
                current_frame_count += 1
                num_detected_people = len(tracks)

                action_name = 'normal'
                confidence = 0
                if num_detected_people:
                    for i, track in enumerate(tracks):
                        skeletons = track.skeletons[-1]

                        tid = track.track_id
                        skeletons[:, :2]
                        
                        frame_skeletons.append(np.array(skeletons[:, :2]))
                        for _ in range(args.num_persons - num_detected_people):
                            frame_skeletons.append(np.zeros((args.num_keypoints, 2)))
                else:
                    # No person detected, pad with zeros for all
                    for _ in range(args.num_persons):
                        frame_skeletons.append(np.zeros((args.num_keypoints, 2)))
                
                frame_skeletons = np.array(frame_skeletons)
                
                padded_keypoints = pad_keypoints(frame_skeletons, args.num_persons, args.num_keypoints)  # args.num_persons, args.num_keypoints로 수정
                current_batch_keypoints.append(padded_keypoints)
                if current_frame_count == args.num_frames:  
                    all_batch_keypoints.append(current_batch_keypoints)             
                    
                    current_batch_keypoints = []
                    current_frame_count = 0
                if len(all_batch_keypoints) == args.window_size:
                    seq+=1
                    all_batch_keypoints = np.array(all_batch_keypoints, dtype=np.float32)                
                    ort_397outs = ort_session_397.run([ort_397_outputs], {ort_397_inputs : all_batch_keypoints})
                    mean_score_2 = np.mean(ort_397outs) # 이 값을 사용해야 됨.
                    confidence = mean_score_2

                    logger.info(f"action: {action_name} {meta_data['num_frame']}, 폭행 확률 : {mean_score_2}, 사람 수 : {num_detected_people}")

                    if check_violence(confidence=mean_score_2, num_detected_people=num_detected_people):
                        action_name = 'violence'
                        logger.info(f"action: violence {meta_data['num_frame']}, 폭행 확률 : {mean_score_2}, 사람 수 : {num_detected_people}")
                        event_pipe.send({'action': "violence", 'id':tid, 'meta_data': meta_data})            
                    all_batch_keypoints = []

                if debug_args.visualize:
                    if init_flag == True:
                        visualizer.mkdir(meta_data['timestamp'])
                        init_flag = False
                    if action_name == 'violence':
                        visualizer.save_temp_image([meta_data["v_frame"], action_name, confidence], meta_data["num_frame"], color="red")
                    else:
                        visualizer.save_temp_image([meta_data["v_frame"], action_name, confidence], meta_data["num_frame"])
            else:
                time.sleep(0.0001)
        except:
            print("violence.py pipe 오류 발생")
            time.sleep(0.0001)