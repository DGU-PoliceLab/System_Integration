import argparse
import sys
sys.path.insert(0, '/System_Integration/_HAR/MHNCITY')
from model import TemporalDynamicGCN, evaluate_frames
import numpy as np
import time
from _Utils.logger import get_logger
from collections import deque

LOGGER = get_logger(name="[MhnCity.Violence]", console=False, file=False)
THRESHHOLD = 0.49
WINDOW_SIZE = 12

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_1', type=str, default='_HAR/MHNCITY/models/model_epoch_270.pth', help='fight 1 model checkpoint path')
    parser.add_argument('--model_2', type=str, default='_HAR/MHNCITY/models/model_epoch_397.pth', help='fight 1 model checkpoint path')
    parser.add_argument('--threshhold', type=float, default=0.49, help='Violence threshhold')
    parser.add_argument('--window_size', type=int, default=12, help='Window size')
    parser.add_argument('--num_frames', type=int, default=5, help='num frames')
    parser.add_argument('--num_persons', type=int, default=5, help='num persons')
    parser.add_argument('--num_keypoints', type=int, default=17, help='num keypoints')
    parser.add_argument('--device', type=str, default='cuda', help='CPU/CUDA device option')
    args = parser.parse_args()
    return args

def check_violence(confidence, threshhold):
    if threshhold < confidence:
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

def temp_preprocess(skeletons, window_size):
    skeletons = deque(skeletons, maxlen=window_size)

    for i, sk in enumerate(skeletons):
        if i == WINDOW_SIZE:
            break
        indices_14 = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]        
        skeletons[i] = sk[indices_14]

    return np.array(skeletons)

def Violence(data_pipe, event_pipe):
    args = parse_args()

    fight_model_1 = TemporalDynamicGCN(window_size=args.window_size, num_frames=args.num_frames, num_persons=args.num_persons, num_keypoints=args.num_keypoints, num_features=2, num_classes=1, model_path=args.model_1)
    fight_model_1 = fight_model_1.to(args.device)
    fight_model_2 = TemporalDynamicGCN(window_size=args.window_size, num_frames=args.num_frames, num_persons=args.num_persons, num_keypoints=args.num_keypoints, num_features=2, num_classes=1, model_path=args.model_2)
    fight_model_2 = fight_model_2.to(args.device)

    data_pipe.send(True)
    while True:
        data = data_pipe.recv()
        if data:
            tracks, meta_data = data
            for i, track in enumerate(tracks):
                skeletons = track.skeletons
                if len(skeletons) < args.window_size:
                    continue
                tid = track.track_id
                skeletons = temp_preprocess(skeletons=skeletons, window_size=args.window_size)
                max_score_1, mean_score_1  = evaluate_frames(fight_model_1, skeletons)
                max_score_2, mean_score_2 = evaluate_frames(fight_model_2, skeletons)
                # max_avg_score = adjust_max(max_score_1, max_score_2)
                mean_avg_score = adjust_mean(mean_score_1, mean_score_2)
                LOGGER.debug(f"mean_avg_score : {mean_avg_score}")           

                if check_violence(confidence=mean_avg_score, threshhold=args.threshhold):
                    tid = 1
                    LOGGER.info("action: violence")
                    event_pipe.send({'action': "violence", 'id':tid, 'cctv_id':meta_data['cctv_id'], 'current_datetime':meta_data['current_datetime']})
        else:
            time.sleep(0.0001)
