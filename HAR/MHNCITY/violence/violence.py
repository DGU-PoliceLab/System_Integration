import argparse
import sys
sys.path.insert(0, '/System_Integration/HAR/MHNCITY/violence')
from model import TemporalDynamicGCN, evaluate_frames
import numpy as np
import time
from Utils.logger import get_logger
from collections import deque

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_1', type=str, default='HAR/MHNCITY/models/model_epoch_270.pth', help='fight 1 model checkpoint path')
    parser.add_argument('--model_2', type=str, default='HAR/MHNCITY/models/model_epoch_397.pth', help='fight 1 model checkpoint path')
    parser.add_argument('--threshhold', type=float, default=0.57, help='Violence threshhold')
    parser.add_argument('--window_size', type=int, default=12, help='Window size')
    parser.add_argument('--num_frames', type=int, default=3, help='num frames')
    parser.add_argument('--num_persons', type=int, default=2, help='num persons')
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

def pad_keypoints(keypoints, num_persons, num_keypoints):
    # Pad with zeros if less than 'num_persons'
    if keypoints.shape[0] < num_persons:
        padding = ((0, num_persons - keypoints.shape[0]), (0, 0), (0, 0))
        #print ('padding', padding)
        keypoints = np.pad(keypoints, padding, mode='constant')
    return keypoints[:num_persons, :num_keypoints, :2]  # Select first 'num_persons' and 'num_keypoints'


def Violence(data_pipe, event_pipe):
    logger = get_logger(name="[MhnCity.Violence]", console=True, file=False)
    args = parse_args()
    fight_model_1 = TemporalDynamicGCN(window_size=args.window_size, num_frames=args.num_frames, num_persons=args.num_persons, num_keypoints=args.num_keypoints, num_features=2, num_classes=1, model_path=args.model_1)
    fight_model_1 = fight_model_1.to(args.device)
    fight_model_2 = TemporalDynamicGCN(window_size=args.window_size, num_frames=args.num_frames, num_persons=args.num_persons, num_keypoints=args.num_keypoints, num_features=2, num_classes=1, model_path=args.model_2)
    fight_model_2 = fight_model_2.to(args.device)
    all_batch_keypoints = []
    current_batch_keypoints = []
    current_frame_count = 0

    data_pipe.send(True)
    while True:
        try:
            data = data_pipe.recv()
            if data:
                if data == "end_flag":
                    break
                tracks, meta_data = data
                frame_skeletons = []
                current_frame_count += 1
                num_detected_people = len(tracks)
                for i, track in enumerate(tracks):
                    skeletons = track.skeletons
                    if len(skeletons) < args.window_size:
                        continue
                    tid = track.track_id
                    for i in range(len(skeletons)):
                        temp = skeletons[i]
                        temp = temp[:, :2]
                        frame_skeletons.append(temp)
                    
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
                    max_score_1, mean_score_1  = evaluate_frames(fight_model_1, all_batch_keypoints)
                    max_score_2, mean_score_2 = evaluate_frames(fight_model_2, all_batch_keypoints)
                    
                    mean_avg_score = adjust_mean(mean_score_1, mean_score_2)
                    logger.debug(f"mean_avg_score : {mean_avg_score}")           
                    logger.info(f"mean_avg_score : {mean_avg_score}")         
                    logger.info(f"mean_score_1 : {mean_score_1}") 
                    logger.info(f"mean_score_2 : {mean_score_2}") 
                    if mean_score_2 < 0.52:
                        print("OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
                        print("OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
                        print("OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
                        print("OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
                        print("OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
                    else:
                        print("________________________________________________________")
                        print("________________________________________________________")
                        print("________________________________________________________")
                        print("________________________________________________________")
                        print("________________________________________________________")
                        print("________________________________________________________")
                    if check_violence(confidence=mean_avg_score, threshhold=args.threshhold):
                        tid = 1
                        logger.info("action: violence")
                        event_pipe.send({'action': "violence", 'id':tid, 'cctv_id':meta_data['cctv_id'], 'current_datetime':meta_data['current_datetime'], 'location':meta_data['cctv_name'],
                                    'combine_data': None})
                                    
                    all_batch_keypoints = []        
            else:
                time.sleep(0.0001)
        except Exception as e:
            logger.error(f"Error occured in violence, {e}")
        