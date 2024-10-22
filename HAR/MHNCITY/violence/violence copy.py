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

def check_violence(confidence, threshhold, num_detected_people):
    if threshhold < confidence and num_detected_people > 1:
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
    logger = get_logger(name="[MhnCity.Violence]", console=True, file=False)
    args = parse_args()
    from variable import get_debug_args
    debug_args = get_debug_args()

    seq=0
    ort_session_270 = onnxruntime.InferenceSession("/System_Integration/HAR/MHNCITY/violence/models/model_gcn270.onnx")
    ort_session_397 = onnxruntime.InferenceSession("/System_Integration/HAR/MHNCITY/violence/models/model_gcn397.onnx")
    # print(ort_session_270.get_inputs()[0].name)
    # print(ort_session.get_inputs())
    ort_270_inputs = ort_session_270.get_inputs()[0].name
    ort_270_outputs = ort_session_270.get_outputs()[0].name
    # print(ort_session_397.get_inputs()[0].name)
    # print(ort_session.get_inputs())
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
        # import time
        # start = time.time()
        data = data_pipe.recv()
        if data:
            if data == "end_flag":
                logger.warning("Violence process end.")
                if debug_args.visualize:    
                    visualizer.merge_img_to_video()
                break
            tracks, meta_data = data
            frame_skeletons = []
            total_skeletons = []
            current_frame_count += 1
            num_detected_people = len(tracks)

            action_name = 'normal' #TODO 2번 정의함
            confidence = 0          #TODO 2번 정의함
            if num_detected_people:
                for i, track in enumerate(tracks):
                    skeletons = track.skeletons[-1]

                    tid = track.track_id
                    skeletons[:, :2]
                    
                    frame_skeletons.append(np.array(skeletons[:, :2]))
                    total_skeletons.append(np.array(skeletons[:, :2]))
                    for _ in range(args.num_persons - num_detected_people):
                        frame_skeletons.append(np.zeros((args.num_keypoints, 2)))
                # print(f"현재 frame 스켈레톤 : {total_skeletons}")
                # print(f"인원수 대응 스켈레톤 길이 : {len(total_skeletons)}")
                # for i in range(num_detected_people):
                #     print(f"{i+1}번째 스켈레톤 : {total_skeletons[i]}")
                
                # if num_detected_people >= 2:
                #     # 최대 5명까지만 고려
                #     num_people = min(num_detected_people, 5)
                    
                #     # 모든 가능한 쌍의 거리를 저장할 리스트
                #     distance_list = []
                #     for i in range(num_people):
                #         for j in range(i+1, num_people):
                #             xi, yi = total_skeletons[i][15], total_skeletons[i][16]
                #             xj, yj = total_skeletons[j][15], total_skeletons[j][16]
                            
                #             distance = (xj - xi)**2 + (yj - yi)**2
                #             distance_list.append(distance)

                    # 거리 출력
                    # for dist in distance_list:
                    #     logger.info(f"거리 = {dist}, 감지 인원수 : {num_people}")
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
                # 피클따기
                # with open("policelab_violence_data.pickle", "wb") as fw:
                #             pickle.dump(all_batch_keypoints, fw)
                seq+=1

                all_batch_keypoints = np.array(all_batch_keypoints, dtype=np.float32)
                # ort_270outs = ort_session_270.run([ort_270_outputs], {ort_270_inputs : all_batch_keypoints})
                # print("---------------------------------------------")
                # print(f"현재 seqence : {seq}")
                # print(f"seqence {seq} --- 270 out MAX")
                # print(f"ort_270outs : {ort_270outs}")
                # print(f"270 : {round(np.max(ort_270outs), 9)}")
                
                ort_397outs = ort_session_397.run([ort_397_outputs], {ort_397_inputs : all_batch_keypoints})
                # print("397 out")
                # print(f"ort_397outs : {ort_397outs}")
                # print(f"seqence {seq} --- 397 out MAX")
                # print(f"397 : {round(np.max(ort_397outs), 9)}")

                # print("---------------------------------------------")
                # 여기가 폭행 모델의 전처리된 최종 input 값이 들어가는 곳 (all_batch_keypoints)
                # max_score_1 = np.max(ort_270outs)
                # max_score_2 = np.max(ort_397outs) 
                # mean_score_1 = np.mean(ort_270outs)
                mean_score_2 = np.mean(ort_397outs) # 이 값을 사용해야 됨.
                # mean_avg_score = adjust_mean(max_score_1, max_score_2)
                # logger.info(f"mean_avg_score : {max_score_2}")

                confidence = mean_score_2
                if check_violence(confidence=mean_score_2, threshhold=args.threshhold, num_detected_people=num_detected_people):
                    action_name = 'violence'
                    tid = 1
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
            # logger.info(f"time : {round(time.time() - start, 6)}")
        else:
            time.sleep(0.0001)
