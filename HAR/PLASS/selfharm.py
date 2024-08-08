import sys
sys.path.insert(0, '/System_Integration/HAR/PLASS')
import time
import numpy as np
import mmengine
from threading import Thread
from multiprocessing import Process, Pipe
from mmaction.apis import inference_skeleton, init_recognizer
from Utils.logger import get_logger
from Utils._visualize import Visualizer
from variable import get_selfharm_args, get_debug_args

# 함수 또는 쓰레드에서 결과값을 가져오기 위한 전역변수
EVENT = ["normal", 0.0]

# tracks 데이터를 모델에서 활용가능한 형태로 전처리
def pre_processing(tracks):
    # 모델에서 사용할 형태
    form = {
        'tid': None,
        'bbox_scores': [],
        'bboxes': [],
        'keypoints_visible': [],
        'keypoint_scores': [],
        'keypoints': []
    }
    # 전처리 과정
    for track in tracks:
        form['tid'] = track.track_id
        bbox_score = track.score
        form['bbox_scores'].append(bbox_score)
        bbox = track.tlbr
        form['bboxes'].append(bbox)
        skeleton = track.skeletons[0]
        temp_keypoints = []
        temp_score = []
        for keypoint in skeleton:
            temp_keypoints.append([keypoint[0], keypoint[1]])
            temp_score.append(keypoint[2])
        form['keypoints_visible'].append(temp_score)
        form['keypoint_scores'].append(temp_score)
        form['keypoints'].append(temp_keypoints)

    # 모델에서 사용할 형태에 전처리된 데이터 추가
    form['bbox_scores'] = np.array(form['bbox_scores'], dtype=np.float32)
    form['bboxes'] = np.array(form['bboxes'], dtype=np.float32)
    form['keypoints_visible'] = np.array(form['keypoints_visible'], dtype=np.float32)
    form['keypoint_scores'] = np.array(form['keypoint_scores'], dtype=np.float32)
    form['keypoints'] = np.array(form['keypoints'], dtype=np.float32)
    # 전처리된 데이터 반환
    return form

def inference(model, label_map, pose_data, meta_data, pipe, logger):
    global EVENT
    try:
        # tid값 받아오기
        tid = pose_data[-1]['tid']
        # 데이터를 모델에 넣어 행동 결과값 받아오기
        result = inference_skeleton(model, pose_data, (meta_data[-1]['frame_size']))
        # 행동 결과값 중 가장 높은 예측값을 가진 행동 가져오기
        max_pred_index = result.pred_score.argmax().item()
        # 숫자로된 행동 결과값을 행동 라벨 이름으로 매칭하기
        action_label = label_map[max_pred_index]
        # 가장 높은 예측값을 가진 행동의 예측값 가저오기
        confidence = result.pred_score[max_pred_index]
        logger.debug(f"action: {action_label}, confidence: {confidence}")
        if action_label in ["scratching_arm", "biting"]:
            action_label = "normal"
        # 행동 라벨이 selfharm(normal이 아닌값)의 예측값이 0.85 초과거나 normal의 예측갑이 0.3미만일 때 selfharm으로 처리
        if (action_label != 'normal' and action_label != "hittingbody" and confidence > 0.95) or (action_label == 'normal' and confidence < 0.1):
            if (action_label == 'choking_hand' or action_label == 'choking_cloth') and confidence > 0.95:
                logger.info(f"action: selfharm, confidence: {confidence}")
                pipe.send({'action': "selfharm", 'id':tid, 'meta_data': meta_data[-1]}) # TODO meta_data가 리스트임. 수정요망
            elif (action_label == 'normal' and confidence < 0.1):
                logger.info(f"action: selfharm, confidence: {confidence}")
                pipe.send({'action': "selfharm", 'id':tid, 'meta_data': meta_data[-1]}) # TODO meta_data가 리스트임. 수정요망
            elif (confidence > 0.985):
                logger.info(f"action: selfharm, confidence: {confidence}")
                pipe.send({'action': "selfharm", 'id':tid, 'meta_data': meta_data[-1]}) # TODO meta_data가 리스트임. 수정요망

        EVENT = [action_label, confidence]
    except Exception as e:
        logger.error(f'Error occured in inference_thread, error: {e}')

def Selfharm(data_pipe, event_pipe):
    global EVENT
    logger = get_logger(name="[PLASS]", console=True, file=False)
    args = get_selfharm_args()
    debug_args = get_debug_args()
    if debug_args.visualize:
        visualizer = Visualizer("selfharm")
    # 모델 설정 불러오기
    config = mmengine.Config.fromfile(args.config)
    # 모델 불러오기
    model = init_recognizer(config, args.checkpoint, args.device)
    # 라벨맵 불러오기
    label_map = [x.strip() for x in open(args.label_map).readlines()]

    # 정확한 행동 추출을 위한 프레임을 모으는 배열
    pose_array = []
    meta_array = []
    prev_data = None
    
    data_pipe.send(True)
    try:
        while True:
            try:
            # 데이터가 step_size이상 모였을 때, 행동 추론
                if len(pose_array) > args.step_size:
                    # 사용할 데이터(step_size 만큼) 가져오기
                    pose_data = pose_array[:args.step_size]
                    meta_data = meta_array[:args.step_size]
                    # 가져온 데이터를 배열에서 빼주기
                    pose_array = pose_array[args.step_size:]
                    meta_array = meta_array[args.step_size:]
                    # 쓰레드를 사용하여 추론할지 결정
                    if args.thread_mode:
                        # 추론 쓰레드 생성
                        infrence_thread = Thread(
                            target=inference, 
                            args=(model, label_map, pose_data, meta_data, event_pipe, logger))
                        # 추론 시작
                        infrence_thread.start()
                    else:
                        # 추론 시작
                        inference(model, label_map, pose_data, meta_data, event_pipe, logger)
                # 실시간 데이터 받아오기
                data = data_pipe.recv()
                if data and data != prev_data:
                    # 종료 플래그일 경우 프로세스 종료 루틴
                    if data == "end_flag":
                        logger.warning("Selfharm process end.")
                        if debug_args.visualize:    
                            visualizer.merge_img_to_video()
                        break
                    tracks, meta_data = data
                    # 이전 데이터 갱신
                    prev_data = data
                    # 새로운 데이터 전처리
                    pose_data = pre_processing(tracks)


                    # 사람이 있는 경우 데이터 모으기
                    if len(tracks) > 0:
                        pose_array.append(pose_data)
                        meta_array.append(meta_data)
                    # 시각화
                    if debug_args.visualize:
                        visualizer.mkdir(meta_data['timestamp'])
                        visualizer.save_temp_image([meta_data["v_frame"], EVENT[0], EVENT[1]], meta_data["num_frame"])
                else:
                    time.sleep(0.0001)
            except:
                time.sleep(0.0001)
    except Exception as e:
        logger.error(f"Error occured in selfharm, {e}")
        exit()
