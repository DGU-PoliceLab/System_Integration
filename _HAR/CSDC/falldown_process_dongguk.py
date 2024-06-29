import json
import numpy as np
import torch
from queue import Queue
from CSDC.ActionsEstLoader import TSSTG
import time
from multiprocessing import Process, Pipe

'''
frame : 원본 이미지
# det개수만큼 입력 들어옴
falldown_input : [[x1,y1,x2,y2,det_score,class,id],
                  [x1,y1,x2,y2,det_score,class,id],
                  [x1,y1,x2,y2,det_score,class,id]]
cctv_id : CCTV번호
'''

'''
falldown inference 에 들어오는 input
원본 이미지, [x1,y1,x2,y2,track id,det score], cctv번호 (=frame, falldown_input, cctv_id)
작동 순서
pose 모델 load
pose result 내에 run.py에서 얻어낸 track id를 추가
[bbox, 1, 0.5] -> (=bbox, cls, conf)
[bbox, 1, 0.6]
[bbox, 1, 0.7]
[bbox, 1, 0.7]
action 모델의 추론 조건 : [ 같은 ID의 13point의 관절점 좌표 15개를 모은 다음에 -> graph로 만들어서 ] -> action 모델에 넣어줌 -> normal / falldown 이냐를 판단한 결과를 return
action 모델 load
'''
###########################

class Falldown:
    def __init__(self):
        self.FALLDOWN_THRESHHOLD = 0.60
        self.FALLDOWN_FRAME_THRESHHOLD = 3
        self.FRAME_STEP = 14
        self.image_size = (1920, 1080)
        self.action_model = TSSTG()


    def check_falldown(self, action_name='Normal', confidence=0):
        if action_name == 'Fall Down' and FALLDOWN_THRESHHOLD < confidence:
            return True

        return False
    
    def track_process(self, track):
        if not track.is_confirmed():
            return None, None
        tid = track.track_id
        bbox = track.to_tlbr().astype(int)
        center = track.get_center().astype(int)
        pts = np.array(track.keypoints_list, dtype=np.float32)
        out = self.action_model.predict(pts, image_size)
        action_name = self.action_model.class_names[out[0].argmax()]
        confidence = out[0].max()

        return action_name, confidence
        

    def falldown_inference(self, falldown_queue, event_occurred_queue):
        
        event_occur_dict = {}

        while True:
            if not falldown_queue.empty():
                tracks, meta_data = falldown_queue.get()
                # tracks: tracker가 추적하는 데이터. HAR model의 input
                # meta_data: 현재 frame의 정보를 담고 있는 데이터. 카메라 정보, 해당 시점의 시간 정보 같은 HAR과 관련 없는 정보가 담김.
                
                # Predict Actions of each track.
                for track in tracks:
                    action_name, confidence = self.track_process(track)
                    if action_name is None:
                        continue
                    
                    if self.check_falldown(action_name, confidence):
                        tid = track.track_id  # Track ID
                        event_date = meta_data.get('event_date', 0)  # Placeholder for actual event date
                        event_time = meta_data.get('event_time', 0)  # Placeholder for actual event time
                        current_datetime = time.time()  # Current timestamp
                        event_occurred_queue.put(
                            (meta_data['cctv_id'], "selfharm", meta_data['cctv_name'], tid, event_date, event_time, "Insert as soon as", current_datetime, current_datetime)
                        )

            else:
                time.sleep()
                    
###########################




FALLDOWN_THRESHHOLD = 0.60
FALLDOWN_FRAME_THRESHHOLD = 3
FRAME_STEP = 14
image_size = (1920, 1080)


def check_falldown(action_name='Normal', confidence=0):
    if action_name == 'Fall Down' and FALLDOWN_THRESHHOLD < confidence:
        return True

    return False   

def falldown_inference(falldown_queue, event_occurred_queue):
    action_model_path = '/System_Integration/CSDC/models/tsstg-model-best-1.pth'
    action_model = TSSTG(action_model_path)

    event_occur_dict = {}

    while True:
        if not falldown_queue.empty():
            tracks, meta_data = falldown_queue.get()
            # tracks: tracker가 추적하는 데이터. HAR model의 input
            # meta_data: 현재 frame의 정보를 담고 있는 데이터. 카메라 정보, 해당 시점의 시간 정보 같은 HAR과 관련 없는 정보가 담김.
            
            # Predict Actions of each track.
            for i, track in enumerate(tracks):
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                bbox = track.to_tlbr().astype(int)
                center = track.get_center().astype(int)
                
                pts = np.array(track.keypoints_list, dtype=np.float32)
                out = action_model.predict(pts, image_size)
                action_name = action_model.class_names[out[0].argmax()]
                confidence = out[0].max()

        elif check_falldown(action_name=action_name, confidence=confidence):
            tid = 0
            event_date = 0
            event_time = 0
            current_datetime = 0
            current_datetime = 0
            event_occurred_queue.put((meta_data['cctv_id'], "selfharm", meta_data['cctv_name'], tid, event_date, event_time, "Insert as soon as", current_datetime, current_datetime))

        else:
            time.sleep()
            
