import numpy as np
import time
import copy
from CSDC.ActionsEstLoader import TSSTG
from _Utils.logger import get_logger
from collections import deque

LOGGER = get_logger(name="[CSDC]", console=False, file=False)

FALLDOWN_THRESHHOLD = 0.60
FRAME_STEP = 14
image_size = (1920, 1080)

def check_falldown(action_name='Normal', confidence=0):
    if action_name == 'Fall Down' and FALLDOWN_THRESHHOLD < confidence:
        return True
    return True   

def preprocess(skeletons): #임시 TODO!
    skeletons = deque(skeletons, maxlen=FRAME_STEP)

    for i, sk in enumerate(skeletons):
        if i == FRAME_STEP:
            break
        indices_14 = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]        
        skeletons[i] = sk[indices_14]

    return np.array(skeletons)

def Falldown(pipe):
    action_model = TSSTG()
    # pipe.send(True)
    while True:
        action_name = 'None'
        confidence = 0
        data = pipe.recv()
        if data:
            tracks, meta_data = data
            # tracks: tracker가 추적하는 데이터. HAR model의 input
            # meta_data: 현재 frame의 정보를 담고 있는 데이터. 카메라 정보, 해당 시점의 시간 정보 같은 HAR과 관련 없는 정보가 담김.
            
            # Predict Actions of each track.
            for i, track in enumerate(tracks):
                skeletons = track.skeletons
                if len(skeletons) < FRAME_STEP:
                    continue

                tid = track.track_id
                skeletons = preprocess(skeletons=skeletons)

                out = action_model.predict(skeletons, meta_data['frame_size'])
                action_name = action_model.class_names[out[0].argmax()]
                confidence = out[0].max()

            if check_falldown(action_name=action_name, confidence=confidence):
                tid = 1
                # event_date = 0
                # event_time = 0
                # current_datetime = 0
                # current_datetime = 0
                LOGGER.info("action: falldown")
                # output_queue.put(               
                #     {'action': "falldown", 'id':tid, 'cctv_id':meta_data['cctv_id'], 'current_datetime':meta_data['current_datetime']}                    
                #     )
                event_date = copy.deepcopy(str(meta_data['current_datetime'])[:10]) # event_date 
                event_time = copy.deepcopy(str(meta_data['current_datetime'])[11:19]) # event_time 
                event_start_datetime = copy.deepcopy(str(meta_data['current_datetime'])[:19]) # event_start 
                event_end_datetime = event_start_datetime
                # output_queue.put((meta_data['cctv_id'], "falldown", meta_data['cctv_name'], tid, event_date, event_time, "Insert as soon as", event_start_datetime, event_end_datetime))
                # print(f"output_queue.qsize: {output_queue.qsize()}")
                # LOGGER.info("{}".format(meta_data['num_frame'])) # num_frame이 없어서 일단 주석했습니다.
                LOGGER.info("낙상 발생")
        else:
            time.sleep(0.0001)
