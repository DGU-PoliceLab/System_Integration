import time
import numpy as np
from collections import deque
from HAR.CSDC.module.Loader import TSSTG
from Utils.logger import get_logger
from Utils._visualize import Visualizer
from variable import get_falldown_args, get_debug_args
logger = get_logger(name="[CSDC]", console=True, file=True)

def preprocess(skeletons, frame_step):
    skeletons = deque(skeletons, maxlen=frame_step)
    for i, sk in enumerate(skeletons):
        if i == frame_step:
            break
        indices_14 = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]        
        skeletons[i] = sk[indices_14]

    return np.array(skeletons)

MAX_LEN = 24
N = 14
THRESHOLD = 0.70
event_deque = deque(maxlen=MAX_LEN)
def check_event(action_name='Normal', confidence=0, threshold=0.6):
    
    if action_name == 'Fall Down' and 0.88 < confidence:
        event_deque.append('Normal')
    elif action_name == 'Fall Down' and THRESHOLD < confidence:
        event_deque.append('Fall Down')
    else:
        event_deque.append('Normal')

    fall_conter = 0
    for evt in event_deque:
        if evt == "Fall Down":
            fall_conter += 1
    
    if N <= fall_conter:
        event_deque.clear()
        return True
    return False   


is_longterm_check = False
MAX_PERSON = 10
FPS = 30
hold_frames = [[] for x in range(MAX_PERSON)]
count = [0 for x in range(MAX_PERSON)]
LONGTERM_THRESHOLD = 1500.0
HOLD_TIME = 10

def check_longterm(tracks, meta_data):
    event = ['normal', 1.0]
    tids = []
    global is_longterm_check
    for track in tracks:
        tid = track.track_id
        tids.append(tid)
        skeleton = track.skeletons[0]
        hold_frames[tid % MAX_PERSON].append(skeleton)
        if len(hold_frames[tid % MAX_PERSON]) > FPS * HOLD_TIME:
            hold_frames[tid % MAX_PERSON].pop(0)
    for i in range(MAX_PERSON):
        if i not in tids:
            hold_frames[i] = []
            count[i] = 0
    for i, hold in enumerate(hold_frames):
        if len(hold) >= FPS * HOLD_TIME:
            cur = None
            similarity = 0
            for skeletons in hold:
                if cur is None:
                    cur = skeletons
                else:
                    simil = np.linalg.norm(cur - skeletons)
                    similarity += simil
            confidence = similarity/(FPS * HOLD_TIME)
            if confidence < LONGTERM_THRESHOLD:
                count[i] += 1
                event[0] = 'normal'
                event[1] = str(count)
            else:
                count[i] = 0
                event[0] = 'longterm(counting)'
                event[1] = str(count)
    for i, c in enumerate(count):
        if c > HOLD_TIME * FPS:
            event[0] = 'longterm(detect)'
            count[i] = 0
            is_longterm_check = False #롱텀 체크 종료 조건이 롱텀 발생밖에 없음. TODO 버그
            return True
    return False

def Falldown(data_pipe, event_pipe):
    args = get_falldown_args()
    debug_args = get_debug_args()
    if debug_args.visualize:
        visualizer = Visualizer("falldown")
        init_flag = True
    action_model = TSSTG()
    data_pipe.send(True)
    
    global is_longterm_check #나쁜구조 TODO
    while True:
        action_name = 'None'
        confidence = 0
        data = data_pipe.recv()
        if data:
            if data == "end_flag":
                logger.warning("Falldown process end.")
                if debug_args.visualize:    
                    visualizer.merge_img_to_video()
                break
            tracks, meta_data = data

            for i, track in enumerate(tracks):
                skeletons = track.skeletons
                if len(skeletons) < args.frame_step:
                    continue

                tid = track.track_id
                skeletons = preprocess(skeletons=skeletons, frame_step=args.frame_step)

                out = action_model.predict(skeletons, meta_data['frame_size'])
                action_name = action_model.class_names[out[0].argmax()]
                confidence = out[0][1]

                if action_name == "Fall Down":
                    break

            # TODO 영상에서 낙상이 있으면 작동되게 구현되어 있음. 롱텀 제대로 구현하려면 낙상 이벤트가 id별로 관리되야함.
            if check_event(action_name=action_name, confidence=confidence, threshold=args.threshhold):
                event_pipe.send({'action': "falldown", 'id':tid, 'meta_data': meta_data})
                logger.info(f"action: falldown, {confidence}  {meta_data['num_frame']}")

                is_longterm_check = True
            else:
                action_name = 'Normal' #TODO 임시. 그려지는 루틴이 분리가 안되서 임시 처리

            if is_longterm_check:
                if check_longterm(tracks=tracks, meta_data=meta_data):
                    event_pipe.send({'action': "longterm_status", 'id':tid, 'meta_data': meta_data})
                    logger.info(f"action: longterm, {meta_data['num_frame']}")
                    action_name = "longterm"

            if debug_args.visualize:
                if init_flag == True:
                    visualizer.mkdir(meta_data['timestamp'])
                    init_flag = False
                visualizer.save_temp_image([meta_data["v_frame"], action_name, confidence], meta_data["num_frame"])
        else:
            time.sleep(0.0001)
