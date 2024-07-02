import numpy as np
import time
import argparse
from collections import deque
from CSDC.ActionsEstLoader import TSSTG
from multiprocessing import Process, Pipe
from _Utils.logger import get_logger
from _Utils._visualize import visualize
from variable import get_debug_args

LOGGER = get_logger(name="[CSDC]", console=True, file=False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshhold', type=float, default=0.6, help='falldown threshhold')
    parser.add_argument('--frame_step', type=int, default=14, help='inference frame step')
    args = parser.parse_args()
    return args

def preprocess(skeletons, frame_step):
    skeletons = deque(skeletons, maxlen=frame_step)
    for i, sk in enumerate(skeletons):
        if i == frame_step:
            break
        indices_14 = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]        
        skeletons[i] = sk[indices_14]

    return np.array(skeletons)

def check_falldown(action_name='Normal', confidence=0, threshold=0.6):
    if action_name == 'Fall Down' and threshold < confidence:
        return True
    return False   

def Falldown(data_pipe, event_pipe):
    args = parse_args()
    debug_args = get_debug_args()
    if debug_args.visualize:
        frame_pipe, frame_pipe_child = Pipe()
        visualize_process = Process(target=visualize, args=('falldown', frame_pipe_child))
        visualize_process.start()
        frame_pipe.recv()
    action_model = TSSTG()
    data_pipe.send(True)
    while True:
        action_name = 'None'
        confidence = 0
        data = data_pipe.recv()
        if data:
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
            
            if check_falldown(action_name=action_name, confidence=confidence, threshold=args.threshhold):
                tid = 1
                LOGGER.info("action: falldown")
                event_pipe.send({'action': "falldown", 'id':tid, 'cctv_id':meta_data['cctv_id'], 'current_datetime':meta_data['current_datetime']})
            if debug_args.visualize:
                frame_pipe.send([meta_data['frame'], action_name, confidence])
        else:
            time.sleep(0.0001)
