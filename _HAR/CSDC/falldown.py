import time
import numpy as np
from multiprocessing import Process, Pipe
from collections import deque
from copy import deepcopy
from CSDC.ActionsEstLoader import TSSTG
from _HAR.MHNCITY.longterm.longterm import Longterm
from _Utils.logger import get_logger
from _Utils._visualize import Visualizer
from variable import get_falldown_args, get_debug_args

def preprocess(skeletons, frame_step):
    skeletons = deque(skeletons, maxlen=frame_step)
    for i, sk in enumerate(skeletons):
        if i == frame_step:
            break
        indices_14 = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]        
        skeletons[i] = sk[indices_14]

    return np.array(skeletons)

def check_event(action_name='Normal', confidence=0, threshold=0.6):
    if action_name == 'Fall Down' and threshold < confidence:
        return True
    return False   

def Falldown(data_pipe, event_pipe):
    logger = get_logger(name="[CSDC]", console=True, file=False)
    args = get_falldown_args()
    debug_args = get_debug_args()
    if debug_args.visualize:
        visualizer = Visualizer("falldown")
        init_flag = True
    if args.longterm_status:
        longterm_input_pipe, longterm_output_pipe = Pipe()
        longterm_process = Process(target=Longterm, args=(longterm_output_pipe, event_pipe,)) 
        longterm_process.start()
    action_model = TSSTG()
    data_pipe.send(True)
    
    while True:
        action_name = 'None'
        confidence = 0
        data = data_pipe.recv()
        if data:
            if data == "end_flag":
                logger.warning("Falldown process end.")
                if args.longterm_status:
                    longterm_input_pipe.send("end_flag")
                if debug_args.visualize:    
                    visualizer.merge_img_to_video()
                break
            tracks, meta_data = data
            if args.longterm_status:
                longterm_input_tracks = deepcopy(tracks)
                longterm_input_data = [longterm_input_tracks, meta_data]
            for i, track in enumerate(tracks):
                skeletons = track.skeletons
                if len(skeletons) < args.frame_step:
                    continue

                tid = track.track_id
                skeletons = preprocess(skeletons=skeletons, frame_step=args.frame_step)

                out = action_model.predict(skeletons, meta_data['frame_size'])
                action_name = action_model.class_names[out[0].argmax()]
                confidence = out[0][1]
            
            if check_event(action_name=action_name, confidence=confidence, threshold=args.threshhold):
                logger.info("action: falldown")
                event_pipe.send({'action': "falldown", 'id':tid, 'cctv_id':meta_data['cctv_id'], 'current_datetime':meta_data['current_datetime'], 'location':meta_data['cctv_name'],
                                 'combine_data': None})


                if args.longterm_status:
                    longterm_input_pipe.send(longterm_input_data)

            if debug_args.visualize:
                if init_flag == True:
                    visualizer.mkdir(meta_data['timestamp'])
                    init_flag = False
                visualizer.save_temp_image([meta_data["v_frame"], action_name, confidence], meta_data["num_frame"])
        else:
            time.sleep(0.0001)
