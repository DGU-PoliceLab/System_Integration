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

EVENT = ["normal", 0.0]

def pre_processing(tracks):
    form = {
        'tid': None,
        'bbox_scores': [],
        'bboxes': [],
        'keypoints_visible': [],
        'keypoint_scores': [],
        'keypoints': []
    }
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

    form['bbox_scores'] = np.array(form['bbox_scores'], dtype=np.float32)
    form['bboxes'] = np.array(form['bboxes'], dtype=np.float32)
    form['keypoints_visible'] = np.array(form['keypoints_visible'], dtype=np.float32)
    form['keypoint_scores'] = np.array(form['keypoint_scores'], dtype=np.float32)
    form['keypoints'] = np.array(form['keypoints'], dtype=np.float32)
    return form

def inference(model, label_map, pose_data, meta_data, pipe, logger):
    global EVENT
    try:
        tid = pose_data[-1]['tid']
        result = inference_skeleton(model, pose_data, (meta_data[-1]['frame_size']))
        max_pred_index = result.pred_score.argmax().item()
        action_label = label_map[max_pred_index]
        confidence = result.pred_score[max_pred_index]
        logger.debug(f"action: {action_label}")
        if action_label != 'normal' and confidence > 0.85:
            print("selfharm", confidence)
            logger.info(f"selfharm detected! {meta_data[-1]['current_datetime']}")
            pipe.send({'action': "selfharm", 'id':tid, 'cctv_id':meta_data[-1]['cctv_id'], 'current_datetime':meta_data[-1]['current_datetime'], 'location':meta_data[-1]['cctv_name'], 'combine_data': None})
        else:
            print("normal", confidence)
        EVENT = [action_label, confidence]
    except Exception as e:
        logger.error(f'Error occured in inference_thread, error: {e}')

def Selfharm(data_pipe, event_pipe):
    global EVENT
    logger = get_logger(name="[PLASS]", console=False, file=False)
    args = get_selfharm_args()
    debug_args = get_debug_args()
    if debug_args.visualize:
        visualizer = Visualizer("selfharm")
    config = mmengine.Config.fromfile(args.config)
    model = init_recognizer(config, args.checkpoint, args.device)
    label_map = [x.strip() for x in open(args.label_map).readlines()]
    pose_array = []
    meta_array = []
    prev_data = None
    
    data_pipe.send(True)
    try:
        while True:
            if len(pose_array) > args.step_size:
                pose_data = pose_array[:args.step_size]
                meta_data = meta_array[:args.step_size]
                pose_array = pose_array[args.step_size:]
                meta_array = meta_array[args.step_size:]
                if args.thread_mode:
                    infrence_thread = Thread(
                        target=inference, 
                        args=(model, label_map, pose_data, meta_data, event_pipe, logger))
                    infrence_thread.start()
                else:
                    inference(model, label_map, pose_data, meta_data, event_pipe, logger)
            data = data_pipe.recv()
            if data and data != prev_data:
                if data == "end_flag":
                    logger.warning("Selfharm process end.")
                    if debug_args.visualize:    
                        visualizer.merge_img_to_video()
                    break
                tracks, meta_data = data
                prev_data = data
                pose_data = pre_processing(tracks)
                if len(tracks) > 0:
                    pose_array.append(pose_data)
                    meta_array.append(meta_data)
                if debug_args.visualize:
                    visualizer.mkdir(meta_data['timestamp'])
                    visualizer.save_temp_image([meta_data["v_frame"], EVENT[0], EVENT[1]], meta_data["num_frame"])
            else:
                time.sleep(0.0001)
    except Exception as e:
        logger.error(f"Error occured in selfharm, {e}")
        exit()
