import argparse
import sys
sys.path.insert(0, '/System_Integration/_HAR/PLASS')
import time
from threading import Thread
import numpy as np
import mmengine
from mmaction.apis import inference_skeleton, init_recognizer
from _Utils.logger import get_logger

LOGGER = get_logger(name="[PLASS]", console=True, file=True)
EVENT = None

def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 demo')
    parser.add_argument(
        '--config',
        default="_HAR/PLASS/models/config.py",
        help='skeleton model config file path')
    parser.add_argument(
        '--checkpoint',
        default="_HAR/PLASS/models/checkpoint.pth",
        help='skeleton model checkpoint file/url')
    parser.add_argument(
        '--label-map',
        default='_HAR/PLASS/models/labelmap.txt',
        help='label map file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--step-size', type=int, default=15, help='inference step size')
    args = parser.parse_args()
    return args

def pre_processing(tracks):
    form = {
        'bbox_scores': [],
        'bboxes': [],
        'keypoints_visible': [],
        'keypoint_scores': [],
        'keypoints': []
    }
    for track in tracks:
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

def inference(model, label_map, pose_data, meta_data, pipe):
    global EVENT
    try:
        result = inference_skeleton(model, pose_data, (meta_data[0]['frame_size']))
        max_pred_index = result.pred_score.argmax().item()
        action_label = label_map[max_pred_index]
        LOGGER.debug(f"action: {action_label}")
        if action_label == 'selfharm':
            LOGGER.info(f"selfharm detected! {meta_data[0]['current_datetime']}")
            pipe.send({'action': action_label, 'id': 1, 'cctv_id': meta_data[0]['cctv_id'], 'current_datetime': meta_data[0]['current_datetime']})
    except Exception as e:
        LOGGER.error(f'Error occured in inference_thread, error: {e}')

def Selfharm(data_pipe, event_pipe):
    global EVENT
    args = parse_args()
    config = mmengine.Config.fromfile(args.config)
    model = init_recognizer(config, args.checkpoint, args.device)
    label_map = [x.strip() for x in open(args.label_map).readlines()]
    pose_array = []
    meta_array = []
    prev_data = None

    data_pipe.send(True)
    while True:
        if len(pose_array) > args.step_size:
            pose_data = pose_array[:args.step_size]
            meta_data = meta_array[:args.step_size]
            pose_array = pose_array[args.step_size:]
            meta_array = meta_array[args.step_size:]
            infrence_thread = Thread(
                target=inference, 
                args=(model, label_map, pose_data, meta_data, event_pipe)).start()
        data = data_pipe.recv()
        if data and data != prev_data:
            tracks, meta_data = data
            prev_data = data
            pose_data = pre_processing(tracks)
            pose_array.append(pose_data)
            meta_array.append(meta_data)
        else:
            time.sleep(0.0001)

if __name__ == '__main__':
    Selfharm()
