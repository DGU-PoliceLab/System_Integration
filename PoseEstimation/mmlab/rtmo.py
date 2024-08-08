import sys
sys.path.insert(0, "/System_Integration/PoseEstimation/")
from argparse import ArgumentParser
from typing import Dict
import cv2
from datetime import datetime
from queue import Queue

from Utils.logger import get_logger

from mmpose.apis.inferencers import MMPoseInferencer, get_model_aliases

filter_args = dict(bbox_thr=0.3, nms_thr=0.3, pose_based_nms=False)
POSE2D_SPECIFIC_ARGS = dict(
    yoloxpose=dict(bbox_thr=0.01, nms_thr=0.65, pose_based_nms=True),
    rtmo=dict(bbox_thr=0.65, nms_thr=0.65, pose_based_nms=True),
)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'inputs',
        type=str,
        nargs='?',
        help='Input image/video path or folder path.')
    parser.add_argument(
        '--pose2d',
        type=str,
        default="rtmo",
        help='Pretrained 2D pose estimation algorithm. It\'s the path to the '
        'config file or the model name defined in metafile.')
    parser.add_argument(
        '--pose2d-weights',
        type=str,
        default="PoseEstimation/mmlab/mmpose/checkpoints/rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.pth",
        help='Path to the custom checkpoint file of the selected pose model. '
        'If it is not specified and "pose2d" is a model name of metafile, '
        'the weights will be loaded from metafile.')
    parser.add_argument(
        '--pose3d',
        type=str,
        default=None,
        help='Pretrained 3D pose estimation algorithm. It\'s the path to the '
        'config file or the model name defined in metafile.')
    parser.add_argument(
        '--pose3d-weights',
        type=str,
        default=None,
        help='Path to the custom checkpoint file of the selected pose model. '
        'If it is not specified and "pose3d" is a model name of metafile, '
        'the weights will be loaded from metafile.')
    parser.add_argument(
        '--det-model',
        type=str,
        default=None,
        help='Config path or alias of detection model.')
    parser.add_argument(
        '--det-weights',
        type=str,
        default=None,
        help='Path to the checkpoints of detection model.')
    parser.add_argument(
        '--det-cat-ids',
        type=int,
        nargs='+',
        default=0,
        help='Category id for detection model.')
    parser.add_argument(
        '--scope',
        type=str,
        default='mmpose',
        help='Scope where modules are defined.')
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device used for inference. '
        'If not specified, the available device will be automatically used.')
    parser.add_argument(
        '--show-progress',
        action='store_true',
        default=False,
        help='Display the progress bar during inference.')
    args, _ = parser.parse_known_args()
    for model in POSE2D_SPECIFIC_ARGS:
        if model in args.pose2d:
            filter_args.update(POSE2D_SPECIFIC_ARGS[model])
            break
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display the image/video in a popup window.')
    parser.add_argument(
        '--draw-bbox',
        action='store_true',
        default=True,
        help='Whether to draw the bounding boxes.')
    parser.add_argument(
        '--draw-heatmap',
        action='store_true',
        default=False,
        help='Whether to draw the predicted heatmaps.')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=filter_args['bbox_thr'],
        help='Bounding box score threshold')
    parser.add_argument(
        '--nms-thr',
        type=float,
        default=filter_args['nms_thr'],
        help='IoU threshold for bounding box NMS')
    parser.add_argument(
        '--pose-based-nms',
        type=lambda arg: arg.lower() in ('true', 'yes', 't', 'y', '1'),
        default=filter_args['pose_based_nms'],
        help='Whether to use pose-based NMS')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--tracking-thr', type=float, default=0.3, help='Tracking threshold')
    parser.add_argument(
        '--use-oks-tracking',
        action='store_true',
        help='Whether to use OKS as similarity in tracking')
    parser.add_argument(
        '--disable-norm-pose-2d',
        action='store_true',
        help='Whether to scale the bbox (along with the 2D pose) to the '
        'average bbox scale of the dataset, and move the bbox (along with the '
        '2D pose) to the average bbox center of the dataset. This is useful '
        'when bbox is small, especially in multi-person scenarios.')
    parser.add_argument(
        '--disable-rebase-keypoint',
        action='store_true',
        default=False,
        help='Whether to disable rebasing the predicted 3D pose so its '
        'lowest keypoint has a height of 0 (landing on the ground). Rebase '
        'is useful for visualization when the model do not predict the '
        'global position of the 3D pose.')
    parser.add_argument(
        '--num-instances',
        type=int,
        default=1,
        help='The number of 3D poses to be visualized in every frame. If '
        'less than 0, it will be set to the number of pose results in the '
        'first frame.')
    parser.add_argument(
        '--radius',
        type=int,
        default=3,
        help='Keypoint radius for visualization.')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization.')
    parser.add_argument(
        '--skeleton-style',
        default='mmpose',
        type=str,
        choices=['mmpose', 'openpose'],
        help='Skeleton style selection')
    parser.add_argument(
        '--black-background',
        action='store_true',
        help='Plot predictions on a black image')
    parser.add_argument(
        '--vis-out-dir',
        type=str,
        # default='_Output',
        default='',
        help='Directory for saving visualized results.')
    parser.add_argument(
        '--pred-out-dir',
        type=str,
        # default='_Output',
        default='',
        help='Directory for saving inference results.')
    parser.add_argument(
        '--show-alias',
        action='store_true',
        help='Display all the available model aliases.')

    call_args = vars(parser.parse_args())

    init_kws = [
        'pose2d', 'pose2d_weights', 'scope', 'device', 'det_model',
        'det_weights', 'det_cat_ids', 'pose3d', 'pose3d_weights',
        'show_progress'
    ]
    init_args = {}
    for init_kw in init_kws:
        init_args[init_kw] = call_args.pop(init_kw)

    display_alias = call_args.pop('show_alias')

    return init_args, call_args, display_alias

def get_model():
    init_args, call_args, display_alias = parse_args()
    inferencer = MMPoseInferencer(**init_args)
    return inferencer, init_args, call_args, display_alias

def main(log_opt=[True, True]):
    init_args, call_args, display_alias = parse_args()
    inferencer = MMPoseInferencer(**init_args)
    cur_frame = 0
    cap = cv2.VideoCapture(call_args['inputs'])
    if cap.isOpened():
        while True:
            ret, frame = cap.read()
            if ret:
                temp_call_args = call_args
                temp_call_args['inputs'] = frame
                for _ in inferencer(**temp_call_args):
                    pred = _['predictions'][0]
                    l_p = len(pred)
                    n_person = 1
                    pred.sort(key = lambda x: x['bbox'][0][0])
                    for p in pred:
                        keypoints = p['keypoints']
                        bbox = p['bbox']
                        bbox_score = p['bbox_score']
                        n_person += 1
                cur_frame += 1
            else:
                break
    cap.release()

def rtmo(source, rtmo_queue, log_opt=["RTMO", False, False]):
    logger = get_logger(name = '[RTMO]', console=False, file=False)
    init_args = {'pose2d': 'rtmo', 'pose2d_weights': "PoseEstimation/mmlab/mmpose/checkpoints/rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.pth", 'scope': 'mmpose', 'device': None, 'det_model': None, 'det_weights': None, 'det_cat_ids': 0, 'pose3d': None, 'pose3d_weights': None, 'show_progress': False}
    call_args = {'inputs': source, 'show': False, 'draw_bbox': True, 'draw_heatmap': False, 'bbox_thr': 0.5, 'nms_thr': 0.65, 'pose_based_nms': True, 'kpt_thr': 0.3, 'tracking_thr': 0.3, 'use_oks_tracking': False, 'disable_norm_pose_2d': False, 'disable_rebase_keypoint': False, 'num_instances': 1, 'radius': 3, 'thickness': 1, 'skeleton_style': 'mmpose', 'black_background': False, 'vis_out_dir': '', 'pred_out_dir': '', 'vis-out-dir': '_Output'}
    inferencer = MMPoseInferencer(**init_args)
    cur_frame = 0
    cap = cv2.VideoCapture(call_args['inputs'])
    if cap.isOpened():
        while True:
            ret, frame = cap.read()
            if ret:
                temp_call_args = call_args
                temp_call_args['inputs'] = frame
                results = inferencer(**temp_call_args)
                rtmo_queue.put(results)
                for _ in results:
                    pred = _['predictions'][0]
                    l_p = len(pred)
                    logger.info(f'frame {cur_frame} - {l_p} person detect!')
                    n_person = 1
                    pred.sort(key = lambda x: x['bbox'][0][0])
                    for p in pred:
                        keypoints = p['keypoints']
                        bbox = p['bbox']
                        bbox_score = p['bbox_score']
                        logger.info(f'person #{n_person} - bbox:{bbox}(score: {bbox_score}) - keypoints:{keypoints}')
                        n_person += 1
                cur_frame += 1
            else:
                break
    else:
        logger.error('video error')
    cap.release()

if __name__ == '__main__':
    main()
