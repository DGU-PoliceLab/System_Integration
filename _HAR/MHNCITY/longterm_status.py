import time
import cv2
import numpy as np
from skeleton_results import pose_results 

# 움직임 판별
def is_movement_large(coordinates1, coordinates2, threshold):
    # 좌표 간 차이 계산 로직 작성
    return difference > threshold

# 장시간 고정자세 탐지
def detect_static_pose(skeleton_data, joint_of_interest, threshold, duration_threshold):
    start_time = time.time()
    static_pose_count = 0
    
    for frame in skeleton_data:
        joint_coordinates = extract_joint_coordinates(frame)
        
        if is_movement_large(joint_coordinates, previous_frame_coordinates, threshold):
            static_pose_count = 0
        else:
            static_pose_count += 1
            
        if static_pose_count >= duration_threshold:
            elapsed_time = time.time() - start_time
            return True, elapsed_time
        
        previous_frame_coordinates = joint_coordinates
    
    return False, 0



# 스켈레톤 좌표를 저장한 리스트
pose_results['keypoints']

# 스켈레톤 좌표 분석을 위한 변수
threshold_distance = 10  # 이동 거리 임계값 (조정 가능)
duration_threshold = 5  # 장시간 고정자세로 판단할 지속 시간 (조정 가능)
fixed_pose_detected = False
fixed_pose_duration = 0

# 이미지 로드
image_path = "이미지_파일_경로"
image = cv2.imread(image_path)

# 장시간 고정자세 탐지
for i in range(1, len(skeleton_coords)):
    # 이전 프레임과 현재 프레임의 스켈레톤 좌표 가져오기
    prev_coord = skeleton_coords[i-1]
    curr_coord = skeleton_coords[i]

    # 이동 거리 계산
    distance = np.linalg.norm(np.array(prev_coord) - np.array(curr_coord))

    if distance <= threshold_distance:
        # 이동 거리가 임계값 이하인 경우
        fixed_pose_duration += 1
        if fixed_pose_duration >= duration_threshold:
            fixed_pose_detected = True
            break
    else:
        # 이동 거리가 임계값 이상인 경우
        fixed_pose_duration = 0

# 이미지에 텍스트 추가
if fixed_pose_detected:
    text = "장시간 고정자세"
    text_position = (10, 30)  # 텍스트 위치 (조정 가능)
    text_color = (0, 255, 0)  # 텍스트 색상 (BGR 형식, 여기서는 초록색)
    text_size = 1  # 텍스트 크기 (조정 가능)
    text_thickness = 2  # 텍스트 두께 (조정 가능)

    cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, text_size, text_color, text_thickness)

# 이미지 보여주기
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()



# 장시간 고정자세 탐지에 필요한 변수 및 임계값 설정
THRESHOLD_TIME = 0.3  # 장시간 고정자세 탐지를 위한 시간 임계값 (초)
THRESHOLD_SCORE = 10.0  # 움직임 점수 임계값

# 장시간 고정자세를 탐지할 프레임 인덱스 저장
static_frames = []

# 스켈레톤 데이터를 순회하면서 장시간 고정자세 탐지
for frame_idx in range(1, len(pose_results)):
    # 이전 프레임의 스켈레톤 좌표
    prev_keypoints = pose_results[frame_idx - 120]['keypoints']

    # 현재 프레임의 스켈레톤 좌표
    current_keypoints = pose_results[frame_idx]['keypoints']

    # 스켈레톤 좌표의 움직임 계산
    movement_vectors = current_keypoints - prev_keypoints
    movement_score = np.mean(movement_vectors)

    # # 움직임 벡터의 크기 계산
    # movement_magnitudes = np.linalg.norm(movement_vectors, axis=2)

    # # 움직임 점수 계산
    # movement_score = np.mean(movement_magnitudes)

    # 움직임 점수를 기준으로 장시간 고정자세 탐지
    # if movement_score < THRESHOLD_SCORE and frame_idx > 0 and frame_idx - 1 in static_frames:
    #     static_frames.append(frame_idx)
    if movement_score < THRESHOLD_SCORE:
        static_frames.append(frame_idx)

# 장시간 고정자세 탐지 시 영상 오른쪽 상단에 텍스트 추가하는 함수
def add_text_to_image(image, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (image.shape[1] - 300, 50)
    font_scale = 1
    color = (0, 0, 255)  # Red color
    thickness = 2
    cv2.putText(image, text, org, font, font_scale, color, thickness)

# 스켈레톤 좌표가 적용된 영상 읽어오기
video = cv2.VideoCapture('demo/longterm_status_out.mp4')  # 스켈레톤 좌표가 적용된 영상 경로를 지정해주세요

# 비디오 프레임의 크기와 FPS 정보 가져오기
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)

# 저장할 비디오 파일의 경로와 설정
output_path = "demo/longterm_video.mp4"  # 저장할 비디오 파일 경로
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 코덱 설정
output_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while True:
    # 비디오 프레임 읽어오기
    ret, frame = video.read()

    # 비디오 프레임이 제대로 읽어왔는지 확인
    if not ret:
        break

    # 프레임에 장시간 고정자세 텍스트 추가
    if video.get(cv2.CAP_PROP_POS_FRAMES) in static_frames:
        add_text_to_image(frame, "longterm_status")

    # 프레임 표시
    cv2.imshow("Video", frame)

    # 비디오 저장
    output_video.write(frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 비디오 재생이 끝나면 종료
video.release()
output_video.release()
cv2.destroyAllWindows()


# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import tempfile

import cv2
import mmcv
import mmengine
import numpy as np
import torch
from mmengine import DictAction
from mmengine.utils import track_iter_progress

from mmaction.apis import (detection_inference, inference_recognizer,
                           init_recognizer, pose_inference)
from mmaction.registry import VISUALIZERS
from mmaction.utils import frame_extract

try:
    import moviepy.editor as mpy
except ImportError:
    raise ImportError('Please install moviepy to enable output file')

FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.75
FONTCOLOR = (255, 255, 255)  # BGR, white
THICKNESS = 1
LINETYPE = 1


def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 demo')
    parser.add_argument('--video', default = './test_data/longterm_status.mp4', help='./test_data/1_18_1.mp4')
    parser.add_argument('--out_filename', default = './longterm_status.mp4', help='./results/1_18_1_out.mp4')
    parser.add_argument(
        '--config',
        default=('configs/skeleton/posec3d/'
                 'slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py'),
        help='skeleton model config file path')
    parser.add_argument(
        '--checkpoint',
        default=('https://download.openmmlab.com/mmaction/skeleton/posec3d/'
                 'slowonly_r50_u48_240e_ntu60_xsub_keypoint/'
                 'slowonly_r50_u48_240e_ntu60_xsub_keypoint-f3adabf1.pth'),
        help='skeleton model checkpoint file/url')
    parser.add_argument(
        '--det-config',
        default='demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py',
        help='human detection config file path (from mmdet)')
    parser.add_argument(
        '--det-checkpoint',
        default=('http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/'
                 'faster_rcnn_r50_fpn_2x_coco/'
                 'faster_rcnn_r50_fpn_2x_coco_'
                 'bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'),
        help='human detection checkpoint file/url')
    parser.add_argument(
        '--det-score-thr',
        type=float,
        default=0.9,
        help='the threshold of human detection score')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=0,
        help='the category id for human detection')
    parser.add_argument(
        '--pose-config',
        default='demo/demo_configs/'
        'td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py',
        help='human pose estimation config file path (from mmpose)')
    parser.add_argument(
        '--pose-checkpoint',
        default=('https://download.openmmlab.com/mmpose/top_down/hrnet/'
                 'hrnet_w32_coco_256x192-c78dce93_20200708.pth'),
        help='human pose estimation checkpoint file/url')
    parser.add_argument(
        '--label-map',
        default='tools/data/skeleton/label_map_ntu60.txt',
        help='label map file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--short-side',
        type=int,
        default=480,
        help='specify the short-side length of the image')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    args = parser.parse_args()
    return args


def visualize(args, frames, data_samples, action_label):
    pose_config = mmengine.Config.fromfile(args.pose_config)
    visualizer = VISUALIZERS.build(pose_config.visualizer)
    visualizer.set_dataset_meta(data_samples[0].dataset_meta)

    vis_frames = []
    print('Drawing skeleton for each frame')
    for d, f in track_iter_progress(list(zip(data_samples, frames))):
        f = mmcv.imconvert(f, 'bgr', 'rgb')
        visualizer.add_datasample(
            'result',
            f,
            data_sample=d,
            draw_gt=False,
            draw_heatmap=False,
            draw_bbox=True,
            show=False,
            wait_time=0,
            out_file=None,
            kpt_thr=0.3)
        vis_frame = visualizer.get_image()
        cv2.putText(vis_frame, action_label, (10, 30), FONTFACE, FONTSCALE,
                    FONTCOLOR, THICKNESS, LINETYPE)
        vis_frames.append(vis_frame)

    vid = mpy.ImageSequenceClip(vis_frames, fps=30)
    vid.write_videofile(args.out_filename, remove_temp=True)


def main():
    global pose_results
    args = parse_args()

    tmp_dir = tempfile.TemporaryDirectory()
    frame_paths, frames = frame_extract(args.video, tmp_dir.name)

    num_frame = len(frame_paths)
    h, w, _ = frames[0].shape

    # Get Human detection results.
    det_results, _ = detection_inference(args.det_config, args.det_checkpoint,
                                         frame_paths, args.det_score_thr,
                                         args.det_cat_id, args.device)
    torch.cuda.empty_cache()

    # Get Pose estimation results.
    pose_results, pose_data_samples = pose_inference(args.pose_config,
                                                     args.pose_checkpoint,
                                                     frame_paths, det_results,
                                                     args.device)
    torch.cuda.empty_cache()

    fake_anno = dict(
        frame_dir='',
        label=-1,
        img_shape=(h, w),
        original_shape=(h, w),
        start_index=0,
        modality='Pose',
        total_frames=num_frame)
    num_person = max([len(x['keypoints']) for x in pose_results])

    num_keypoint = 17
    keypoint = np.zeros((num_frame, num_person, num_keypoint, 2),
                        dtype=np.float16)
    keypoint_score = np.zeros((num_frame, num_person, num_keypoint),
                              dtype=np.float16)
    for i, poses in enumerate(pose_results):
        # if poses['keypoints'].shape[0] < 3 :
        #     poses['keypoints'] = np.concatenate((poses['keypoints'], np.zeros((1, 17, 2))), axis=0)
            
        # if poses['keypoint_scores'].shape[0] < 3 :
        #     poses['keypoints_scores'] = np.concatenate((poses['keypoints'], np.zeros((1, 17, 2))), axis=0)
                    
        keypoint[i, :len(poses['keypoints'])] = poses['keypoints']
        keypoint_score[i, :len(poses['keypoint_scores'])] = poses['keypoint_scores']

    fake_anno['keypoint'] = keypoint.transpose((1, 0, 2, 3))
    fake_anno['keypoint_score'] = keypoint_score.transpose((1, 0, 2))

    config = mmengine.Config.fromfile(args.config)
    config.merge_from_dict(args.cfg_options)

    model = init_recognizer(config, args.checkpoint, args.device)
    result = inference_recognizer(model, fake_anno)

    max_pred_index = result.pred_scores.item.argmax().item()
    label_map = [x.strip() for x in open(args.label_map).readlines()]
    action_label = label_map[max_pred_index]

    visualize(args, frames, pose_data_samples, action_label)

    tmp_dir.cleanup()


if __name__ == '__main__':
    main()



# --- longterm
import cv2
import numpy as np

# 장시간 고정자세 탐지에 필요한 변수 및 임계값 설정
THRESHOLD_TIME = 0.3  # 장시간 고정자세 탐지를 위한 시간 임계값 (초)
THRESHOLD_SCORE = 10.0  # 움직임 점수 임계값

# 장시간 고정자세를 탐지할 프레임 인덱스 저장
static_frames = []

# 비디오 저장
output_path = "demo/longterm_test.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# output_video = cv2.VideoWriter(output_path, fourcc, 30.0, (frame_width, frame_height))


# 스켈레톤 데이터를 순회하면서 장시간 고정자세 탐지
for frame_idx in range(1, len(pose_results)):
    # 이전 프레임의 스켈레톤 좌표
    prev_keypoints = pose_results[frame_idx - 120]['keypoints']

    # 현재 프레임의 스켈레톤 좌표
    current_keypoints = pose_results[frame_idx]['keypoints']

    # 스켈레톤 좌표의 움직임 계산
    movement_vectors = current_keypoints - prev_keypoints
    movement_score = np.mean(movement_vectors)
    
    # movement_score가 임계값보다 작으면 해당 frame을 static_frames에 추가 및 장시간 고정자세 텍스트 추가
    if -10<movement_score<10:
        static_frames.append(frame_idx)
        cv2.putText(pose_results[frame_idx+1], "longterm_status", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    output_video.write(pose_results)



# --- longterm status
import cv2
import numpy as np


video = cv2.VideoCapture('./longterm_status_testout.mp4')
# 비디오 프레임의 크기와 FPS 정보 가져오기
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)

# 저장할 비디오 파일의 경로와 설정
output_path = "demo/longterm_status_testout.mp4"  # 저장할 비디오 파일 경로
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 코덱 설정
output_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
ret, frame = video.read()

# 장시간 고정자세 탐지에 필요한 변수 및 임계값 설정
THRESHOLD_TIME = 3  # 장시간 고정자세 탐지를 위한 시간 임계값 (초)
THRESHOLD_DISTANCE = 15.0  # 움직임 거리 임계값

# 장시간 고정자세를 탐지할 프레임 인덱스 저장
static_frames = []

# 장시간 고정자세 탐지
for frame_idx in range(1, len(pose_results)):
    # 이전 프레임의 스켈레톤 좌표
    prev_keypoints = pose_results[frame_idx - 120]['keypoints']

    # 현재 프레임의 스켈레톤 좌표
    current_keypoints = pose_results[frame_idx]['keypoints']

    # 스켈레톤 좌표의 움직 임 계산
    movement_vectors = current_keypoints - prev_keypoints
    movement_distances = np.linalg.norm(movement_vectors, axis=2)

    # 각 포인트별 움직임 여부 확인
    point_movement = np.any(movement_distances > THRESHOLD_DISTANCE)

    # 움직임이 없는 경우 장시간 고정자세로 탐지
    if not point_movement:
        static_frames.append(frame_idx)
        print('Longterm_status = True')
    else :
        print('Longterm_status = False')