import cv2
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
import queue
import datetime
import time
from PIL import Image
from multiprocessing import Process
from threading import Thread

from HRI.head_bbox import *
from HRI.facial_emotion import MTNet, get_model_path # 감정인식

from _Utils.logger import get_logger
import argparse


LOGGER = get_logger(name="[HRI]", console=False, file=False)

DEVICE = 'cuda:0'
NUM_OF_EMOTION = 3

def  parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_filename', default='out.avi') # 경로수정 필요
    parser.add_argument('--test_video', type=str, default='distancedata0_video.mp4', help='distancedata0_video.mp4') #비디오 인풋입니다.
    parser.add_argument('--face_detector', type=str, default='RetinaNetResNet50', help='DSFDDetector/RetinaNetResNet50')
    args = parser.parse_args()
    return args


def init_model():
    emotion_model = None
    emotion_model = MTNet(NUM_OF_EMOTION, num_race=6, num_sex=2).to(DEVICE)
    emotion_model.load_state_dict(torch.load('/System_Integration/_HAR/HRI/models/model_state.pth', map_location=torch.device('cuda')))

    emotion_model.eval()
    return emotion_model

def map_emotion_to_index(emotion):
    if emotion == 'NE':
        return 0
    elif emotion == 'NEG-1':
        return 1
    elif emotion == 'NEG-2':
        return 2
    else:
        return -1  # 알 수 없는 경우에 대한 처리

def temp_preprocess(skeleton, bbox):
    head_bbox = calculate_head_box(skeleton, bbox)

    return head_bbox

def Emotion(input_queue, output_queue):
    args = parse_args()

    IMG_SIZE = 260    
    emotions_list = []
    emotion_to_class = {0: 'NE', 1: 'NEG-1', 2: 'NEG-2'}

    test_transforms = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # Run Emotion detection - hri
    # import sys
    # sys.path.insert(0, "/System_Integration/_HAR/HRI")
    # import face_detection
    # detector = face_detection.build_detector(args.face_detector, confidence_threshold=.5, nms_iou_threshold=.3)
       
    emotion_model = init_model()

    while True:
        if not input_queue.empty():
            tracks, meta_data = input_queue.get()
            tid_list = list()
            emotion_results = list()
            face_detections = meta_data['face_detections']

            for i, track in enumerate(tracks):
                frame = meta_data['frame']

                bbox = track.tlbr
                tid = track.track_id

                if face_detections is not None:
                    for i in range(face_detections.shape[0]):
                        hbox = face_detections[i][0:4]

                        def check_hbox(bbox, hbox):
                            bx1 = int(bbox[0])
                            by1 = int(bbox[1])
                            bx2 = int(bbox[2])
                            by2 = int(bbox[3])  

                            hx1 = int(hbox[0])
                            hy1 = int(hbox[1])
                            hx2 = int(hbox[2])
                            hy2 = int(hbox[3])

                            if bx1 <= hx1 and by1 <= hy1 and bx2 >= hx2 and by2 >= hy2:
                                return True
                            else:
                                return False

                        if check_hbox(bbox=bbox, hbox=hbox) == False:
                            continue

                        fx1 = int(hbox[0])
                        fy1 = int(hbox[1])
                        fx2 = int(hbox[2])
                        fy2 = int(hbox[3])

                        face_img = frame[fy1:fy2, fx1:fx2, :]                    
                        # cv2.imwrite(str(meta_data['num_frame'])+'_'+str(tid)+".png", face_img)

                        face_img_transformed = test_transforms(Image.fromarray(face_img).convert('RGB')).unsqueeze(0).to(DEVICE)
                        e_out, _, _ = emotion_model(face_img_transformed)
                        predicted_emotion = torch.argmax(e_out).item()
                        emotion = emotion_to_class[predicted_emotion]
                    
                        # 감정 결과를 리스트에 추가
                        tid_list.append(tid)
                        emotion_results.append(emotion)
                        mapped_emotion_results = [map_emotion_to_index(emotion) for emotion in emotion_results]
                            
                        # output_queue.put(               
                        #     {'mapped_emotion_results': mapped_emotion_results, 'id':tid, 'cctv_id':meta_data['cctv_id'], 'current_datetime':meta_data['current_datetime']}                    
                        #     )
                    
                        # 저장된 감정 결과 출력
                        LOGGER.info(f"감정 {tid_list} {mapped_emotion_results} {meta_data['num_frame']}, output Queue Size : {output_queue.qsize()}")
                      
        else:  
            time.sleep(0.00001)
