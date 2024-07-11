import sys
sys.path.insert(0, '/System_Integration/_HAR/HRI')
import _MOT.face_detection as face_detection
import torch
from facial_emotion import MTNet, get_model_path
from torchvision import transforms
from PIL import Image
from collections import Counter
from _Utils.logger import get_logger
from variable import get_emotion_args, get_debug_args

############################################################################## 감정은 문자열이 아닌 숫자로 DB에 들어가기 때문에 필요한 함수.
def map_emotion_to_index(emotion):
    if emotion == 'NE':
        return 0
    elif emotion == 'NEG-1':
        return 1
    elif emotion == 'NEG-2':
        return 2
    else:
        return -1  # 알 수 없는 경우에 대한 처리
##############################################################################

def print_most_common_label(emotions, logger):
    common_label = Counter(emotions).most_common(1)[0][0]
    logger.info(f"emotion Label: {common_label}")
    return common_label

##############################################################################
def Emotion(data_pipe, event_pipe, emotion_result_queue): # 실시간 객체 정보는 큐-스레드로 관리하기 때문에 큐를 매개변수로 추가
##############################################################################
    logger = get_logger(name="[HRI]", console=True, file=True)
    args = get_emotion_args()
    debug_args = get_debug_args()
    test_transforms = transforms.Compose(
        [
            transforms.Resize((260,260)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225])
        ]
        )
    detector = face_detection.build_detector(args.face_detector, confidence_threshold=.5, nms_iou_threshold=.3)

    num_emotion = 3
    emotion_to_class = {0: 'NE', 1: 'NEG-1', 2: 'NEG-2'}

    model = MTNet(num_emotion, num_race=6, num_sex=2).to(args.device)
    model.load_state_dict(torch.load(args.model_state))
    model.eval()

    emotions_list = []
    frame_rate = 30
    data_pipe.send(True)
    while True:
        frame, meta_data, num_frame = data_pipe.recv()
        if frame is not None:
            detections = detector.detect(frame)
            for i in range(detections.shape[0]):
                x1,y1,x2,y2=detections[i][0:4]
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
                face_img=frame[y1:y2,x1:x2,:]
                try:
                    face_img2 = test_transforms(Image.fromarray(face_img).convert('RGB')).unsqueeze(0).to(args.device)
                except Exception as e:
                    logger.error(f"Error: {e}")
                    continue # pass로 할지 continue로 할지 고민중입니다.
                e_out, i_out, _ = model(face_img2)
                emotion = emotion_to_class[torch.argmax(e_out).item()]
                emotions_list.append(emotion)
            if len(emotions_list) >= frame_rate:
                common_label = print_most_common_label(emotions_list, logger)
                ##############################################################################
                mapped_emotion_results = [map_emotion_to_index(common_label)]
                print(f"mapped_emotion_results : {mapped_emotion_results}")
                emotion_result_queue.put({"mapped_emotion_results": mapped_emotion_results, 'id':1, "cctv_id": meta_data['cctv_id'], "current_datetime": meta_data['current_datetime']}) # common_label 값 알아서 일단 0620 시연버전으로 만들기
                # event_pipe.send({'action': common_label, 'id':1, 'cctv_id':meta_data['cctv_id'], 'current_datetime':meta_data['current_datetime']})
                ############################################################################## 매우 심각한 감정에 대한 알람 처리는 아직 미구현 상태
                emotions_list = []
        logger.info(f"[Emotion] num_frame : {num_frame}")
if __name__ == '__main__':
    Emotion()



# 이런 방식으로 해야한다고 하심. -봉준님
# 그리고 emotion은 event_pipe가 아닌 radar 값이랑 모이는 pipe로 가야하는데, 이 부분은 제가 하겠습니다.
# def Emotion(input_queue, output_queue):
#     args = parse_args()

#     IMG_SIZE = 260    
#     emotions_list = []
#     emotion_to_class = {0: 'NE', 1: 'NEG-1', 2: 'NEG-2'}

#     test_transforms = transforms.Compose([
#             transforms.Resize((IMG_SIZE, IMG_SIZE)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])
       
#     emotion_model = init_model()

#     while True:
#         if not input_queue.empty():
#             tracks, meta_data = input_queue.get()
#             tid_list = list()
#             emotion_results = list()
#             face_detections = meta_data['face_detections']

#             for i, track in enumerate(tracks):
#                 frame = meta_data['frame']

#                 bbox = track.tlbr
#                 tid = track.track_id

#                 if face_detections is not None:
#                     for i in range(face_detections.shape[0]):
#                         hbox = face_detections[i][0:4]

#                         def check_hbox(bbox, hbox):
#                             bx1 = int(bbox[0])
#                             by1 = int(bbox[1])
#                             bx2 = int(bbox[2])
#                             by2 = int(bbox[3])  

#                             hx1 = int(hbox[0])
#                             hy1 = int(hbox[1])
#                             hx2 = int(hbox[2])
#                             hy2 = int(hbox[3])

#                             if bx1 <= hx1 and by1 <= hy1 and bx2 >= hx2 and by2 >= hy2:
#                                 return True
#                             else:
#                                 return False

#                         if check_hbox(bbox=bbox, hbox=hbox) == False:
#                             continue

#                         fx1 = int(hbox[0])
#                         fy1 = int(hbox[1])
#                         fx2 = int(hbox[2])
#                         fy2 = int(hbox[3])

#                         face_img = frame[fy1:fy2, fx1:fx2, :]                    
#                         # cv2.imwrite(str(meta_data['num_frame'])+'_'+str(tid)+".png", face_img)

#                         face_img_transformed = test_transforms(Image.fromarray(face_img).convert('RGB')).unsqueeze(0).to(DEVICE)
#                         e_out, _, _ = emotion_model(face_img_transformed)
#                         predicted_emotion = torch.argmax(e_out).item()
#                         emotion = emotion_to_class[predicted_emotion]
                    
#                         # 감정 결과를 리스트에 추가
#                         tid_list.append(tid)
#                         emotion_results.append(emotion)
#                         mapped_emotion_results = [map_emotion_to_index(emotion) for emotion in emotion_results]
                            
#                         output_queue.put(               
#                             {'mapped_emotion_results': mapped_emotion_results, 'id':tid, 'cctv_id':meta_data['cctv_id'], 'current_datetime':meta_data['current_datetime']}                    
#                             )
                    
#                         # 저장된 감정 결과 출력
#                         LOGGER.info(f"감정 {tid_list} {mapped_emotion_results}, output Queue Size : {output_queue.qsize()}")
                      
#         else:  
#             time.sleep(0.00001)

############################################################################################################################################################