import sys
import time
sys.path.insert(0, '/System_Integration/HAR/HRI')
import torch
from torchvision import transforms
from facial_emotion import MTNet, get_model_path
from PIL import Image
from collections import Counter
from Utils.logger import get_logger
from Utils._visualize import Visualizer
from variable import get_emotion_args, get_debug_args
from multiprocessing import Process, Pipe
import cv2


def map_emotion_to_index(emotion):
    if emotion == 'NE':
        return 0
    elif emotion == 'NEG-1':
        return 1
    elif emotion == 'NEG-2':
        return 2
    else:
        return -1  # 알 수 없는 경우에 대한 처리
    
def Emotion(data_pipe, event_pipe):
    logger = get_logger(name="[HRI]", console=True, file=False)
    args = get_emotion_args()
    debug_args = get_debug_args()
    if debug_args.visualize:
        visualizer = Visualizer("emotion")
        init_flag = True
    use_transforms = transforms.Compose(
        [
            transforms.Resize((260,260)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        )

    num_emotion = 3
    emotion_to_class = {0: 'NE', 1: 'NEG-1', 2: 'NEG-2'}

    model = MTNet(num_emotion, num_race=6, num_sex=2).to(args.device)
    model.load_state_dict(torch.load(args.model_state))
    model.eval()
   
    data_pipe.send(True)
    while True:        
        data = data_pipe.recv()
        if data:
            if data == "end_flag":
                logger.warning("Emotion process end.")
                if debug_args.visualize:    
                    visualizer.merge_img_to_video()
                break
            tracks, meta_data, face_detections, frame, combine_data = data
            num_frame = meta_data['num_frame']
            event_count = 0
            cctv_id = str(meta_data['cctv_id'])

            import os #TODO TEMP
            snapshot_path = "/System_Integration/Output/NAS/"+cctv_id
            os.makedirs(snapshot_path, exist_ok = True)

            combine_list = []
            for i, track in enumerate(tracks):
                bbox = track.tlbr
                tid = track.track_id

                if face_detections is not None: #TEMP 3중 루프문임. 수정 요망.
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
                        try:       
                            face_img_transformed = use_transforms(Image.fromarray(face_img).convert('RGB')).unsqueeze(0).to(args.device)
                        except Exception as e:
                            logger.warning(f"error: {e}")
                            continue
                        e_out, _, _ = model(face_img_transformed)
                        predicted_emotion = torch.argmax(e_out).item()
                        emotion = emotion_to_class[predicted_emotion]
                        
                        cctv_id = meta_data['cctv_id']

                        file_name = f"{meta_data['timestamp']}_{tid}.jpg"
                        cv2.imwrite(f"{snapshot_path}/{file_name}", face_img) #TODO TEMP
                        file_name = f"{cctv_id}/{file_name}"

                        combine_result_data = {'tid': tid, 'temperature': None, 'breath': None, 'heart': None}
                        for i in range(len(combine_data)):
                            if combine_data[i]['tid'] == tid:
                                combine_result_data = combine_data[i]

                        try:
                            emotion_index = map_emotion_to_index(emotion)
                            combined_emotion_data = {'emotion_index': emotion_index, 'id':tid, 'cctv_id':meta_data['cctv_id'], 
                                            'current_datetime':meta_data['current_datetime'], 'location':meta_data['cctv_name'],
                                            'combine_dict': combine_result_data, 'db_insert_file_path':file_name}
                            logger.debug(f"combined_emotion_data: {combined_emotion_data}")
                            combine_list.append(combined_emotion_data)
                        except Exception as e:
                            logger.warning(e)
            logger.debug(f"combine_list: {combine_list}")
            event_pipe.send({'action': "emotion", "combine_list": combine_list})
        else:
            time.sleep(0.0001)