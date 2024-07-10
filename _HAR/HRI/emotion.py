import sys
import time
sys.path.insert(0, '/System_Integration/_HAR/HRI')
import torch
from torchvision import transforms
from facial_emotion import MTNet, get_model_path
from PIL import Image
from collections import Counter
from _Utils.logger import get_logger
from variable import get_emotion_args, get_debug_args
from multiprocessing import Process, Pipe
from _Utils._visualize import visualize
import cv2

def Emotion(data_pipe, event_pipe):
    logger = get_logger(name="[HRI]", console=True, file=False)
    args = get_emotion_args()
    debug_args = get_debug_args()
    use_transforms = transforms.Compose(
        [
            transforms.Resize((260,260)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225])
        ]
        )

    num_emotion = 3
    emotion_to_class = {0: 'NE', 1: 'NEG-1', 2: 'NEG-2'}

    model = MTNet(num_emotion, num_race=6, num_sex=2).to(args.device)
    model.load_state_dict(torch.load(args.model_state))
    model.eval()
   
    # if debug_args.visualize:
    #     frame_pipe, frame_pipe_child = Pipe()
    #     visualize_process = Process(target=visualize, args=('face', frame_pipe_child))
    #     visualize_process.start()
    #     frame_pipe.recv()

    data_pipe.send(True)
    while True:        
        data = data_pipe.recv()
        if data:
            if data == "end_flag":
                logger.warning("Emotion process end.")
                break
            tracks, meta_data, face_detections, frame = data
            
            event_count = 0
            for i, track in enumerate(tracks):
                bbox = track.tlbr
                tid = track.track_id
                num_frame = meta_data['num_frame']

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
                        face_img_transformed = use_transforms(Image.fromarray(face_img).convert('RGB')).unsqueeze(0).to(args.device)
                        e_out, _, _ = model(face_img_transformed)
                        predicted_emotion = torch.argmax(e_out).item()
                        emotion = emotion_to_class[predicted_emotion]
                        
                        logger.info(f"emotion Label: {emotion}")
                        event_pipe.send({'action': emotion, 'id':tid, 'cctv_id':meta_data['cctv_id'], 'current_datetime':meta_data['current_datetime']})
                        event_count += 1

                        if debug_args.visualize:
                            hx1 = int(hbox[0])
                            hy1 = int(hbox[1])
                            hx2 = int(hbox[2])
                            hy2 = int(hbox[3])
                            cv2.rectangle(meta_data['frame'], (hx1, hy1), (hx2, hy2), (255, 0, 0), 2)
                            font =  cv2.FONT_HERSHEY_PLAIN
                            ret_string = f'frame: {num_frame}    id: {tid}  action: {emotion}'

                            cv2.putText(meta_data['frame'], ret_string, (350, 40*event_count), font, 2, (255, 0, 0), 2, cv2.LINE_AA)
                            # cv2.imwrite(f"face_{num_frame}_{tid}.png", face_img)
                   
            if debug_args.visualize:
                cv2.imwrite(f"face_{num_frame}.png", meta_data['frame'])
        else:
            time.sleep(0.0001)

if __name__ == '__main__':
    Emotion()