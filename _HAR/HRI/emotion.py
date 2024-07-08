import sys
sys.path.insert(0, '/System_Integration/_HAR/HRI')
import torch
from torchvision import transforms
from facial_emotion import MTNet, get_model_path
from PIL import Image
from collections import Counter
import _MOT.face_detection as face_detection
from _Utils.logger import get_logger
from variable import get_emotion_args, get_debug_args

def print_most_common_label(emotions, logger):
    common_label = Counter(emotions).most_common(1)[0][0]
    logger.info(f"emotion Label: {common_label}")
    return common_label

def Emotion(data_pipe, event_pipe):
    logger = get_logger(name="[HRI]", console=True, file=False)
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
        
        data = data_pipe.recv()
        if data:
            if data == "end_flag":
                logger.warning("Emotion process end.")
                break
            frame, meta_data = data
            if frame is not None:
                detections = detector.detect(frame)
                for i in range(detections.shape[0]):
                    x1,y1,x2,y2=detections[i][0:4]
                    x1 = int(x1)
                    y1 = int(y1)
                    x2 = int(x2)
                    y2 = int(y2)
                    face_img=frame[y1:y2,x1:x2,:]
                    face_img2 = test_transforms(Image.fromarray(face_img).convert('RGB')).unsqueeze(0).to(args.device)
                    e_out, i_out, _ = model(face_img2)
                    emotion = emotion_to_class[torch.argmax(e_out).item()]
                    emotions_list.append(emotion)
                if len(emotions_list) >= frame_rate:
                    common_label = print_most_common_label(emotions_list, logger)
                    event_pipe.send({'action': common_label, 'id':1, 'cctv_id':meta_data['cctv_id'], 'current_datetime':meta_data['current_datetime']})
                    emotions_list = []

if __name__ == '__main__':
    Emotion()