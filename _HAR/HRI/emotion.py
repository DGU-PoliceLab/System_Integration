import sys
sys.path.insert(0, '/System_Integration/_HAR/HRI')
import face_detection
import torch
from facial_emotion import MTNet, get_model_path
from torchvision import transforms
from PIL import Image
import argparse
from collections import Counter
from _Utils.logger import get_logger

LOGGER = get_logger(name="[HRI]", console=True, file=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_state', type=str, default='_HAR/HRI/Models/model_state.pth', help='model state checkpoint path')
    parser.add_argument('--face_detector', type=str, default='RetinaNetResNet50', help='DSFDDetector/RetinaNetResNet50')
    parser.add_argument(
        '--device', type=str, default='cuda', help='CPU/CUDA device option')
    args = parser.parse_args()
    return args

def print_most_common_label(emotions):
    LOGGER.info(f"emotion Label: {Counter(emotions).most_common(1)[0][0]}")

def Emotion(pipe):
    args = parse_args()
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
    pipe.send(True)
    while True:
        frame = pipe.recv()
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
            # 초당 프레임 수를 기준으로 1초 간격으로 감정 결과 출력
            if len(emotions_list) >= frame_rate:
                print_most_common_label(emotions_list)
                emotions_list = []

if __name__ == '__main__':
    Emotion()