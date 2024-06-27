import cv2
import os
import face_detection
import torch
from PIL import Image
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_filename', default='out.avi') # 경로수정 필요
    parser.add_argument('--test_video', type=str, default='distancedata0_video.mp4', help='distancedata0_video.mp4') #비디오 인풋입니다.
    parser.add_argument('--face_detector', type=str, default='RetinaNetResNet50', help='DSFDDetector/RetinaNetResNet50')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    detector = face_detection.build_detector(args.face_detector, confidence_threshold=.5, nms_iou_threshold=.3)

    cap = cv2.VideoCapture(os.path.join(os.getcwd(), args.test_video))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(args.out_filename, fourcc, 30.0, (1280, 720))

    output_folder = os.path.join(os.getcwd(), '_out')
    os.makedirs(output_folder, exist_ok=True)

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)

        for i in range(detections.shape[0]):
            x1, y1, x2, y2 = detections[i][0:4]
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            face_img = frame[y1:y2, x1:x2, :]
            face_filename = os.path.join(output_folder, f"frame_{frame_idx}_face_{i}.png")
            cv2.imwrite(face_filename, face_img)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow('face detector', frame)
        out.write(frame)

        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

        frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # GPU 메모리 사용량 출력(없어도 됩니다.)
    if torch.cuda.is_available():
        print(f"최종 GPU 메모리 사용량: {torch.cuda.memory_allocated(device) / (1024 ** 2):.2f} MB")
        print(f"최종 GPU 메모리 캐시: {torch.cuda.memory_reserved(device) / (1024 ** 2):.2f} MB")
    else:
        print("CUDA를 사용할 수 없습니다. CPU 메모리 사용량은 측정하지 않습니다.")

if __name__ == '__main__':
    main()

