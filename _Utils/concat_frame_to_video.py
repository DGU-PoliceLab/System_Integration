import cv2
import os
import glob
import ffmpegcv

OUTPUT_DIR = '_Output'

def save_vid_clip(timestamp):
    # 영상으로 만들고 싶은 파일들이 있는 폴더 경로 지정
    folder_path = os.path.join(OUTPUT_DIR, timestamp) # 여기에 이지미를 모으고 비디오를 만든 후에 _vidtemp폴더를 초기화
    # 폴더 내 모든 jpg 파일 목록 가져오기
    img_paths = glob.glob(os.path.join(folder_path, '*.jpg'))

    # 정렬하는 부분 (파일명 순으로 정렬할 수 있도록)
    img_paths.sort()

    # 영상 저장 설정
    video_name = os.path.join(OUTPUT_DIR, 'output_video.mp4')
    frame = cv2.imread(img_paths[0])
    height, width, layers = frame.shape
    video = ffmpegcv.VideoWriter(video_name, "h264")


    # 이미지들을 영상으로 만들기
    for img_path in img_paths:
        img = cv2.imread(img_path)
        video.write(img)

    cv2.destroyAllWindows()
    video.release()