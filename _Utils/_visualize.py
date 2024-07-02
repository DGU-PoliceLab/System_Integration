import cv2
from datetime import datetime
from _Utils.draw_action import draw

def visualize(target, pipe):
    now = datetime.now()
    timestamp = str(now).replace(" ", "").replace(":",";")
    fourcc = cv2.VideoWriter_fourcc('M','P', '4', 'V')
    out = cv2.VideoWriter(f'/System_Integration/_Output/video_clip_{target}_{timestamp}.mp4', fourcc, 30, (1280, 720))
    pipe.send(True)
    try:
        while True:
            data = pipe.recv()
            if data:
                frame, action, score = data
                frame = draw(frame, action, score)
                out.write(frame)
            else:
                break
        out.release()
    except Exception as e:
        print(f"Error occured in visualize.py: {e}")
        out.release()


