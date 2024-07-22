import os
import shutil
import cv2
from Utils.draw_action import draw
from variable import get_debug_args

class Visualizer():
    
    def __init__(self, action):
        self.args = get_debug_args()
        self.action = action
        self.frame_no = 0
        self.timestamp = None
        self.output = self.args.output
        self.path = None
        
    def mkdir(self, timestamp):
        self.path = os.path.join(self.args.output, str(timestamp), self.action)
        if not os.path.isdir(self.path):
            self.timestamp = timestamp
            os.mkdir(self.path)

    def save_temp_image(self, data, frame_no = -1, color="default"):
        if frame_no == -1:
            frame_no = self.frame_no
            self.frame_no += 1
        if data:
            draw_frame, action, score = data
            if action != None or score != None:
                image = draw(draw_frame, action, score, color=color)
            else:
                image = draw_frame
            cv2.imwrite(os.path.join(self.path, f"{frame_no}.png"), image)

    def merge_img_to_video(self):
        paths = []
        try:
            if os.path.isfile(os.path.join(self.path, "0.png")):
                images = os.listdir(self.path)
                images.sort(key=lambda x: int(x.split(".")[0]))
                for image in images:
                    paths.append(os.path.join(self.path, image))

            img = cv2.imread(paths[0])
            height, width, layers = img.shape
            size = (width,height)
            
            out = cv2.VideoWriter(os.path.join(self.output, self.timestamp, f"{self.action}.mp4"), cv2.VideoWriter_fourcc('m','p','4','v'), 30, size)
            for path in paths:
                frame = cv2.imread(path)
                out.write(frame)

            out.release()
            # shutil.rmtree(self.path)
        except:
            pass