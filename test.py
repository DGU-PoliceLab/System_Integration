import os
import cv2

path = "/System_Integration/_Output/2024-07-1209-15-15-737356/selfharm"

paths = []
if os.path.isfile(os.path.join(path, "0.png")):
    images = os.listdir(path)
    images.sort(key=lambda x: int(x.split(".")[0]))
    for image in images:
        paths.append(os.path.join(path, image))

    img = cv2.imread(paths[0])
    height, width, layers = img.shape
    size = (width,height)

    out = cv2.VideoWriter("/System_Integration/_Output/2024-07-1209-15-15-737356/selfharm.mp4", cv2.VideoWriter_fourcc('M','P','4','V'), 30, size)
    for path in paths:
        frame = cv2.imread(path)
        out.write(frame)

    out.release()