import cv2
import base64

def frame_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    base64_str = base64.b64encode(buffer)
    base64_img = "data:image/jpg;base64," + str(base64_str).replace("b'", "").replace("'","")
    return base64_img

def extract_face(frame, bbox):
    face = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    base64_face = frame_to_base64(face)
    return base64_face