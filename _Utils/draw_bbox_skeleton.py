import cv2

COLOR = (0,255,0)
WHITE = (255, 255, 255)
THICK = 1
RADIUS = 3

def draw(frame, id, bbox, skeleton):
    frame = draw_bbox(frame, id, bbox)
    frame = draw_skeleton(frame, skeleton)
    return frame

def draw_bbox(frame, id, bbox):
    x1 = int(bbox[0])
    y1 = int(bbox[1])
    x2 = int(bbox[0] + bbox[2])
    y2 = int(bbox[1] + bbox[3])
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR, THICK)
    frame = draw_id_box(frame, id, x1, y1)
    return frame

def draw_id_box(frame, id, x, y):
    text_size, _ = cv2.getTextSize(str(id), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    text_w, text_h = text_size
    cv2.rectangle(frame, (x, y), (x + text_w, y + text_h), COLOR, -1)
    cv2.putText(frame, str(id), (x, y + text_h), cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 2)
    return frame

def draw_skeleton(frame, skeleton):
    frame = draw_keypoint(frame, skeleton)
    frame = draw_keypoint_connection(frame, skeleton)
    return frame

def draw_keypoint(frame, skeleton):
    cnt = 0
    for keypoint in skeleton:
        x = int(keypoint[0])
        y = int(keypoint[1])
        cv2.circle(frame, (x, y), RADIUS, COLOR, -1)
        # cv2.putText(frame, str(cnt), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cnt+=1
    return frame

def draw_keypoint_connection(frame, skeleton):
    keypoint = define_keypoint(len(skeleton))
    if len(skeleton) == 17:
        connection = [[0, 1], [0, 2], [0, 5], [0, 6], [1,3], [2,4], [5, 7], [6,8], [7,9],[8,10], [5,11], [6,12], [11,12], [11,13], [12,14], [13,15], [14, 16]]
        for connect in connection:
            start, end = connect
            x1 = int(skeleton[start][0])
            y1 = int(skeleton[start][1])
            x2 = int(skeleton[end][0])
            y2 = int(skeleton[end][1])
            cv2.line(frame, (x1, y1), (x2, y2), COLOR, THICK)
    return frame


def define_keypoint(n):
    desc_keypoint = {}
    if n == 17:
        desc_keypoint = {
            'nose': 0,
            'right_eye': 1,
            'left_eye': 2,
            'right_ear': 3,
            'left_ear': 4,
            'right_shoulder': 5,
            'left_shoulder': 6,
            'right_elbow': 7,
            'left_elbow': 8,
            'right_wrist': 9,
            'left_wrist': 10,
            'right_pelvis': 11,
            'left_pelvis': 12,
            'right_knee': 13,
            'left_knee': 14,
            'right_ankle': 15,
            'left_ankle': 16
        }
    return desc_keypoint

        