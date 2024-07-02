import cv2

COLOR = (0,255,0)
WHITE = (255, 255, 255)
THICK = 1
RADIUS = 3

def draw(frame, action, score):
    text = f'{action}: {score:.2f}'
    frame = draw_result(frame, text, 20, 20)
    return frame

def draw_result(frame, text, x, y):
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    text_w, text_h = text_size
    cv2.rectangle(frame, (x, y), (x + text_w, y + text_h), COLOR, -1)
    cv2.putText(frame, text, (x, y + text_h), cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 2)
    return frame