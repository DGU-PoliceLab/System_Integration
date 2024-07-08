import cv2

COLOR = (0,255,0)
WHITE = (255, 255, 255)
THICK = 1
RADIUS = 3

def draw(frame, x, y, heartbeat, breath):
    heartbeat_text = f"h: {heartbeat}"
    breath_text = f"b: {breath}"
    text_size_1, _ = cv2.getTextSize(str(heartbeat_text), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    text_size_2, _ = cv2.getTextSize(str(breath_text), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    _, text_h_1 = text_size_1
    _, text_h_2 = text_size_2
    y1 = int(y + (text_h_1 * 2))
    y2 = int(y + (text_h_1 * 2 + text_h_2))
    cv2.putText(frame, str(heartbeat_text), (x, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR, 2)
    cv2.putText(frame, str(breath_text), (x, y2), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR, 2)
    return frame