import numpy as np

def calculate_head_box(keypoints, bbox):
    nose = keypoints[2]
    right_ear = keypoints[5]
    left_ear = keypoints[6]

   # Ensure the box is within the image boundaries
    x_min = max(0, nose[0] - 100)
    y_min = max(0, nose[1] - 100)
    x_max = max(0, nose[0] + 100)
    y_max = max(0, nose[1] + 100)


    return [x_min, y_min, x_max, y_max]


def tmep_calculate_head_box(keypoints, bbox, frame_shape):
    nose = keypoints[2]
    right_ear = keypoints[5]
    left_ear = keypoints[6]

    # Calculate the distance from the nose to the farthest ear
    distance_right_ear = np.linalg.norm(np.array(nose) - np.array(right_ear))
    distance_left_ear = np.linalg.norm(np.array(nose) - np.array(left_ear))

    farthest_ear_distance = max(distance_right_ear, distance_left_ear)
    half_distance = farthest_ear_distance

    # Calculate the top and bottom y-coordinates of the head box
    y_min = bbox[1]
    y_max = int(nose[1] + abs(nose[1] - y_min))

    # Calculate the left and right x-coordinates of the head box
    x_min = int(nose[0] - half_distance)
    x_max = int(nose[0] + half_distance)

    # Ensure the box is within the image boundaries
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(frame_shape[1], x_max)
    y_max = min(frame_shape[0], y_max)

    return [x_min, y_min, x_max, y_max]
