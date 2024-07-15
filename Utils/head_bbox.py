def calculate_head_box(keypoints, bbox):
    # Keypoints indices for eyes, nose, mouth, ears, shoulders
    right_eye = keypoints[0]
    left_eye = keypoints[1]
    nose = keypoints[2]
    right_mouth = keypoints[3]
    left_mouth = keypoints[4]
    right_ear = keypoints[5]
    left_ear = keypoints[6]
    right_shoulder = keypoints[7]
    left_shoulder = keypoints[8]

    # Calculate the line between the shoulders
    shoulder_line = (right_shoulder, left_shoulder)
    shoulder_slope = (left_shoulder[1] - right_shoulder[1]) / (left_shoulder[0] - right_shoulder[0]) if (left_shoulder[0] - right_shoulder[0]) != 0 else 0
    shoulder_intercept = right_shoulder[1] - shoulder_slope * right_shoulder[0]

    # Function to calculate intersection with the shoulder line
    def intersect_with_shoulder(x):
        return shoulder_slope * x + shoulder_intercept

    # Calculate vertical lines from ears to the shoulder line
    right_ear_intersection = (right_ear[0], intersect_with_shoulder(right_ear[0]))
    left_ear_intersection = (left_ear[0], intersect_with_shoulder(left_ear[0]))

    # Calculate head box
    x_min = min(right_ear[0], left_ear[0])
    y_min = min(right_ear_intersection[1], left_ear_intersection[1], bbox[0][1])
    x_max = max(right_ear[0], left_ear[0])
    y_max = max(right_ear[1], left_ear[1], nose[1])

    return [x_min, y_min, x_max, y_max]
