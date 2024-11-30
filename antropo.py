import cv2
import numpy as np
import mediapipe as mp

def load_hsv_ranges(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        min_hue_temp = int(lines[0].split('=')[1])
        max_hue_temp = int(lines[1].split('=')[1])
        min_saturation_temp = int(lines[2].split('=')[1])
        max_saturation_temp = int(lines[3].split('=')[1])
        min_value_temp = int(lines[4].split('=')[1])
        max_value_temp = int(lines[5].split('=')[1])
    return min_hue_temp, max_hue_temp, min_saturation_temp, max_saturation_temp, min_value_temp, max_value_temp

def detect_color(frame, min_hue, max_hue, min_saturation, max_saturation, min_value, max_value):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    kernel = np.ones((5, 5), np.uint8)
    lower_bound = np.array([min_hue, min_saturation, min_value])
    upper_bound = np.array([max_hue, max_saturation, max_value])
    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def crop_image(main_frame, mask):
    frame = main_frame.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found!")
        return None
    
    min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(main_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x + w)
        max_y = max(max_y, y + h)
    
    if min_x < float('inf') and min_y < float('inf'):
        cv2.rectangle(main_frame, (min_x, min_y), (max_x, max_y), (0, 255, 255), 5)
        return frame[min_y:max_y+50, min_x:max_x]
    return None

def pose(landmarks, w, h):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Left side landmarks
    LHip  = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].y * h]
    LKnee = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].y * h]
    LAnkle = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].y * h]

    # Right side landmarks
    RHip  = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].y * h]
    RKnee = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value].y * h]
    RAnkle = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value].y * h]

    # Shoulder landmarks
    LShoulder = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y * h]
    RShoulder = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
    
    # Elbow landmarks
    LElbow = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].y * h]
    RElbow = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value].y * h]
    
    # Elbow landmarks
    LWrist = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].y * h]
    RWrist = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value].y * h]

    return False

def main():
    image_path = "./Raw_data/abi.jpg"  # Replace with your image file path
    frame = cv2.imread(image_path)
    if frame is None:
        print("Error: Could not load image.")
        return

    frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE) 

    min_hue_temp, max_hue_temp, min_saturation_temp, max_saturation_temp, min_value_temp, max_value_temp = load_hsv_ranges("color.txt")
    mask = detect_color(frame, min_hue_temp, max_hue_temp, min_saturation_temp, max_saturation_temp, min_value_temp, max_value_temp)
    crop_frame = crop_image(frame, mask)



    cv2.imshow('Image Viewer', frame)
    cv2.imshow('Binary Mask', mask)
    if crop_frame is not None:
        cv2.imshow('Cropped Image', crop_frame)

    cv2.waitKey(0) 
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()  
