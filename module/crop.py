import cv2
import numpy as np

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
