import cv2
import numpy as np

# Global variables for rectangle and HSV parameters
rect_x, rect_y, rect_width, rect_height = 100, 100, 200, 200
min_hue, max_hue = 0, 0
min_saturation, max_saturation = 0, 0
min_value, max_value = 0, 0
kernel = np.ones((5, 5), np.uint8)
file_name = './data/color.txt'

def adjust_rectangle_qt(x1, y1, x2, y2):
    global rect_x, rect_y, rect_width, rect_height
    rect_x = x1
    rect_y = y1
    rect_width = x2 - x1
    rect_height = y2 - y1

def adjust_rectangle(event, x, y, flags, param):
    """Callback to adjust the rectangle position and size."""
    global rect_x, rect_y, rect_width, rect_height
    if event == cv2.EVENT_LBUTTONDOWN:
        rect_x, rect_y = x, y
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        rect_width, rect_height = x - rect_x, y - rect_y

def write_file(param_file_name):
    """Save HSV parameters to a file."""
    with open(param_file_name, 'w') as param_file:
        param_file.write(f"min_hue={min_hue}\n")
        param_file.write(f"max_hue={max_hue}\n")
        param_file.write(f"min_saturation={min_saturation}\n")
        param_file.write(f"max_saturation={max_saturation}\n")
        param_file.write(f"min_value={min_value}\n")
        param_file.write(f"max_value={max_value}\n")
    print(f"Parameters saved to {param_file_name}")

def process_frame(frame):
    """Process the current frame and return the adjusted frame and mask."""
    global min_hue, max_hue, min_saturation, max_saturation, min_value, max_value

    # Draw adjustable rectangle
    adjusted_frame = frame.copy()
    cv2.rectangle(adjusted_frame, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (0, 0, 255), 2)

    # Extract ROI and calculate HSV range
    roi = frame[rect_y:rect_y + rect_height, rect_x:rect_x + rect_width]
    if roi.shape[0] > 0 and roi.shape[1] > 0:
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        min_hue, max_hue = np.min(hsv_roi[:, :, 0]), np.max(hsv_roi[:, :, 0])
        min_saturation, max_saturation = np.min(hsv_roi[:, :, 1]), np.max(hsv_roi[:, :, 1])
        min_value, max_value = np.min(hsv_roi[:, :, 2]), np.max(hsv_roi[:, :, 2])

    # Apply mask based on HSV range
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([min_hue, min_saturation, min_value])
    upper_bound = np.array([max_hue, max_saturation, max_value])
    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Draw contours and calculate boundary
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(adjusted_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        min_x, min_y = min(min_x, x), min(min_y, y)
        max_x, max_y = max(max_x, x + w), max(max_y, y + h)
    if min_x < float('inf') and min_y < float('inf'):
        cv2.rectangle(adjusted_frame, (min_x, min_y), (max_x, max_y), (0, 255, 255), 5)

    return adjusted_frame, mask
