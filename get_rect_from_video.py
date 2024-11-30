import cv2
import numpy as np

# Initialize the webcam
cap = cv2.VideoCapture("demo.avi")  # Use 0 for the default webcam; you can change the index if you have multiple cameras.

# Create a window to display the camera feed
cv2.namedWindow('Camera Feed')

# Default values for the rectangle
rect_x, rect_y, rect_width, rect_height = 100, 100, 200, 200

# Default HSV parameters
min_hue, max_hue = 0, 0
min_saturation, max_saturation = 0, 0
min_value, max_value = 0, 0

# Set the default file name
file_name = 'color.txt'
def adjust_rectangle(event, x, y, flags, param):
    global rect_x, rect_y, rect_width, rect_height
    if event == cv2.EVENT_LBUTTONDOWN:
        rect_x, rect_y = x, y
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        rect_width, rect_height = x - rect_x, y - rect_y

min_hue_temp = min_hue
min_saturation_temp = min_saturation
min_value_temp = min_value
max_hue_temp = max_hue
max_saturation_temp = max_saturation
max_value_temp = max_value
cv2.setMouseCallback('Camera Feed', adjust_rectangle)
kernel = np.ones((5, 5), np.uint8)

def write_file(param_file_name):
    with open(param_file_name, 'w') as param_file:
        param_file.write(f"min_hue={min_hue}\n")
        param_file.write(f"max_hue={max_hue}\n")
        param_file.write(f"min_saturation={min_saturation}\n")
        param_file.write(f"max_saturation={max_saturation}\n")
        param_file.write(f"min_value={min_value}\n")
        param_file.write(f"max_value={max_value}\n")
while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Draw the adjustable rectangle on the frame
    adjusted_frame = frame.copy()
    cv2.rectangle(adjusted_frame, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (0, 0, 255), 2)

    # Display the frame with the adjustable rectangle
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_bound = np.array([min_hue_temp, min_saturation_temp, min_value_temp])
    upper_bound = np.array([max_hue_temp, max_saturation_temp, max_value_temp])

    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
    mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    inverted_frame = cv2.bitwise_not(mask)

    object_frame = cv2.bitwise_and(frame, frame, mask=mask)
    # Extract the region of interest (ROI) within the rectangle
    roi = frame[rect_y:rect_y + rect_height, rect_x:rect_x + rect_width]

    if roi.shape[0] > 0 and roi.shape[1] > 0:  # Check if the ROI is non-empty
        # Calculate the minimum and maximum HSV values within the ROI
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        min_hue = np.min(hsv_roi[:, :, 0])
        max_hue = np.max(hsv_roi[:, :, 0])
        min_saturation = np.min(hsv_roi[:, :, 1])
        max_saturation = np.max(hsv_roi[:, :, 1])
        min_value = np.min(hsv_roi[:, :, 2])
        max_value = np.max(hsv_roi[:, :, 2])

    # Display the calculated HSV parameter ranges
    # cv2.putText(adjusted_frame, f'Min Hue: {int(min_hue)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # cv2.putText(adjusted_frame, f'Max Hue: {int(max_hue)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # cv2.putText(adjusted_frame, f'Min Saturation: {int(min_saturation)}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # cv2.putText(adjusted_frame, f'Max Saturation: {int(max_saturation)}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # cv2.putText(adjusted_frame, f'Min Value: {int(min_value)}', (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # cv2.putText(adjusted_frame, f'Max Value: {int(max_value)}', (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(adjusted_frame, f'color: {str(file_name)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Camera Feed', adjusted_frame)
    cv2.imshow('Binary Mask', mask)
    # cv2.imshow('Inverted Frame', inverted_frame)
    # cv2.imshow('Processed Binary Mask', mask_closed)

    # Exit the loop when the 'q' key is pressed
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('s'):
        write_file(file_name)

        min_hue_temp = min_hue
        min_saturation_temp = min_saturation
        min_value_temp = min_value

        max_hue_temp = max_hue
        max_saturation_temp = max_saturation
        max_value_temp = max_value
        
# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()