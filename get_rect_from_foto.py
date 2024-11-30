import cv2
import numpy as np

# Load the image
image_path = "./Raw_data/abi.jpg"  # Replace with your image file path
frame = cv2.imread(image_path)

if frame is None:
    print("Error: Could not load image.")
    exit()

# Resize the image to half its original dimensions
frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

# Create a window to display the image
cv2.namedWindow('Image Viewer')

# Default values for the rectangle
rect_x, rect_y, rect_width, rect_height = 100, 100, 200, 200

# Default HSV parameters
min_hue, max_hue = 0, 0
min_saturation, max_saturation = 0, 0
min_value, max_value = 0, 0

# Create a kernel for morphology operations
kernel = np.ones((5, 5), np.uint8)

# Set the default file name
file_name = 'color.txt'

def adjust_rectangle(event, x, y, flags, param):
    """Callback to adjust the rectangle position and size."""
    global rect_x, rect_y, rect_width, rect_height
    if event == cv2.EVENT_LBUTTONDOWN:
        rect_x, rect_y = x, y
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        rect_width, rect_height = x - rect_x, y - rect_y

cv2.setMouseCallback('Image Viewer', adjust_rectangle)

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

while True:
    # Draw the adjustable rectangle on the image
    adjusted_frame = frame.copy()
    cv2.rectangle(adjusted_frame, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (0, 0, 255), 2)

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

    # Apply HSV filtering based on calculated ranges
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([min_hue, min_saturation, min_value])
    upper_bound = np.array([max_hue, max_saturation, max_value])
    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Variables to store the lowest and highest X and Y coordinates
    min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')

    # Draw bounding rectangles around contours and calculate the combined region
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(adjusted_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Update min and max coordinates
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x + w)
        max_y = max(max_y, y + h)

    # Draw the boundary
    if min_x < float('inf') and min_y < float('inf'):
        cv2.rectangle(adjusted_frame, (min_x, min_y), (max_x, max_y), (0, 255, 255), 5)

    # Display the calculated HSV parameter ranges
    # cv2.putText(adjusted_frame, f'Min Hue: {int(min_hue)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # cv2.putText(adjusted_frame, f'Max Hue: {int(max_hue)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # cv2.putText(adjusted_frame, f'Min Sat: {int(min_saturation)}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # cv2.putText(adjusted_frame, f'Max Sat: {int(max_saturation)}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # cv2.putText(adjusted_frame, f'Min Val: {int(min_value)}', (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # cv2.putText(adjusted_frame, f'Max Val: {int(max_value)}', (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the original image and mask
    cv2.imshow('Image Viewer', adjusted_frame)
    cv2.imshow('Binary Mask', mask)

    # Handle key presses
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):  # Quit
        break
    elif key & 0xFF == ord('s'):  # Save parameters
        write_file(file_name)

# Close all OpenCV windows
cv2.destroyAllWindows()
