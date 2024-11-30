import cv2
import numpy as np

# Load the image
image_path = './Raw_data/abi.jpg'
image = cv2.imread(image_path)

# Resize the image to half of its original dimensions
height, width = image.shape[:2]
resized_image = cv2.resize(image, (width // 2, height // 2))

# Rotate the resized image 90 degrees clockwise
rotated_image = cv2.rotate(resized_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

# Convert to grayscale
gray = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)

# Apply GaussianBlur to reduce noise and improve edge detection
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Use Canny edge detector to detect edges
edges = cv2.Canny(blurred, 50, 150)

# Find contours based on edges
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours by area to get the largest rectangle (assuming the black line forms the largest rectangle)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
black_line_contour = contours[0] if contours else None

# Draw the detected contour on the rotated image (for visualization)
output_image = rotated_image.copy()
if black_line_contour is not None:
    cv2.drawContours(output_image, [black_line_contour], -1, (0, 255, 0), 2)

# Get the bounding box of the black line and crop the image
if black_line_contour is not None:
    x, y, w, h = cv2.boundingRect(black_line_contour)
    cropped_image = rotated_image[y:y+h, x:x+w]

    # Show the cropped image
    cv2.imshow('Cropped Image', cropped_image)
    cv2.imshow('Detected Black Line', output_image)
else:
    print("Black line not detected.")

# Display the edges for reference
cv2.imshow('Edges', edges)

# Wait for a key press and close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
