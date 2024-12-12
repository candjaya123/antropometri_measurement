import cv2
import os
from module import crop as cr
from module import count as cn

# Calibration file name
calib_file_name = './color.txt'

def Init():
    print("Initialization complete.")
    return True

def Count(output_path, image_path):

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read the image from {image_path}.")
        return
    
    frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # # Load HSV ranges from calibration file
    min_hue, max_hue, min_saturation, max_saturation, min_value, max_value = cr.load_hsv_ranges(calib_file_name)
    mask = cr.detect_color(frame, min_hue, max_hue, min_saturation, max_saturation, min_value, max_value)

    # Crop the frame using the mask
    crop_frame = cr.crop_image(frame, mask)

    # Process the cropped image to detect objects
    output_image = cn.process_image(crop_frame)
    if output_image is not None:
        cv2.imshow('Hasil Pengukuran', output_image)
        output_path = os.path.join(output_path, os.path.basename(image_path))
        cv2.imwrite(output_path, output_image)
        print(f"Output saved to {output_path}.")

def main():
    # Folder containing images
    input_folder = './Raw_data'
    output_folder = './Hasil_Deteksi_crop'
    os.makedirs(output_folder, exist_ok=True)

    # Initialize state
    if not Init():
        print("Initialization failed.")
        return

    # Iterate through all images in the folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            print(f"Processing {image_path}...")
            Count(output_folder,image_path)

    # Clean up resources
    cv2.destroyAllWindows()
    print("Processing complete.")

if __name__ == "__main__":
    main()
