import cv2
import enum
from module import pose as ps
from module import crop as cr
from module import calib as cl
from module import count as cn

# State machine states
class State(enum.Enum):
    INIT = 0
    CALIB = 1
    POSE = 2
    COUNT = 3
    RESULT = 4

# Calibration file name
calib_file_name = './data/color.txt'
captured_filename = './captured_frame.jpg'

count_time = False

def Init():
    print("Initialization complete camera ready.")
    return True

def Calib(frame):
    cv2.namedWindow('Image Viewer')
    cv2.setMouseCallback('Image Viewer', cl.adjust_rectangle)

    # Process the frame for calibration
    adjusted_frame, mask = cl.process_frame(frame)

    # Display frames
    cv2.imshow('Image Viewer', adjusted_frame)
    cv2.imshow('Binary Mask', mask)
    
    # Handle key presses
    key = cv2.waitKey(1)
    if key & 0xFF == ord('t'):  # Exit calibration
        return True
    elif key & 0xFF == ord('s'):  # Save calibration parameters
        cl.write_file(calib_file_name)
        print(f"Calibration parameters saved to {calib_file_name}.")
    elif key & 0xFF == ord('q'):  # Quit without saving
        print("Calibration aborted.")
        return False

def Count(frame):
    # Load HSV ranges from calibration file
    min_hue, max_hue, min_saturation, max_saturation, min_value, max_value = cr.load_hsv_ranges(calib_file_name)
    mask = cr.detect_color(frame, min_hue, max_hue, min_saturation, max_saturation, min_value, max_value)

    # Crop the frame using the mask
    crop_frame = cr.crop_image(frame, mask)

    # Process the cropped image to detect objects
    output_image = cn.process_image(crop_frame)
    if output_image is not None:
        cv2.imshow('Hasil Pengukuran', output_image)
        output_path = './tes_out/hasil_deteksi.jpg'
        cv2.imwrite(output_path, output_image)
        print(f"Output saved to {output_path}.")

def main():
    global count_time
    # Capture video from the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    # Initialize state
    current_state = State.INIT

    # State machine logic
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame from webcam.")
            break
        # Resize the frame for consistent processing
        frame = cv2.resize(frame, (540, 960))
        calib_frame = frame.copy()
        pose_frame = frame.copy()

        match current_state:
            case State.INIT:
                print("State: INIT")
                Init()
                current_state = State.CALIB
            case State.CALIB:
                print("State: CALIB")
                Calib(calib_frame)
            case State.POSE:
                print("State: POSE")
                if ps.check_pose(pose_frame):
                    count_time = True
                    break
            case _:
                print("Unknown state encountered!")
                break

        # Display the frame (optional)
        cv2.imshow('Live Feed', frame)

        # Exit handling
        key = cv2.waitKey(1) & 0xFF

        if key == ord('1'):
            current_state = State.POSE

        if count_time or key == ord('2'):
            cv2.imwrite(captured_filename, frame)
            img = cv2.imread(captured_filename)
            Count(img)
    
        elif key == ord('q'):
            print("Quitting...")
            break


    # Clean up resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
