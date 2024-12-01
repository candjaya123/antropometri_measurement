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
calib_file_name = 'color.txt'

def Init():
    """
    Initialization logic.
    """
    print("Initialization complete.")
    return True

def Calib(frame):
    """
    Calibration state. Allows user to adjust parameters and save settings.
    """
    cv2.namedWindow('Image Viewer')
    cv2.setMouseCallback('Image Viewer', cl.adjust_rectangle)

    while True:
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
    """
    Counting state. Processes the frame to detect and count objects.
    """
    # Load HSV ranges from calibration file
    print("hai")
    min_hue, max_hue, min_saturation, max_saturation, min_value, max_value = cr.load_hsv_ranges(calib_file_name)
    mask = cr.detect_color(frame, min_hue, max_hue, min_saturation, max_saturation, min_value, max_value)
    print("hai 2")

    # Crop the frame using the mask
    crop_frame = cr.crop_image(frame, mask)
    print("hai 3")

    # Process the cropped image to detect objects
    output_image = cn.process_image(crop_frame)
    print("hai 4")
    if output_image is not None:
        cv2.imshow('Hasil Pengukuran', output_image)
        output_path = './tes_out/hasil_deteksi.jpg'
        cv2.imwrite(output_path, output_image)
        print(f"Output saved to {output_path}.")

def main():
    """
    Main function to run the state machine.
    """
    # Load the image from file
    image_path = './Raw_data/abi.jpg'  # Replace with the actual image path
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read the image from {image_path}.")
        return

    # Resize and rotate the frame if necessary
    frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    calib_frame = frame.copy()
    pose_frame = frame.copy()
    count_frame = frame.copy()

    # Initialize state
    current_state = State.INIT

    # State machine logic
    while True:
        match current_state:
            case State.INIT:
                print("State: INIT")
                if Init():
                    current_state = State.CALIB
            case State.CALIB:
                print("State: CALIB")
                if Calib(calib_frame):
                    current_state = State.POSE
            case State.POSE:
                print("State: POSE")
                if ps.check_pose(pose_frame):
                    current_state = State.COUNT
            case State.COUNT:
                print("State: COUNT")
                Count(count_frame)
                break  # Exit after counting
            case _:
                print("Unknown state encountered!")
                break

        # Display the frame (optional)
        # cv2.imshow('Processed Image', frame)

        # Exit handling
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):  # Quit the application
            print("Exiting program.")
            break

    # Clean up resources
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
