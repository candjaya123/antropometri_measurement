import cv2
import mediapipe as mp
from . import tool as get

# Global variables to store results of landmark processing
Shoulder = False
Elbow = False
Knee = False
Leg = False

Lshoulder_slope = 0
Rshoulder_slope = 0
Lelbow_slope = 0
Relbow_slope = 0
Lknee_slope = 0
Rknee_slope = 0 
leg_distance = 0

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Helper function to calculate necessary metrics
def process_landmarks(landmarks, w, h):
    global Lshoulder_slope, Rshoulder_slope, Lelbow_slope, Relbow_slope, Lknee_slope, Rknee_slope, leg_distance

    # Left and right landmarks
    LShoulder = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y * h]
    RShoulder = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
    LElbow = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].y * h]
    RElbow = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value].y * h]
    LHand = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].y * h]
    RHand = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value].y * h]

    LHip = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].y * h]
    RHip = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].y * h]
    LKnee = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].y * h]
    RKnee = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value].y * h]
    LAnkle = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].y * h]
    RAnkle = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value].y * h]

    # Calculate distances and slopes
    Lshoulder_slope = get.Angle(LElbow,LShoulder,LHip)
    Rshoulder_slope = get.Angle(RElbow,RShoulder,RHip)
    Lelbow_slope = get.Angle(LHand,LElbow,LShoulder)
    Relbow_slope = get.Angle(RHand,RElbow,RShoulder)
    Lknee_slope = get.Angle(LHip,LKnee,LAnkle)
    Rknee_slope = get.Angle(RHip,RKnee,RAnkle)
    leg_distance = get.Distance(LAnkle,RAnkle)

def check_pose(frame, button):
    global Shoulder, Elbow, Knee, Leg
    # Get frame dimensions
    adjusted_frame = frame.copy()
    h, w = adjusted_frame.shape[:2]

    # Recolor the image to RGB
    image_rgb = cv2.cvtColor(adjusted_frame, cv2.COLOR_BGR2RGB)

    # Set up MediaPipe Pose
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        # Process landmarks using MediaPipe Pose
        results = pose.process(image_rgb)

        if button == False:
            Shoulder = False
            Elbow = False 
            Knee = False
            Leg = False
        
        if results.pose_landmarks:
            process_landmarks(results.pose_landmarks.landmark, w, h)

            # Use mp_drawing to draw the landmarks
            mp_drawing.draw_landmarks(
                adjusted_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(66, 245, 230), thickness=2, circle_radius=2)
            )

        # face = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE.value].visibility > 0.8
        # body = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility > 0.8
        # leg = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility > 0.8
        # ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value].visibility > 0.8

        # if face == False and body == False and leg == False and ankle == False:
        #     Shoulder = False
        #     Elbow = False 
        #     Knee = False
        #     Leg = False

        if Lshoulder_slope < 80:
            Shoulder = False
            cv2.putText(adjusted_frame, f"bahu kiri kurang naik: {(80 - Lshoulder_slope):.2f}", (0, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        elif Rshoulder_slope < 80:
            Shoulder = False
            cv2.putText(adjusted_frame, f"bahu kanan kurang naik: {(80 - Rshoulder_slope):.2f}", (0, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        elif Lshoulder_slope > 80 and Rshoulder_slope > 80:
            Shoulder = True
            cv2.putText(adjusted_frame, f"bahu pas", (0, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        if Lelbow_slope < 165:
            Elbow = False
            cv2.putText(adjusted_frame, f"siku kiri kurang lurus: {(165 - Lelbow_slope):.2f}", (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        elif Relbow_slope < 165:
            Elbow = False
            cv2.putText(adjusted_frame, f"siku kanan kurang lurus: {(165 - Relbow_slope):.2f}", (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        elif Lelbow_slope > 165 and Relbow_slope > 165:
            Elbow = True
            cv2.putText(adjusted_frame, f"siku pas", (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if Lknee_slope < 170:
            Knee = False
            cv2.putText(adjusted_frame, f"lutut kiri kurang lurus: {(170 - Lelbow_slope):.2f}", (0, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        elif Rknee_slope < 170:
            Knee = False
            cv2.putText(adjusted_frame, f"lutut kanan kurang lurus: {(170 - Relbow_slope):.2f}", (0, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        elif Rknee_slope > 170 and Lknee_slope > 170:
            Knee = True
            cv2.putText(adjusted_frame, f"lutut pas", (0, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if leg_distance < 40 :   
            Leg = False
            cv2.putText(adjusted_frame, f"lebarkan kaki", (0, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        elif leg_distance > 40: 
            Leg = True
            cv2.putText(adjusted_frame, f"lebar kaki pas", (0, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    if Shoulder and Elbow and Knee and Leg:
        print('TRUUUUEEEEEE')
        return True, adjusted_frame
    elif Shoulder == False and Elbow == False and Knee == False and Leg == False:
        print('FALSEEEEE')
        return False, adjusted_frame
    
    return False, adjusted_frame