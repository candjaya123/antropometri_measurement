import cv2
import mediapipe as mp
import numpy as np

# Inisialisasi Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Fungsi untuk menghitung jarak pixel antara dua landmark
def calculate_pixel_distance(landmark1, landmark2, image_width, image_height):
    x1, y1 = int(landmark1[0] * image_width), int(landmark1[1] * image_height)
    x2, y2 = int(landmark2[0] * image_width), int(landmark2[1] * image_height)
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Fungsi untuk mengonversi dari pixel ke cm menggunakan hasil regresi
def convert_pixel_to_cm(pixel_value, a=0.002, b=0.5, c=1.0):
    print(pixel_value)
    return a * (pixel_value ** 2) + b * pixel_value + c

# Fungsi untuk memproses satu gambar dan menghitung pengukuran
def process_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
        image_height, image_width, _ = image.shape
        
        # Hitung tinggi badan (rata-rata dari NOSE ke LEFT_HEEL dan NOSE ke RIGHT_HEEL)
        height_left = calculate_pixel_distance(landmarks[mp_pose.PoseLandmark.NOSE.value],
                                               landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value],
                                               image_width, image_height)
        height_right = calculate_pixel_distance(landmarks[mp_pose.PoseLandmark.NOSE.value],
                                                landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value],
                                                image_width, image_height)
        height_pixel = (height_left + height_right) / 2
        height_cm = convert_pixel_to_cm(height_pixel,0,0.1689,32.619)

        # Hitung lebar tangan (rata-rata dari LEFT_SHOULDER ke LEFT_WRIST dan RIGHT_SHOULDER ke RIGHT_WRIST)
        left_hand_span = calculate_pixel_distance(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value],
                                                  image_width, image_height)
        right_hand_span = calculate_pixel_distance(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value],
                                                   image_width, image_height)
        hand_span_pixel = (left_hand_span + right_hand_span) / 2
        hand_span_cm = convert_pixel_to_cm(hand_span_pixel,0,0.1905,13.772)

        # Hitung lebar bahu (dari bahu kiri ke bahu kanan)
        shoulder_width_pixel = calculate_pixel_distance(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value], 
                                                        image_width, image_height)
        shoulder_width_cm = convert_pixel_to_cm(shoulder_width_pixel,0,0.1022,20.933)

        # Hitung panjang paha (rata-rata paha kiri dan paha kanan)
        left_thigh_length = calculate_pixel_distance(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                                     image_width, image_height)
        right_thigh_length = calculate_pixel_distance(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                                      image_width, image_height)
        thigh_length_pixel = (left_thigh_length + right_thigh_length) / 2
        thigh_length_cm = convert_pixel_to_cm(thigh_length_pixel,0,0.1027,26.626)

        # Hitung panjang kaki dari hip ke ankle (rata-rata kaki kiri dan kaki kanan)
        left_leg_length = calculate_pixel_distance(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                                   landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value],
                                                   image_width, image_height)
        right_leg_length = calculate_pixel_distance(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                                    landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value],
                                                    image_width, image_height)
        leg_length_pixel = (left_leg_length + right_leg_length) / 2
        leg_length_cm = convert_pixel_to_cm(leg_length_pixel,0,0.1343,44.799)

        rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        original_height, original_width = rotated_image.shape[:2]
    
         # Mengubah ukuran gambar menjadi setengah dari resolusi asli
        half_size_image = cv2.resize(rotated_image, (original_width // 2, original_height // 2))


        # Tampilkan hasil pengukuran pada gambar
        cv2.putText(half_size_image, f'Height: {height_cm:.2f} cm', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(half_size_image, f'Hand Span: {hand_span_cm:.2f} cm', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(half_size_image, f'Shoulder Width: {shoulder_width_cm:.2f} cm', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(half_size_image, f'Thigh Length: {thigh_length_cm:.2f} cm', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(half_size_image, f'leg Length: {leg_length_cm:.2f} cm', (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        return half_size_image
    else:
        print("Pose tidak terdeteksi.")
        return None

# Path gambar yang akan diproses
image_path = './Raw_data/Sultan.jpg'  # Ganti dengan path gambar

# Proses gambar
output_image = process_image(image_path)

# Tampilkan atau simpan gambar hasil deteksi
if output_image is not None:
    cv2.imshow('Hasil Pengukuran', output_image)
    cv2.waitKey(0)  # Tekan tombol apa saja untuk menutup jendela
    cv2.destroyAllWindows()

    # Jika ingin menyimpan gambar, gunakan perintah berikut
    cv2.imwrite('hasil_deteksi.jpg', output_image)
