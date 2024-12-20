import sys
import cv2
import enum
import socket
from threading import Thread
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QLabel, QWidget, QHBoxLayout, QSplitter, QSizePolicy, QGroupBox
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QObject
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
from module.style import AppStyle
from module import pose as ps
from module import crop as cr
from module import calib as cl
from module import count as cn  

ESP32_IP = "192.168.152.78"
ESP32_PORT = 4000
# State machine states
class State(enum.Enum):
    INIT = 0
    CALIB = 1
    POSE = 2
    COUNT = 3
    END = 4

# Calibration file name
calib_file_name = './data/color.txt'
captured_filename = './captured_frame.jpg'
class ESP32DataReceiver(QObject):
    data_received = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.running = True

    def connect_to_esp32(self):
        try:
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.connect((ESP32_IP, ESP32_PORT))
            print(f"Connected to ESP32 at {ESP32_IP}:{ESP32_PORT}")
            self.receive_data()
        except Exception as e:
            print(f"Failed to connect to ESP32: {e}")

    def receive_data(self):
        try:
            while self.running:
                data = self.client_socket.recv(1024)
                if data:
                    weight = data.decode("utf-8")
                    self.data_received.emit(weight)  # Emit signal with received weight
        except Exception as e:
            print(f"Error receiving data: {e}")
        finally:
            self.client_socket.close()

    def stop(self):
        self.running = False
        self.client_socket.close()

class CameraApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.current_state = State.INIT

        self.capture = cv2.VideoCapture(1)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1920)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.initUI()

        self.weight_saved = False
        # ESP32 data receiver
        self.esp32_receiver = ESP32DataReceiver()
        self.esp32_receiver.data_received.connect(self.update_weight_label)
        self.esp32_thread = Thread(target=self.esp32_receiver.connect_to_esp32, daemon=True)
        self.esp32_thread.start()


        # Mouse interaction attributes
        self.drawing = False
        self.start_point = None
        self.end_point = None


        self.mouse_x1 = 0
        self.mouse_y1 = 0
        self.mouse_x2 = 0
        self.mouse_y2 = 0

        self.height_lenght = 0
        self.hand_lenght = 0 
        self.shoulder_lenght = 0
        self.thigh_lenght = 0
        self.leg_lenght = 0

        self.save = 0

    def initUI(self):
        self.setWindowTitle("State Machine Camera App")

        # Main layout
        self.splitter = QSplitter(self)

        # Sidebar layout
        self.sidebar = QWidget()
        self.sidebar.setFixedWidth(300)
        sidebar_layout = QVBoxLayout()

        # Buttons
        self.btn_init = QPushButton("INIT", self)
        # self.btn_calib = QPushButton("CALIB", self)
        # self.btn_save_calib = QPushButton("SAVE CALIB", self)
        self.btn_save_weight = QPushButton("SAVE WEIGHT", self)
        self.btn_pose = QPushButton("POSE", self)
        self.btn_count = QPushButton("COUNT", self)
        self.btn_quit = QPushButton("QUIT", self)

        self.btn_init.clicked.connect(self.init_state)
        # self.btn_calib.clicked.connect(self.calib_state)
        # self.btn_save_calib.clicked.connect(self.save_calib)
        self.btn_save_weight.clicked.connect(self.save_weight)
        self.btn_pose.clicked.connect(self.pose_state)
        self.btn_count.clicked.connect(self.count_state)
        self.btn_quit.clicked.connect(self.close)

        button_box = QVBoxLayout()
        button_box.addWidget(self.btn_init)
        # button_box.addWidget(self.btn_calib)
        # button_box.addWidget(self.btn_save_calib)
        button_box.addWidget(self.btn_save_weight)
        button_box.addWidget(self.btn_pose)
        button_box.addWidget(self.btn_count)
        button_box.addWidget(self.btn_quit)

        button_group = QGroupBox("Controls")
        button_group.setLayout(button_box)
        sidebar_layout.addWidget(button_group)
        button_group.setStyleSheet(AppStyle("assets/css/groupbox.css").stylesheet)

        # Info display area
        info_group = QGroupBox("Info")
        info_layout = QVBoxLayout()
        self.state_label = QLabel("Current State: INIT")
        self.height_label = QLabel("height: N/A")
        self.hand_label = QLabel("hand: N/A")
        self.shoulder_label = QLabel("shoulder: N/A")
        self.thigh_label = QLabel("thigh: N/A")
        self.leg_label = QLabel("leg: N/A")
        self.weight_label = QLabel("Weight: N/A")
        info_layout.addWidget(self.state_label)
        info_layout.addWidget(self.weight_label)
        info_layout.addWidget(self.height_label)
        info_layout.addWidget(self.hand_label)
        info_layout.addWidget(self.shoulder_label)
        info_layout.addWidget(self.thigh_label)
        info_layout.addWidget(self.leg_label)
        info_group.setLayout(info_layout)
        sidebar_layout.addWidget(info_group)

        self.sidebar.setLayout(sidebar_layout)

        # Camera display
        self.camera_label = QLabel(self)
        self.camera_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.camera_label.setAlignment(Qt.AlignCenter)

        # Add widgets to the splitter
        self.splitter.addWidget(self.sidebar)
        self.splitter.addWidget(self.camera_label)

        # Set central widget
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.splitter)
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def update_frame(self):
        ret, frame = self.capture.read()
        if not ret:
            return

        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame = cv2.resize(frame, (540, 960))

        if self.current_state == State.INIT:
            image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_BGR888)
            self.camera_label.setPixmap(QPixmap.fromImage(image))

        elif self.current_state == State.CALIB:
            cl.adjust_rectangle_qt(self.mouse_x1, self.mouse_y1, self.mouse_x2, self.mouse_y2)
            calib_frame, mask = cl.process_frame(frame)
            image = QImage(calib_frame, calib_frame.shape[1], calib_frame.shape[0], QImage.Format_BGR888)
            self.camera_label.setPixmap(QPixmap.fromImage(image))

        elif self.current_state == State.POSE:
            condition, pose_frame = ps.check_pose(frame, True)
            image = QImage(pose_frame, pose_frame.shape[1], pose_frame.shape[0], QImage.Format_BGR888)
            self.camera_label.setPixmap(QPixmap.fromImage(image))
            if condition:
                self.current_state = State.COUNT

        elif self.current_state == State.COUNT:
            self.state_label.setText("Current State: COUNT")
            print("State: COUNT")
            cv2.imwrite(captured_filename, frame)
            self.current_state = State.END

        elif self.current_state == State.END:
            print("State: END")
            img = cv2.imread(captured_filename)
            self.height_lenght, self.hand_lenght, self.shoulder_lenght, self.thigh_lenght, self.leg_lenght = self.count_logic(img)
            self.height_label.setText(f"height: {self.height_lenght:.2f} cm")
            self.hand_label.setText(f"hand: {self.hand_lenght:.2f} cm")
            self.shoulder_label.setText(f"shoulder: {self.shoulder_lenght:.2f} cm")
            self.thigh_label.setText(f"thigh: {self.thigh_lenght:.2f} cm")
            self.leg_label.setText(f"leg: {self.leg_lenght:.2f} cm")

            if self.height_lenght != 0 and self.hand_lenght != 0 and self.shoulder_lenght != 0 and self.thigh_lenght != 0 and self.leg_lenght != 0:
                self.current_state = State.INIT

    def mousePressEvent(self, event):
        if self.current_state == State.CALIB and event.button() == Qt.LeftButton:
            self.drawing = True
            self.start_point = event.pos()
            self.mouse_x1, self.mouse_y1 = self.start_point.x() - 310, self.start_point.y() - 10

    def mouseMoveEvent(self, event):
        if self.drawing and self.current_state == State.CALIB:
            self.end_point = event.pos()
            self.mouse_x2, self.mouse_y2 = self.end_point.x() - 310, self.end_point.y() - 10
            self.update()

    def mouseReleaseEvent(self, event):
        if self.drawing and event.button() == Qt.LeftButton and self.current_state == State.CALIB:
            self.drawing = False
            self.end_point = event.pos()
            self.mouse_x2, self.mouse_y2 = self.end_point.x() - 310, self.end_point.y() - 10
            # Process the drawn rectangle (if needed)
            # print(f"Rectangle from {self.start_point} to {self.end_point}")

    def paintEvent(self, event):
        if self.drawing and self.start_point and self.end_point:
            painter = QPainter(self)
            pen = QPen(Qt.red, 2, Qt.SolidLine)
            painter.setPen(pen)
            rect = (self.start_point.x(), self.start_point.y(),
                    self.end_point.x() - self.start_point.x(),
                    self.end_point.y() - self.start_point.y())
            painter.drawRect(*rect)

    def update_weight_label(self, weight):
        if self.weight_saved == False:
            self.save = weight
            self.weight_label.setText(f"Weight: {weight} kg")
        else:
            self.weight_label.setText(f"Weight: {self.save} kg")

    def init_state(self):
        self.weight_saved = False
        self.current_state = State.INIT
        self.state_label.setText("Current State: INIT")
        # print("State: CALIB")
    def calib_state(self):
        self.current_state = State.CALIB
        self.state_label.setText("Current State: CALIB")
        # print("State: CALIB")

    def save_calib(self):
        cl.write_file(calib_file_name)
        print(f"Calibration parameters saved to {calib_file_name}.")

    def save_weight(self, weight):
        self.weight_saved = True
        self.weight_label.setText(f"Weight: {weight} kg")

    def pose_state(self):
        self.current_state = State.POSE
        self.state_label.setText("Current State: POSE")
        # print("State: POSE")

    def count_state(self):
        self.current_state = State.COUNT
        self.state_label.setText("Current State: COUNT")
        # print("State: COUNT")
        ret, frame = self.capture.read()
        if ret:
            cv2.imwrite(captured_filename, frame)
            img = cv2.imread(captured_filename)
            self.count_logic(img)

    def count_logic(self, frame):
        # Load HSV ranges from calibration file
        # min_hue, max_hue, min_saturation, max_saturation, min_value, max_value = cr.load_hsv_ranges(calib_file_name)
        # mask = cr.detect_color(frame, min_hue, max_hue, min_saturation, max_saturation, min_value, max_value)

        # Crop the frame using the mask
        # crop_frame = cr.crop_image(frame, mask)
        # Process the cropped image to detect objects
        output_image, height, hand, shoulder, thigh, leg  = cn.process_image(frame)
        if output_image is not None:
            # cv2.imshow('Hasil Pengukuran', output_image)
            output_path = './tes_out/hasil_deteksi.jpg'
            cv2.imwrite(output_path, output_image)
            print(f"Output saved to {output_path}.")

        return height, hand, shoulder, thigh, leg

    def closeEvent(self, event):
        self.capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(AppStyle("assets/css/style.css").stylesheet)
    main_window = CameraApp()
    main_window.show()
    sys.exit(app.exec_())
