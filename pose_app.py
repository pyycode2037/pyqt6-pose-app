import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QGroupBox
from PyQt6.QtGui import QImage, QPixmap, QPainter
from PyQt6.QtCore import QTimer, Qt
import mediapipe as mp

class PoseEstimationWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("姿態估計應用程式")
        self.image_label = QLabel()
        self.result_label = QLabel()
        self.info_label = QLabel()
        img_layout = QHBoxLayout()
        img_layout.addWidget(self.image_label)
        img_layout.addWidget(self.result_label)

        # 新增影像上方說明列
        label_layout = QHBoxLayout()
        self.label_origin = QLabel('原始影像')
        self.label_result = QLabel('預測結果')
        self.label_origin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_result.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label_layout.addWidget(self.label_origin)
        label_layout.addWidget(self.label_result)

        layout = QVBoxLayout()
        layout.addWidget(self.info_label)
        layout.addLayout(label_layout)  # 插入說明列
        layout.addLayout(img_layout)
        self.setLayout(layout)

        # 新增 frame 大小設定
        self.frame_width = 960  # 預設寬度，可自行調整
        self.frame_height = 720 # 預設高度，可自行調整

        # 新增輸入模式選擇
        self.mode = 'realtime'
        btn_layout = QHBoxLayout()
        self.btn_image = QPushButton('單張影像')
        self.btn_video = QPushButton('影片')
        self.btn_realtime = QPushButton('即時影像')
        self.btn_exit = QPushButton('離開')
        btn_layout.addWidget(self.btn_image)
        btn_layout.addWidget(self.btn_video)
        btn_layout.addWidget(self.btn_realtime)
        btn_layout.addWidget(self.btn_exit)

        # 即時影像控制 GroupBox
        realtime_group = QGroupBox('即時影像控制')
        realtime_layout = QHBoxLayout()
        self.btn_start = QPushButton('開啟鏡頭')
        self.btn_stop = QPushButton('關閉鏡頭')
        realtime_layout.addWidget(self.btn_start)
        realtime_layout.addWidget(self.btn_stop)
        realtime_group.setLayout(realtime_layout)
        realtime_group.setFixedHeight(60)  # 固定高度，可依需求調整

        layout.insertLayout(0, btn_layout)
        layout.insertWidget(1, realtime_group)

        self.btn_image.clicked.connect(self.select_image)
        self.btn_video.clicked.connect(self.select_video)
        self.btn_realtime.clicked.connect(self.set_realtime)
        self.btn_start.clicked.connect(self.start_realtime)
        self.btn_stop.clicked.connect(self.stop_realtime)
        self.btn_exit.clicked.connect(self.close)

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils
        self.current_image = None
        self.current_video = None
        self.video_cap = None

        self.set_realtime()
        self.realtime_active = False
        self.video_playing = False

    def set_realtime(self):
        self.mode = 'realtime'
        self.cap = None
        self.video_cap = None
        self.current_image = None
        self.current_image_path = None  # 清空檔名
        self.realtime_active = False
        self.image_label.clear()
        self.result_label.clear()

    def select_image(self):
        file, _ = QFileDialog.getOpenFileName(self, '選擇影像', '', 'Images (*.png *.jpg *.jpeg)')
        if file:
            self.mode = 'image'
            self.current_image = cv2.imread(file)
            self.current_image_path = file  # 記錄檔名
            self.video_cap = None
            self.cap = None

    def select_video(self):
        file, _ = QFileDialog.getOpenFileName(self, '選擇影片', '', 'Videos (*.mp4 *.avi *.mov)')
        if file:
            self.mode = 'video'
            self.current_video = file
            self.current_image_path = None  # 清空檔名
            self.video_cap = cv2.VideoCapture(file)
            self.cap = None
            self.video_playing = False

    def start_realtime(self):
        if self.mode == 'realtime' and not self.realtime_active:
            self.cap = cv2.VideoCapture(0)
            self.realtime_active = True

    def stop_realtime(self):
        if self.mode == 'realtime' and self.realtime_active:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.realtime_active = False
            self.image_label.clear()
            self.result_label.clear()

    def update_frame(self):
        frame = None
        if self.mode == 'realtime':
            self.info_label.clear()
            self.label_origin.setVisible(False)
            self.label_result.setVisible(False)
            if self.cap is not None and self.realtime_active:
                ret, frame = self.cap.read()
                if frame is not None:
                    frame = cv2.resize(frame, (self.frame_width, self.frame_height), interpolation=cv2.INTER_AREA)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result_img = frame_rgb.copy()
                    results = self.pose.process(result_img)
                    if results.pose_landmarks:
                        self.mp_drawing.draw_landmarks(
                            result_img, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                    h, w, ch = frame_rgb.shape
                    bytes_per_line = ch * w
                    qt_result = QImage(result_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                    qt_result = qt_result.scaled(self.frame_width, self.frame_height, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                    self.image_label.setPixmap(QPixmap.fromImage(qt_result))
                    self.result_label.clear()
                else:
                    self.image_label.clear()
                    self.result_label.clear()
            else:
                self.image_label.clear()
                self.result_label.clear()
            return
        elif self.mode == 'image':
            if self.current_image is not None:
                import os
                if hasattr(self, 'current_image_path') and self.current_image_path:
                    img_name = os.path.basename(self.current_image_path)
                    self.info_label.setText(f"影像檔名：{img_name}")
                else:
                    self.info_label.clear()
                frame = self.current_image.copy()
                self.label_origin.setText('原始影像')
                self.label_result.setText('預測結果')
                self.label_origin.setVisible(True)
                self.label_result.setVisible(True)
            else:
                self.info_label.clear()
                self.label_origin.setVisible(False)
                self.label_result.setVisible(False)
        elif self.mode == 'video':
            if self.video_cap is not None:
                # 取得影片屬性
                frame_count = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = self.video_cap.get(cv2.CAP_PROP_FPS)
                width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                duration = frame_count / fps if fps > 0 else 0
                import os
                video_name = os.path.basename(self.current_video) if self.current_video else ''
                info = f"影片檔名：{video_name} | 長度：{duration:.2f} 秒 | 解析度：{width}x{height} | 幀率：{fps} | 總幀數：{frame_count}"
                self.info_label.setText(info)
                # 讀取影片畫面
                ret, frame = self.video_cap.read()
                if not ret or frame is None:
                    # 播放到結尾，自動重播
                    self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.video_cap.read()
                if frame is not None:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # 預測結果
                    result_img = frame_rgb.copy()
                    results = self.pose.process(result_img)
                    if results.pose_landmarks:
                        self.mp_drawing.draw_landmarks(
                            result_img, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                    h, w, ch = frame_rgb.shape
                    # 合併原圖與預測圖
                    combined = np.zeros((h, w*2, ch), dtype=np.uint8)
                    combined[:, :w, :] = frame_rgb
                    combined[:, w:, :] = result_img
                    # 轉 QImage
                    qt_img = QImage(combined.data, w*2, h, ch*w*2, QImage.Format.Format_RGB888)
                    # 等比例縮放並置中顯示
                    scale = min(self.frame_width/(w*2), self.frame_height/h)
                    new_w = int(w*2*scale)
                    new_h = int(h*scale)
                    scaled_combined = qt_img.scaled(new_w, new_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                    final_img = QImage(self.frame_width, self.frame_height, QImage.Format.Format_RGB888)
                    final_img.fill(Qt.GlobalColor.black)
                    painter = QPainter(final_img)
                    x = (self.frame_width - new_w) // 2
                    y = (self.frame_height - new_h) // 2
                    painter.drawImage(x, y, scaled_combined)
                    painter.end()
                    self.image_label.setPixmap(QPixmap.fromImage(final_img))
                    self.result_label.clear()
                    self.label_origin.setText('原始影片')
                    self.label_result.setText('預測結果')
                    self.label_origin.setVisible(True)
                    self.label_result.setVisible(True)
                else:
                    self.image_label.clear()
                    self.result_label.clear()
                    self.label_origin.setVisible(False)
                    self.label_result.setVisible(False)
            else:
                self.image_label.clear()
                self.result_label.clear()
                self.label_origin.setVisible(False)
                self.label_result.setVisible(False)
            return
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result_img = frame_rgb.copy()
        results = self.pose.process(result_img)
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                result_img, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        # 合併原圖與預測圖
        combined_np = np.zeros((h, w*2, ch), dtype=np.uint8)
        combined_np[:, :w, :] = frame_rgb
        combined_np[:, w:, :] = result_img
        # 轉 QImage
        combined_img = QImage(combined_np.data, w*2, h, ch*w*2, QImage.Format.Format_RGB888)
        # 計算等比例縮放尺寸
        scale = min(self.frame_width/(w*2), self.frame_height/h)
        new_w = int(w*2*scale)
        new_h = int(h*scale)
        scaled_combined = combined_img.scaled(new_w, new_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        # 置中顯示
        final_img = QImage(self.frame_width, self.frame_height, QImage.Format.Format_RGB888)
        final_img.fill(Qt.GlobalColor.black)
        painter = QPainter(final_img)
        x = (self.frame_width - new_w) // 2
        y = (self.frame_height - new_h) // 2
        painter.drawImage(x, y, scaled_combined)
        painter.end()
        self.image_label.setPixmap(QPixmap.fromImage(final_img))
        self.result_label.clear()
        self.label_origin.setVisible(True)
        self.label_result.setVisible(True)

    def closeEvent(self, event):
        if self.cap is not None:
            self.cap.release()
        if self.video_cap is not None:
            self.video_cap.release()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PoseEstimationWidget()
    window.show()
    sys.exit(app.exec())
