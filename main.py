# -*- coding: utf-8 -*-
"""
这是一个整合了所有模块的眼动追踪应用的单文件版本。
主程序是基于 main2_pyqt.py 的 PyQt6 GUI。
【已整合 AI 瞳孔检测器 (detector.py)】
【已整合 九点采样校准逻辑】
【f2 优化 (步骤 1): 已将所有 I/O (数据记录) 移至工作线程】
【f3 优化 (步骤 2A): 使用多进程(multiprocessing) 并行化 AI 瞳孔检测】
【f4.0 修复: 修复了主线程与工作线程间的竞争条件 (线程安全)】
【f5.0 优化: 在 AI 工作进程中实现 ROI 追踪以大幅降低延迟】
"""

# =============================================================================
# 1. IMPORTS (FROM ALL FILES)
# =============================================================================
import sys
import cv2
import logging
import os
import time
from datetime import datetime
import csv
import threading
import numpy as np
from enum import Enum
import cv2.aruco as aruco
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import joblib

# ### AI检测器所需 ###
import onnxruntime as ort

# ### 步骤 2A: 多进程所需 ###
import multiprocessing
import queue # 用于 queue.Empty 和 queue.Full 异常

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton,
                             QLabel, QVBoxLayout, QHBoxLayout, QGridLayout,
                             QStatusBar, QMessageBox, QSizePolicy) # 导入 QSizePolicy
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject, pyqtSlot, QTimer


# =============================================================================
# 2. FROM config.py
# =============================================================================
"""
本文件存放项目的所有配置信息和全局常量。
"""

# 摄像头设备ID配置
CAMERA_CONFIG = {
    "left_eye": 0,   # 左眼摄像头ID (假设物理上的右眼)
    "right_eye": 2,  # 右眼摄像头ID (假设物理上的左眼)
    "scene": 4       # 场景摄像头ID
}

# 从字典中提取ID，方便在代码中直接使用
LEFT_EYE_CAM_ID = CAMERA_CONFIG["left_eye"]    # 代码中的 "左", 物理上的右
RIGHT_EYE_CAM_ID = CAMERA_CONFIG["right_eye"] # 代码中的 "右", 物理上的左
SCENE_CAM_ID = CAMERA_CONFIG["scene"]

# 视频帧的尺寸和帧率
FRAME_WIDTH = 640  # 场景摄像头宽度
FRAME_HEIGHT = 480 # 场景摄像头高度
FPS = 30           # 期望帧率

# --- AI 瞳孔检测器配置 ---
ONNX_MODEL_PATH = "best.int8.onnx"    # 你的量化 (INT8) ONNX 模型路径
AI_INPUT_SIZE = 192                   # 必须和你训练时的 imgsz 一致
AI_CONF_THRESHOLD = 0.35              # AI 框的置信度阈值
AI_NMS_THRESHOLD = 0.4                # NMS 阈值
# 瞳孔精细拟合的二值化类型 (用于 AI 框内的经典算法)
# 选项1: "暗瞳" (Dark Pupil) -> cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
# 选项2: "亮瞳" (Bright Pupil) -> cv2.THRESH_BINARY + cv2.THRESH_OTSU
PUPIL_FINE_TUNE_THRESHOLD_TYPE = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU

# --- 九点校准配置 ---
CALIBRATION_TARGET_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8] # 你的9个ArUco标记ID
CALIBRATION_SAMPLES_TO_COLLECT = 10 # 每个点采集10个有效样本
CALIBRATION_MIN_SAMPLES_REQUIRED = 5 # 每个点至少需要的样本数

# =============================================================================
# 3. FROM camera.py
# =============================================================================
class CameraStream(threading.Thread):
    """
    一个独立的线程，用于连续从指定的摄像头设备捕获视频帧。
    (移植自 complex_app.py 的更健壮的版本)
    """
    def __init__(self, camera_id, width, height, fps):
        """
        初始化摄像头视频流。
        """
        super().__init__()
        self.camera_id = camera_id
        self.requested_width = width
        self.requested_height = height
        self.requested_fps = fps

        self.cap = cv2.VideoCapture(camera_id + cv2.CAP_V4L2)

        if not self.cap.isOpened():
            raise IOError(f"无法打开摄像头 {camera_id}")

        # 尝试设置分辨率和帧率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.requested_fps)

        # 再次读取确认实际设置
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = actual_fps if actual_fps > 0 else self.requested_fps # 使用实际帧率，如果读取失败则使用请求值
        if self.width != self.requested_width or self.height != self.requested_height:
             logging.warning(f"摄像头 {camera_id}: 请求分辨率 {self.requested_width}x{self.requested_height}，实际为 {self.width}x{self.height}。")

        logging.info(f"摄像头 {camera_id} 打开成功: {self.width}x{self.height} @ {self.fps:.2f} FPS (请求 {self.requested_fps} FPS)")


        self.latest_frame = None
        self.read_lock = threading.Lock()
        self.stop_threads = False
        self.daemon = True # 设置为守护线程，主程序退出时自动结束

    def run(self):
        """
        线程的主体函数，循环读取摄像头的视频帧。
        """
        while not self.stop_threads:
            ret, frame = self.cap.read()
            if ret:
                with self.read_lock:
                    self.latest_frame = frame
            else:
                logging.warning(f"无法从摄像头 {self.camera_id} 读取帧。")
                time.sleep(0.1) # 如果读取失败，等待更长时间
        
        self.cap.release()
        logging.info(f"摄像头 {self.camera_id} 已释放。")


    def get_frame(self):
        """
        获取当前最新的一帧。
        """
        with self.read_lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None

    def stop(self):
        """
        设置停止标志，以终止线程的执行。
        """
        logging.info(f"请求停止摄像头 {self.camera_id} 线程...")
        self.stop_threads = True

    def release(self):
        """Alias for stop() for compatibility."""
        self.stop()

# =============================================================================
# 4. FROM vision.py
# =============================================================================
"""
本文件包含项目中所有的计算机视觉处理函数。
"""
# --- ArUco 标记检测初始化 ---
try:
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    logging.info("使用新版 OpenCV ArUco API。")
except AttributeError:
    try:
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters_create()
        detector = None
        logging.warning("警告: 检测到旧版 OpenCV, 使用兼容的 ArUco API。")
    except AttributeError:
        logging.error("无法初始化 ArUco 检测器。请确保已安装 opencv-contrib-python。")
        aruco_dict = None
        parameters = None
        detector = None


def find_all_aruco_markers(image: np.ndarray):
    """
    在图像中寻找所有ArUco标记并返回一个包含其ID和中心点坐标的字典。
    (移植自 complex_app.py)
    """
    if image is None or image.size == 0 or aruco_dict is None:
        return {}
    try:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if detector: # 新版 OpenCV
            corners, ids, rejectedImgPoints = detector.detectMarkers(gray_image)
        elif parameters is not None: # 旧版 OpenCV
            corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray_image, aruco_dict, parameters=parameters)
        else: # 如果初始化失败
            return {}

        markers_dict = {}
        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                marker_corners = corners[i].reshape((4, 2))
                center_x = int(np.mean(marker_corners[:, 0]))
                center_y = int(np.mean(marker_corners[:, 1]))
                markers_dict[marker_id] = (center_x, center_y)
        return markers_dict
    except cv2.error as e:
        logging.error(f"ArUco 检测失败: {e}")
        return {}
    except Exception as e:
        logging.exception("在 find_all_aruco_markers 中发生意外错误")
        return {}


# ---------------------------------------------------------------------------
# ### AI 瞳孔检测器类 (我们的 "AI 心脏") ###
# ---------------------------------------------------------------------------
class AIPupilDetector:
    """
    封装了YOLOv8-Nano (ONNX) AI模型 和 经典CV精细拟合的混合瞳孔检测器。
    (来自 AIEyeTracker.py)
    """
    def __init__(self, model_path=ONNX_MODEL_PATH):
        self.input_size = AI_INPUT_SIZE
        self.conf_threshold = AI_CONF_THRESHOLD
        self.nms_threshold = AI_NMS_THRESHOLD
        
        try:
            # 设置 ONNX Runtime 的 Session 选项
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = cv2.getNumberOfCPUs()
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            self.session = ort.InferenceSession(model_path, sess_options=sess_options, providers=['CPUExecutionProvider'])
            
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            # ### 步骤 2A: 区分日志，显示进程ID
            logging.info(f"[{os.getpid()}] AI瞳孔检测器加载成功: {model_path}")
        except Exception as e:
            logging.error(f"[{os.getpid()}] !!! 致命错误：无法加载 ONNX 模型: {e}")
            logging.error(f"[{os.getpid()}] 请确保 '{ONNX_MODEL_PATH}' 文件在同一目录下。")
            raise e

    def _preprocess(self, frame):
        """
        将 OpenCV 帧 (H, W, C) 转换为模型所需的 (N, C, H, W) 张量。
        """
        img_resized = cv2.resize(frame, (self.input_size, self.input_size))
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        img_batch = np.expand_dims(img_transposed, axis=0)
        return img_batch

    def detect(self, frame):
        """
        在单帧上运行完整的 AI + CV 混合检测流程。
        返回: (pupil_center, pupil_ellipse, ai_bbox) 或 (None, None, None)
        """
        if frame is None or frame.size == 0:
            return None, None, None
            
        frame_height, frame_width = frame.shape[:2]

        # 1. AI 粗定位
        try:
            input_tensor = self._preprocess(frame)
            outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        except Exception as e:
            logging.error(f"[{os.getpid()}] AI模型推理失败: {e}")
            return None, None, None
        
        # 2. AI 结果后处理
        detections = outputs[0][0].T
        boxes = []
        confidences = []

        for det in detections:
            confidence = det[4]
            if confidence > self.conf_threshold:
                cx, cy, w, h = det[:4]
                x1 = int((cx - w / 2) * frame_width / self.input_size)
                y1 = int((cy - h / 2) * frame_height / self.input_size)
                x2 = int((cx + w / 2) * frame_width / self.input_size)
                y2 = int((cy + h / 2) * frame_height / self.input_size)
                boxes.append([x1, y1, x2, y2])
                confidences.append(float(confidence))

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        
        if len(indices) == 0:
            return None, None, None # AI 未找到瞳孔
        # 3. 经典算法精细化
        best_index = indices[0]
        x1, y1, x2, y2 = boxes[best_index]
        ai_bbox = (x1, y1, x2, y2) # <-- v5.0: 保存 BBox

        pad = 5
        roi_x1 = max(0, x1 - pad)
        roi_y1 = max(0, y1 - pad)
        roi_x2 = min(frame_width, x2 + pad)
        roi_y2 = min(frame_height, y2 + pad)
        
        pupil_roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

        if pupil_roi.size == 0:
            return None, None, None # ROI 无效

        try:
            gray_roi = cv2.cvtColor(pupil_roi, cv2.COLOR_BGR2GRAY)
            gray_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)
            
            _, thresholded_roi = cv2.threshold(gray_roi, 0, 255, PUPIL_FINE_TUNE_THRESHOLD_TYPE)

            contours, _ = cv2.findContours(thresholded_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return None, None, None 

            largest_contour = max(contours, key=cv2.contourArea)
            
            if len(largest_contour) < 5:
                return None, None, None 

            ellipse_roi = cv2.fitEllipse(largest_contour)
            
            (cx_roi, cy_roi), (MA, ma), angle = ellipse_roi
            
            ellipse_center_frame = (int(cx_roi + roi_x1), int(cy_roi + roi_y1))
            ellipse_frame = ((cx_roi + roi_x1, cy_roi + roi_y1), (MA, ma), angle)
            
            return ellipse_center_frame, ellipse_frame, ai_bbox # <-- v5.0: 返回 BBox
            
        except cv2.error as fit_e:
            # logging.debug(f"精细拟合失败: {fit_e}")
            return None, None, None
        except Exception as e:
            logging.exception(f"[{os.getpid()}] 精细拟合时发生意外错误")
            return None, None, None

def get_eye_features(pupil_center, pupil_ellipse):
    """
    根据瞳孔检测结果提取特征向量。
    (移植自 complex_app.py，带np.isfinite检查)
    """
    if pupil_center is not None and pupil_ellipse is not None:
        (x, y), (major, minor), angle = pupil_ellipse
        if not np.isfinite(major) or not np.isfinite(minor) or not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(angle):
            logging.warning(f"检测到无效的椭圆参数: center=({x},{y}), axes=({major},{minor}), angle={angle}")
            return np.zeros(5, dtype=np.float32)
        return np.array([x, y, major, minor, angle], dtype=np.float32)
    else:
        return np.zeros(5, dtype=np.float32)


### ======================================================================== ###
### 步骤 2A: 新增 - 瞳孔检测的独立工作进程
### ======================================================================== ###
class PupilDetectionWorker(multiprocessing.Process):
    """
    在一个独立的进程中运行 AI 瞳孔检测器，
    通过队列 (Queue) 接收帧并发送结果。
    v5.0: 增加了 ROI 追踪逻辑
    """
    def __init__(self, input_q, output_q, model_path):
        super().__init__()
        self.input_q = input_q
        self.output_q = output_q
        self.model_path = model_path
        self.detector = None
        self.daemon = True # 确保随主进程退出
    def run(self):
        # 1. 初始化检测器(必须在子进程中)
        try:
            # 子进程会继承 logging 配置
            logging.info(f"[{os.getpid()}] 工作进程启动，正在加载模型...")
            self.detector = AIPupilDetector(self.model_path)
            logging.info(f"[{os.getpid()}] 工作进程模型加载完毕。")
        except Exception as e:
            logging.error(f"[{os.getpid()}] 工作进程初始化失败: {e}")
            return

        # 2. 循环处理
        last_known_bbox = None # <-- v5.0: 记住上一帧的BBox
        roi_padding = 35       # <-- v5.0: ROI向外扩展的像素
        while True:
            try:
                # 阻塞等待新帧
                frame = self.input_q.get()

                # 3. 检查 "毒丸" (None) 信号以退出
                if frame is None:
                    logging.info(f"[{os.getpid()}] 收到退出信号。")
                    break

                # 4. 执行核心工作 (AI 推理) - v5.0 带 ROI 追踪
                frame_height, frame_width = frame.shape[:2]
                
                if last_known_bbox is None:
                    # 状态 1: 全图搜索 (没有记忆)
                    roi_frame = frame
                    offset = (0, 0)
                else:
                    # 状态 2: ROI 搜索 (使用记忆)
                    x1, y1, x2, y2 = last_known_bbox
                    roi_x1 = max(0, x1 - roi_padding)
                    roi_y1 = max(0, y1 - roi_padding)
                    roi_x2 = min(frame_width, x2 + roi_padding)
                    roi_y2 = min(frame_height, y2 + roi_padding)
                    
                    roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]
                    offset = (roi_x1, roi_y1)

                    # 如果裁剪后的 ROI 无效 (比如BBox在边缘)，则退回全图搜索
                    if roi_frame.size == 0:
                        roi_frame = frame
                        offset = (0, 0)
                        last_known_bbox = None # 丢失追踪，强制重置
                # 在 roi_frame (可能是小图) 上运行检测
                pupil_center_roi, pupil_ellipse_roi, bbox_roi = self.detector.detect(roi_frame)
                
                ox, oy = offset # (offset_x, offset_y)
                
                if pupil_center_roi is not None:
                    # --- 成功: 将 ROI 坐标映射回全帧坐标 ---
                    pupil_center = (pupil_center_roi[0] + ox, pupil_center_roi[1] + oy)
                    
                    ( (cx_roi, cy_roi), (MA, ma), angle ) = pupil_ellipse_roi
                    pupil_ellipse = ( (cx_roi + ox, cy_roi + oy), (MA, ma), angle )
                    
                    # --- 记住这一帧的 BBox (全帧坐标) ---
                    bx1, by1, bx2, by2 = bbox_roi
                    last_known_bbox = (bx1 + ox, by1 + oy, bx2 + ox, by2 + oy)
                else:
                    # --- 失败: 清除记忆，触发下次全图搜索 ---
                    pupil_center = None
                    pupil_ellipse = None
                    last_known_bbox = None

                # 5. 清理输出队列 (只保留最新结果)
                try:
                    self.output_q.get_nowait()
                except queue.Empty:
                    pass # 很好，队列是空的

                # 6. 发送新结果 (非阻塞)
                try:
                    self.output_q.put_nowait((pupil_center, pupil_ellipse))
                except queue.Full:
                    pass # 主线程没来得及取，丢弃这个结果
            except queue.Empty:
                pass # 不应发生，因为 input_q.get() 是阻塞的
            except Exception as e:
                logging.exception(f"[{os.getpid()}] 工作进程循环出错: {e}")
        
        logging.info(f"[{os.getpid()}] 工作进程关闭。")

### ======================================================================== ###
### 步骤 2A 结束
### ======================================================================== ###


# =============================================================================
# 5. FROM gaze_estimator.py
# =============================================================================
class GazeEstimator:
    """
    使用多项式回归模型来估计用户的注视点。
    (移植自 complex_app.py，带更健壮的检查)
    """
    def __init__(self, degree=2, model_path='calibration_model.pkl'):
        self.model_path = model_path
        self.model = None
        self.is_trained = False
        self.last_gaze_point = None

        if os.path.exists(self.model_path):
            self.load_model()

        if self.model is None:
            self.model = make_pipeline(PolynomialFeatures(degree, include_bias=False), LinearRegression())

    def train(self, calibration_data):
        if not calibration_data:
            logging.error("训练失败：校准数据为空。")
            return False

        try:
            X_train = np.array([item[0] for item in calibration_data])
            y_train = np.array([item[1] for item in calibration_data])

            if X_train.size == 0 or y_train.size == 0 or \
               not np.all(np.isfinite(X_train)) or not np.all(np.isfinite(y_train)):
                logging.error(f"训练失败：校准数据包含无效值。X shape: {X_train.shape}, Y shape: {y_train.shape}")
                return False

            self.model.fit(X_train, y_train)
            self.is_trained = True
            logging.info("注视点估计模型训练完成。")
            self.save_model()
            return True
        except Exception as e:
            logging.exception("训练注视点模型时出错:")
            self.is_trained = False
            return False


    def predict(self, eye_features):
        if not self.is_trained or eye_features is None:
            return None
        try:
            if not isinstance(eye_features, np.ndarray) or eye_features.shape != (10,):
                logging.warning(f"预测时的特征形状不正确: {eye_features.shape if isinstance(eye_features, np.ndarray) else type(eye_features)}")
                return None

            if not np.all(np.isfinite(eye_features)):
                logging.warning(f"预测时的特征包含无效值: {eye_features}")
                return None

            gaze = self.model.predict(eye_features.reshape(1, -1))[0]

            if not np.all(np.isfinite(gaze)):
                logging.warning(f"模型预测结果无效: {gaze}")
                return self.last_gaze_point

            predicted_point = (int(gaze[0]), int(gaze[1]))
            self.last_gaze_point = predicted_point
            return predicted_point
        except Exception as e:
            logging.error(f"注视点预测失败: {e}")
            return None


    def save_model(self):
        try:
            joblib.dump(self.model, self.model_path)
            logging.info(f"模型已保存到 {self.model_path}")
        except Exception as e:
            logging.error(f"保存模型失败: {e}")

    def load_model(self):
        if not os.path.exists(self.model_path):
            logging.warning(f"模型文件 {self.model_path} 不存在。")
            self.model = None
            self.is_trained = False
            return

        try:
            self.model = joblib.load(self.model_path)
            if hasattr(self.model, 'predict'):
                self.is_trained = True
                logging.info(f"已从 {self.model_path} 加载模型。")
            else:
                logging.error(f"从 {self.model_path} 加载的文件无效。")
                self.model = None
                self.is_trained = False
        except Exception as e:
            logging.error(f"加载模型失败: {e}")
            self.model = None
            self.is_trained = False

# =============================================================================
# 6. FROM tracker.py
# =============================================================================
class TrackerState(Enum):
    IDLE = 0
    CALIBRATING = 1
    TRACKING = 2
    VALIDATING = 3 # ### 新增：验证状态 ###

class EyeTracker:
    """
    整合了 AI 瞳孔检测器 和 九点采样校准逻辑 的主追踪器。
    """
    def __init__(self):
        # 眼睛摄像头使用较低分辨率以提高帧率
        self.eye_cam_width = 320
        self.eye_cam_height = 240
        # 缩放因子，用于将低分辨率眼部特征映射回原始分辨率 (如果需要的话)
        # 注意：我们的 AI 模型在 detect() 内部处理缩放，所以这个因子目前仅用于ROI
        self.processing_scale_factor = FRAME_WIDTH / self.eye_cam_width if self.eye_cam_width > 0 else 1.0

        try:
            # 注意 CAM ID 的交换，以匹配 physical right -> left_cam
            self.left_cam = CameraStream(RIGHT_EYE_CAM_ID, self.eye_cam_width, self.eye_cam_height, FPS)
            self.right_cam = CameraStream(LEFT_EYE_CAM_ID, self.eye_cam_width, self.eye_cam_height, FPS)
            self.scene_cam = CameraStream(SCENE_CAM_ID, FRAME_WIDTH, FRAME_HEIGHT, FPS)

            self.left_cam.start()
            self.right_cam.start()
            self.scene_cam.start()
        except IOError as e:
            logging.error(f"初始化摄像头失败: {e}")
            raise e
        except Exception as e:
            logging.exception("初始化 CameraStream 时未知错误")
            raise e

        # ### 步骤 2A: 移除此处的 AI 检测器初始化
        # try:
        #     self.pupil_detector = AIPupilDetector(ONNX_MODEL_PATH)
        # except Exception as e:
        #     raise IOError(f"AI瞳孔检测器初始化失败: {e}")

        ### ================================================================== ###
        ### 步骤 2A: 初始化多进程通信
        ### ================================================================== ###
        # maxsize=1 确保我们只处理最新的帧，防止延迟
        logging.info("正在创建多进程瞳孔检测器...")
        # 'spawn' 启动方式在 Windows/macOS/Linux 上更安全、一致
        try:
            ctx = multiprocessing.get_context('spawn')
        except Exception:
            logging.warning("get_context('spawn') 失败, 回退到默认方式。")
            ctx = multiprocessing
            
        self.left_input_q = ctx.Queue(maxsize=1)
        self.left_output_q = ctx.Queue(maxsize=1)
        self.right_input_q = ctx.Queue(maxsize=1)
        self.right_output_q = ctx.Queue(maxsize=1)

        self.left_detector_proc = PupilDetectionWorker(
            self.left_input_q, self.left_output_q, ONNX_MODEL_PATH)
        self.right_detector_proc = PupilDetectionWorker(
            self.right_input_q, self.right_output_q, ONNX_MODEL_PATH)

        self.left_detector_proc.start()
        self.right_detector_proc.start()
        logging.info("多进程检测器已启动。")
        ### ================================================================== ###

        self.gaze_estimator = GazeEstimator()
        self.state = TrackerState.IDLE
        
        # ### 新增：移植自 complex_app.py 的校准变量 ###
        self.calibration_data = []
        self.calibration_point_count = 0
        self.all_detected_markers = {}
        self.CALIBRATION_TARGET_IDS = CALIBRATION_TARGET_IDS
        self.current_target_index = 0
        self.SAMPLES_TO_COLLECT = CALIBRATION_SAMPLES_TO_COLLECT
        self.MIN_SAMPLES_REQUIRED = CALIBRATION_MIN_SAMPLES_REQUIRED
        self.feature_buffer = []
        self.is_collecting_samples = False # <- 拼写错误的修复
        self.target_coord_for_collection = None
        
        ### 步骤 2A: 为特征和结果添加缓存
        self.last_left_features = np.zeros(5, dtype=np.float32)
        self.last_right_features = np.zeros(5, dtype=np.float32)
        self.last_left_center = None
        self.last_left_ellipse = None
        self.last_right_center = None
        self.last_right_ellipse = None
        
        logging.info("眼动追踪器初始化完成。")
        if self.gaze_estimator.is_trained:
            logging.info("已加载预训练的校准模型。")

    def start_calibration(self):
        """ (移植自 complex_app.py) """
        self.state = TrackerState.CALIBRATING
        self.calibration_data = []
        self.calibration_point_count = 0
        self.current_target_index = 0
        self.all_detected_markers = {}
        self.is_collecting_samples = False
        self.feature_buffer = []
        if len(self.CALIBRATION_TARGET_IDS) > 0:
            logging.info(f"校准开始。请注视 {self.CALIBRATION_TARGET_IDS[self.current_target_index]} 号标记...")
        else:
            logging.error("校准失败：未定义目标。")

    def add_calibration_point(self):
        """ (移植自 complex_app.py) """
        if self.state != TrackerState.CALIBRATING: return False
        if self.is_collecting_samples: return False
        if self.current_target_index >= len(self.CALIBRATION_TARGET_IDS): return False
        
        target_id = self.CALIBRATION_TARGET_IDS[self.current_target_index]
        
        if target_id not in self.all_detected_markers:
            logging.warning(f"采集失败：{target_id} 号标记未检测到。")
            return False
            
        target_coord = self.all_detected_markers[target_id]
        logging.info(f"开始采集点 {self.calibration_point_count + 1} ({target_id}号)...")
        self.target_coord_for_collection = target_coord
        self.feature_buffer = []
        self.is_collecting_samples = True
        return True

    def finish_sample_collection(self):
        """ (移植自 complex_app.py) """
        self.is_collecting_samples = False
        if len(self.feature_buffer) < self.MIN_SAMPLES_REQUIRED:
            logging.warning(f"采集失败：有效样本不足 ({len(self.feature_buffer)}/{self.MIN_SAMPLES_REQUIRED})。")
            self.feature_buffer = []
            return
            
        averaged_features = np.mean(self.feature_buffer, axis=0)
        self.calibration_data.append((averaged_features, self.target_coord_for_collection))
        self.calibration_point_count += 1
        self.current_target_index += 1
        logging.info(f"采集点 {self.calibration_point_count} 成功 (平均 {len(self.feature_buffer)} 帧)")
        
        if self.current_target_index < len(self.CALIBRATION_TARGET_IDS):
            next_target_id = self.CALIBRATION_TARGET_IDS[self.current_target_index]
            logging.info(f"请注视 {next_target_id} 号标记。")
        else:
            logging.info("所有点采集完毕！请按'完成'。")
            
        self.feature_buffer = []
        self.target_coord_for_collection = None

    def finish_calibration(self):
        """ (移植自 complex_app.py) """
        min_points = len(self.CALIBRATION_TARGET_IDS)
        if self.calibration_point_count < min_points:
            logging.warning(f"数据不足 ({self.calibration_point_count}/{min_points})，无法完成校准。")
            return False
            
        logging.info("正在训练模型...")
        if self.gaze_estimator.train(self.calibration_data):
            ### ================================================== ###
            ### v4.0 逻辑: 恢复 VALIDATING 状态
            ### ================================================== ###
            logging.info("模型训练成功！进入验证模式。")
            self.state = TrackerState.VALIDATING
            return True
        else:
            logging.error("模型训练失败。")
            self.state = TrackerState.IDLE
            return False

    def is_calibration_finished(self):
        return self.gaze_estimator.is_trained

    def start_tracking(self):
        """ (移植自 complex_app.py) """
        if not self.gaze_estimator.is_trained: return
        self.state = TrackerState.TRACKING
        logging.info("追踪开始。")

    def stop_tracking(self):
        self.state = TrackerState.IDLE
        logging.info("追踪停止。")

    def get_all_frames(self):
        """
        *** 核心整合函数 ***
        使用 AI 检测器，并由九点校准的状态机管理。
        """
        left_frame_low_res = self.left_cam.get_frame()
        right_frame_low_res = self.right_cam.get_frame()
        scene_frame_raw = self.scene_cam.get_frame()

        ### ================================================================== ###
        ### 步骤 2A: 提交帧并获取 AI 检测器特征 (并行)
        ### ================================================================== ###
        
        # 1a. 提交新帧进行处理 (非阻塞)
        for q, frame in [(self.left_input_q, left_frame_low_res), (self.right_input_q, right_frame_low_res)]:
            if frame is not None:
                try:
                    q.get_nowait() # 清空上一帧 (如果工作进程慢了)
                except queue.Empty:
                    pass
                try:
                    q.put_nowait(frame) # 提交当前帧
                except queue.Full:
                    pass # 工作进程正忙，跳过此帧
        # 1b. 获取最新可用的处理结果 (非阻塞)
        try:
            # 检查左眼是否有新结果
            l_center, l_ellipse = self.left_output_q.get_nowait()
            # v5.1 修复: 仅当新结果有效时才更新缓存，防止闪烁
            if l_ellipse is not None:
                self.last_left_center, self.last_left_ellipse = l_center, l_ellipse
                l_center, l_ellipse = l_center, l_ellipse
            else:
                # 新结果是 (None, None)，丢弃它，继续使用旧缓存
                l_center, l_ellipse = self.last_left_center, self.last_left_ellipse
        except queue.Empty:
            # 没有新结果，使用缓存
            l_center, l_ellipse = self.last_left_center, self.last_left_ellipse

        try:
            # 检查右眼是否有新结果
            r_center, r_ellipse = self.right_output_q.get_nowait()
            if r_ellipse is not None:
                self.last_right_center, self.last_right_ellipse = r_center, r_ellipse
                r_center, r_ellipse = r_center, r_ellipse
            else:
                # 新结果是 (None, None)，丢弃它，继续使用旧缓存
                r_center, r_ellipse = self.last_right_center, self.last_right_ellipse
        except queue.Empty:
            # 没有新结果，使用缓存
            r_center, r_ellipse = self.last_right_center, self.last_right_ellipse

        # 1c. 将结果赋给局部变量
        left_pupil_center, left_pupil_ellipse = l_center, l_ellipse
        right_pupil_center, right_pupil_ellipse = r_center, r_ellipse
        
        current_left_features = get_eye_features(left_pupil_center, left_pupil_ellipse)
        current_right_features = get_eye_features(right_pupil_center, right_pupil_ellipse)
        ### ================================================================== ###
        
        # 保持上一帧的有效特征
        if not np.all(current_left_features == 0):
            self.last_left_features = current_left_features
        if not np.all(current_right_features == 0):
            self.last_right_features = current_right_features
            
        left_features = self.last_left_features
        right_features = self.last_right_features

        # --- 2. 处理校准样本采集 ---
        if self.is_collecting_samples:
            # 只有当两只眼睛都有效时才采集
            if not np.all(current_left_features == 0) and not np.all(current_right_features == 0):
                combined_features = np.concatenate([current_left_features, current_right_features])
                if np.all(np.isfinite(combined_features)):
                    self.feature_buffer.append(combined_features)
                    if len(self.feature_buffer) >= self.SAMPLES_TO_COLLECT:
                        self.finish_sample_collection()

        # --- 3. 创建显示帧 ---
        left_display = left_frame_low_res if left_frame_low_res is not None else np.zeros((self.eye_cam_height, self.eye_cam_width, 3), dtype=np.uint8)
        right_display = right_frame_low_res if right_frame_low_res is not None else np.zeros((self.eye_cam_height, self.eye_cam_width, 3), dtype=np.uint8)
        scene_display = scene_frame_raw.copy() if scene_frame_raw is not None else np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

        # 绘制瞳孔 (使用 AI 检测器返回的原始低分辨率椭圆)
        if left_pupil_ellipse is not None:
            try: cv2.ellipse(left_display, left_pupil_ellipse, (0, 0, 255), 3) # 红色 AI 椭圆
            except cv2.error as draw_e: logging.warning(f"绘制左眼椭圆失败: {draw_e}")
        if right_pupil_ellipse is not None:
            try: cv2.ellipse(right_display, right_pupil_ellipse, (0, 0, 255), 3) # 红色 AI 椭圆
            except cv2.error as draw_e: logging.warning(f"绘制右眼椭圆失败: {draw_e}")

        main_display = scene_display

        # --- 4. 检测 ArUco 标记 (仅在需要时) ---
        if self.state == TrackerState.CALIBRATING or self.state == TrackerState.VALIDATING:
            if main_display is not None and main_display.size > 0:
                self.all_detected_markers = find_all_aruco_markers(main_display)
            else:
                self.all_detected_markers = {}

        # --- 5. 根据状态绘制 ---
        if self.state == TrackerState.CALIBRATING:
            if self.current_target_index < len(self.CALIBRATION_TARGET_IDS):
                target_id = self.CALIBRATION_TARGET_IDS[self.current_target_index]
                for mid, pos in self.all_detected_markers.items():
                    color = (0, 255, 0) if mid == target_id else (150, 150, 150)
                    thickness = 2 if mid == target_id else 1
                    try: cv2.circle(main_display, pos, 15, color, thickness)
                    except cv2.error as draw_e: logging.warning(f"绘制校准标记 {mid} 失败: {draw_e}")

        elif self.state == TrackerState.VALIDATING or self.state == TrackerState.TRACKING:
            if not np.all(left_features == 0) and not np.all(right_features == 0):
                combined = np.concatenate([left_features, right_features])
                if np.all(np.isfinite(combined)):
                    gaze = self.gaze_estimator.predict(combined)
                    if gaze and main_display is not None and main_display.size > 0:
                        try:
                            h, w = main_display.shape[:2]
                            gaze_x, gaze_y = gaze
                            # 简单的边界检查
                            gaze_x = max(0, min(w - 1, gaze_x))
                            gaze_y = max(0, min(h - 1, gaze_y))
                            gaze = (gaze_x, gaze_y)

                            cv2.circle(main_display, gaze, 20, (255, 0, 0), 2)
                            cv2.line(main_display, (gaze[0]-15, gaze[1]), (gaze[0]+15, gaze[1]), (255, 0, 0), 2)
                            cv2.line(main_display, (gaze[0], gaze[1]-15), (gaze[0], gaze[1]+15), (255, 0, 0), 2)
                        except (cv2.error, OverflowError, TypeError, SystemError) as draw_e:
                            logging.warning(f"绘制注视点失败: {type(draw_e).__name__}: {draw_e}, coord={gaze}")

            if self.state == TrackerState.VALIDATING and main_display is not None and main_display.size > 0:
                for mid, pos in self.all_detected_markers.items():
                    try: cv2.circle(main_display, pos, 15, (0, 255, 0), 2) # 验证时全显示绿色
                    except cv2.error as draw_e: logging.warning(f"绘制验证标记 {mid} 失败: {draw_e}")

        return {
            "left": left_display, "right": right_display, "main": main_display,
            "raw_left": left_frame_low_res, "raw_right": right_frame_low_res, "raw_scene": scene_frame_raw
        }

    def stop(self):
        """ (移植自 complex_app.py) """
        
        ### ================================================================== ###
        ### 步骤 2A: 首先停止多进程工作器
        ### ================================================================== ###
        logging.info("正在停止瞳孔检测工作进程...")
        try:
            if hasattr(self, 'left_input_q'):
                self.left_input_q.put(None) # 发送 "毒丸"
            if hasattr(self, 'right_input_q'):
                self.right_input_q.put(None) # 发送 "毒丸"
            
            if hasattr(self, 'left_detector_proc') and self.left_detector_proc.is_alive():
                self.left_detector_proc.join(timeout=1.0)
                if self.left_detector_proc.is_alive():
                    logging.warning("左眼检测进程超时，将强制终止。")
                    self.left_detector_proc.terminate()
            if hasattr(self, 'right_detector_proc') and self.right_detector_proc.is_alive():
                self.right_detector_proc.join(timeout=1.0)
                if self.right_detector_proc.is_alive():
                    logging.warning("右眼检测进程超时，将强制终止。")
                    self.right_detector_proc.terminate()
            
            # 关闭队列
            for q_name in ['left_input_q', 'left_output_q', 'right_input_q', 'right_output_q']:
                if hasattr(self, q_name):
                    q = getattr(self, q_name)
                    q.close()
                    q.join_thread()
            logging.info("检测工作进程已停止。")
        except Exception as e:
            logging.exception(f"停止检测工作进程时出错: {e}")
        ### ================================================================== ###

        self.state = TrackerState.IDLE
        if hasattr(self, 'left_cam') and self.left_cam.is_alive(): self.left_cam.stop()
        if hasattr(self, 'right_cam') and self.right_cam.is_alive(): self.right_cam.stop()
        if hasattr(self, 'scene_cam') and self.scene_cam.is_alive(): self.scene_cam.stop()
        
        threads_to_join = []
        if hasattr(self, 'left_cam'): threads_to_join.append(self.left_cam)
        if hasattr(self, 'right_cam'): threads_to_join.append(self.right_cam)
        if hasattr(self, 'scene_cam'): threads_to_join.append(self.scene_cam)
        
        for t in threads_to_join:
            try:
                t.join(timeout=1.0)
                if t.is_alive():
                    logging.warning(f"摄像头 {t.camera_id} 线程在超时后仍未结束。")
            except Exception as e:
                logging.error(f"等待摄像头 {t.camera_id} 线程结束时出错: {e}")

        logging.info("眼动追踪器已停止。")


# =============================================================================
# 7. FROM main2_pyqt.py (MAIN APPLICATION)
# =============================================================================
class QtLogHandler(logging.Handler, QObject):
    """ (移植自 complex_app.py) """
    new_log_record = pyqtSignal(str)
    def __init__(self):
        super().__init__()
        QObject.__init__(self)
        self.setFormatter(logging.Formatter('%(message)s'))
    def emit(self, record):
        try:
            msg = self.format(record)
            self.new_log_record.emit(msg)
        except Exception:
            import traceback
            traceback.print_exc(file=sys.stderr)

class TrackerWorker(QObject):
    """ (移植自 complex_app.py，带精确sleep) """
    new_frames = pyqtSignal(dict)
    tracker_initialized = pyqtSignal(bool, str)
    
    ### ================================================================ ###
    ### v4.0 修复: 为 GUI 反馈添加新信号
    ### ================================================================ ###
    tracker_error = pyqtSignal(str, str) # title, message
    tracker_info = pyqtSignal(str, str)  # title, message
    ### ================================================================ ###

    def __init__(self):
        super().__init__()
        self.tracker = None
        self.is_running = False
        
        ### ================================================================ ###
        ### 步骤1 优化：从 GUI 移植过来的变量 ###
        ### ================================================================ ###
        self.is_experiment_running = False
        self.experiment_data = []
        self.video_writers = {}
        self.experiment_path = ""
        ### ================================================================ ###

    ### ==================================================================== ###
    ### 步骤1 优化：从 GUI 移植过来的函数，并设为槽函数 ###
    ### ==================================================================== ###
    @pyqtSlot()
    def start_experiment(self):
        # ### 注意：这里 self.tracker_worker.tracker 已改为 self.tracker
        if not (self.tracker and self.tracker.state == TrackerState.TRACKING and self.tracker.is_calibration_finished()):
            # ### 不再弹出QMessageBox，改为日志记录
            logging.warning("无法开始实验：请确保已完成校准并处于追踪模式。")
            return
            
        self.is_experiment_running = True
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.experiment_path = os.path.join(os.getcwd(), timestamp)
        os.makedirs(self.experiment_path, exist_ok=True)
        logging.info(f"实验数据将保存到: {self.experiment_path}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writers = {}
        
        if not (self.tracker): # ### 已修改
            logging.error("无法开始实验：追踪器未初始化。")
            self.is_experiment_running = False
            return
            
        cam_sources_info = {
            "left_eye": self.tracker.left_cam,  # ### 已修改
            "right_eye": self.tracker.right_cam, # ### 已修改
            "scene": self.tracker.scene_cam     # ### 已修改
        }
        for name, cam in cam_sources_info.items():
            if cam:
                fw, fh, fps_val = cam.width, cam.height, cam.fps
                video_path = os.path.join(self.experiment_path, f"{name}.mp4")
                try:
                    if fw > 0 and fh > 0:
                        eff_fps = fps_val if fps_val > 0 else 10
                        writer = cv2.VideoWriter(video_path, fourcc, eff_fps, (fw, fh))
                        if writer.isOpened():
                            self.video_writers[name] = writer
                            logging.info(f"{name} 写入器创建: {video_path} ({fw}x{fh} @ {eff_fps} FPS)")
                        else: logging.error(f"无法打开 {name} 写入器: {video_path}")
                    else: logging.error(f"无法为 {name} 创建写入器，无效尺寸: {fw}x{fh}")
                except Exception as e: logging.exception(f"创建 {name} 写入器出错")
            else: logging.warning(f"无法为 {name} 创建写入器，摄像头对象不存在。")
            
        self.experiment_data = []
        # ### update_button_states() 会在 GUI 线程中被 update_gui 自动调用
        logging.info("实验进行中...")

    @pyqtSlot()
    def end_experiment(self):
        if not self.is_experiment_running: return
        self.is_experiment_running = False
        logging.info("正在释放视频写入器...")
        active_writers = list(self.video_writers.items())
        for name, writer in active_writers:
            try:
                if writer and writer.isOpened():
                    writer.release()
                    logging.info(f"已释放 {name} 写入器。")
                if name in self.video_writers: del self.video_writers[name]
            except Exception as e:
                logging.error(f"释放 {name} 写入器出错: {e}")
                if name in self.video_writers: del self.video_writers[name]
                
        self.video_writers.clear()
        csv_path = os.path.join(self.experiment_path, "gaze_data.csv")
        try:
            with open(csv_path, 'w', newline='') as f:
                csv_writer = csv.writer(f)
                header = ["timestamp", "gaze_x", "gaze_y", "left_pupil_x", "left_pupil_y", "left_axis_major", "left_axis_minor", "left_angle", "right_pupil_x", "right_pupil_y", "right_axis_major", "right_axis_minor", "right_angle"]
                csv_writer.writerow(header)
                csv_writer.writerows(self.experiment_data)
            logging.info(f"眼动数据已保存: {csv_path}")
        except Exception as e: logging.error(f"保存眼动数据失败: {e}")
        # ### update_button_states() 会在 GUI 线程中被 update_gui 自动调用
        logging.info("实验结束。")

    def record_experiment_data(self, all_frames):
        """ (移植自 complex_app.py) """
        if all_frames is None: return
        left_frame = all_frames.get("raw_left")
        right_frame = all_frames.get("raw_right")
        scene_frame_to_write = all_frames.get("main")
        try:
            writer_left = self.video_writers.get("left_eye")
            if writer_left and writer_left.isOpened() and left_frame is not None: writer_left.write(left_frame)
            writer_right = self.video_writers.get("right_eye")
            if writer_right and writer_right.isOpened() and right_frame is not None:
                # 在写入前旋转右眼摄像头180度
                rotated_right_frame = cv2.rotate(right_frame, cv2.ROTATE_180)
                writer_right.write(rotated_right_frame)
            writer_scene = self.video_writers.get("scene")
            if writer_scene and writer_scene.isOpened() and scene_frame_to_write is not None: writer_scene.write(scene_frame_to_write)
        except Exception as e: logging.exception("写入视频帧出错")
        
        # ### 注意：这里 self.tracker_worker.tracker 已改为 self.tracker
        if self.tracker:
            tracker = self.tracker
            if tracker and (tracker.state == TrackerState.TRACKING or tracker.state == TrackerState.VALIDATING):
                gaze = tracker.gaze_estimator.last_gaze_point
                left_f = tracker.last_left_features
                right_f = tracker.last_right_features
                if gaze is not None and left_f is not None and not np.all(left_f == 0) and right_f is not None and not np.all(right_f == 0):
                    ts_string = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    row = [ts_string, gaze[0], gaze[1]] + left_f.tolist() + right_f.tolist()
                    self.experiment_data.append(row)
    ### ==================================================================== ###
    ### 步骤1 优化结束 ###
    ### ==================================================================== ###

    ### ==================================================================== ###
    ### v4.0 修复: 为状态更改添加线程安全的槽函数
    ### ==================================================================== ###
    @pyqtSlot()
    def on_request_calibration(self):
        if self.tracker:
            logging.info("Worker 收到请求: on_request_calibration")
            self.tracker.start_calibration()

    @pyqtSlot()
    def on_request_add_point(self):
        if self.tracker and self.tracker.state == TrackerState.CALIBRATING:
            logging.info("Worker 收到请求: on_request_add_point")
            if not self.tracker.add_calibration_point():
                if not self.tracker.is_collecting_samples:
                    # 从工作线程安全地发送错误
                    logging.warning("采集失败：目标标记未检测到。")
                    self.tracker_error.emit("采集失败", "目标标记未检测到。")

    @pyqtSlot()
    def on_request_finish_calibration(self):
        if self.tracker and self.tracker.state == TrackerState.CALIBRATING:
            logging.info("Worker 收到请求: on_request_finish_calibration")
            if self.tracker.finish_calibration():
                self.tracker_info.emit("成功", "校准完成，进入验证模式。")
            else:
                min_points = len(self.tracker.CALIBRATION_TARGET_IDS)
                if not self.tracker.is_collecting_samples:
                    self.tracker_error.emit("失败", f"数据点不足 ({self.tracker.calibration_point_count}/{min_points})。")

    @pyqtSlot()
    def on_request_start_tracking(self):
        if self.tracker and self.tracker.state == TrackerState.VALIDATING:
            logging.info("Worker 收到请求: on_request_start_tracking")
            self.tracker.start_tracking()
    ### ==================================================================== ###

    def run(self):
        try:
            self.tracker = EyeTracker()
            self.tracker_initialized.emit(True, "追踪器初始化完成。")
            self.is_running = True
            target_interval = 1.0 / FPS if FPS > 0 else 0.033
            last_emit_time = 0
            min_emit_interval = 0.03 # 限制 GUI 更新频率，例如最多约 33 FPS

            while self.is_running:
                loop_start_time = time.perf_counter()
                try:
                    # ### 步骤 2A: get_all_frames 现在非常快
                    all_frames = self.tracker.get_all_frames()

                    ### ==================================================== ###
                    ### 步骤1 优化：直接在工作线程中记录数据 ###
                    ### ==================================================== ###
                    if self.is_experiment_running:
                        # ### 步骤 2A: 现在这个记录调用会以高帧率 (如 30FPS) 发生
                        self.record_experiment_data(all_frames)
                    ### ==================================================== ###

                    current_time = time.perf_counter()
                    if self.is_running and (current_time - last_emit_time >= min_emit_interval):
                        ### ==================================================== ###
                        ### 步骤1 优化：只发送 GUI 需要的帧，不发送 raw 帧 ###
                        ### ==================================================== ###
                        gui_frames = {
                            "main": all_frames.get("main"),
                            "left": all_frames.get("left"),
                            "right": all_frames.get("right")
                        }
                        self.new_frames.emit(gui_frames) # 只发射显示帧
                        last_emit_time = current_time
                        
                except Exception as frame_error:
                    logging.exception(f"处理帧时出错:")

                loop_end_time = time.perf_counter()
                processing_time = loop_end_time - loop_start_time
                sleep_time = max(0.001, target_interval - processing_time) # 保证至少 sleep 1ms
                time.sleep(sleep_time)
                
                # ### v4.2 修复: 手动处理此线程的事件队列
                # ### 否则，从 GUI 线程发来的信号 (如 on_request_calibration) 永远不会被处理
                QApplication.processEvents()

        except IOError as e:
            error_msg = f"摄像头初始化失败: {e}"
            logging.error(error_msg)
            self.tracker_initialized.emit(False, error_msg)
        except Exception as e:
            error_msg = f"追踪器启动时发生严重错误: {e}"
            logging.exception("TrackerWorker 运行期间发生严重错误:")
            self.tracker_initialized.emit(False, error_msg)
        finally:
            logging.info("TrackerWorker run 方法结束。")
            
            ### ==================================================== ###
            ### 步骤1 优化：确保线程退出时实验也结束 ###
            ### ==================================================== ###
            if self.is_experiment_running:
                logging.warning("线程终止，强制结束实验...")
                self.end_experiment()
            ### ==================================================== ###

            if self.tracker:
                try:
                    self.tracker.stop()
                except Exception as stop_e:
                    logging.error(f"停止 tracker 时出错: {stop_e}")

    def stop(self):
        logging.info("TrackerWorker 收到停止请求。")
        self.is_running = False


class EyeTrackerPiGUI(QMainWindow):
    """ (移植自 complex_app.py，带九点校准的GUI逻辑) """

    ### ================================================================ ###
    ### v4.0 修复: 为状态更改添加新信号
    ### ================================================================ ###
    request_calibration = pyqtSignal()
    request_add_point = pyqtSignal()
    request_finish_calibration = pyqtSignal()
    request_start_tracking = pyqtSignal()
    ### ================================================================ ###
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("眼动追踪 [AI版]")
        
        # vvvvvv   这是你请求的修改   vvvvvv
        self.setFixedSize(848, 410)
        # ^^^^^^   这是你请求的修改   ^^^^^^
        
        try:
            screen = QApplication.primaryScreen()
            if screen:
                screen_geometry = screen.availableGeometry()
                # 修改：使窗口在848x480屏幕上顶部居中
                top_left_x = screen_geometry.left() + (screen_geometry.width() - self.frameGeometry().width()) // 2
                top_left_y = screen_geometry.top()
                self.move(top_left_x, top_left_y)
            else: raise ValueError("无法获取主屏幕")
        except Exception as e:
            logging.warning(f"无法设置窗口位置: {e}。")
            # 回退：如果无法获取屏幕信息，则使用 (0, 0)
            self.setGeometry(0, 0, 848, 480)

        self.tracker_worker = None
        self.tracker_thread = None
        
        ### ==================================================== ###
        ### 步骤1 优化：删除 GUI 中的实验变量 ###
        ### ==================================================== ###
        # self.is_experiment_running = False
        # self.experiment_data = []
        # self.video_writers = {}
        # self.experiment_path = ""
        ### ==================================================== ###
        
        self.eye_container_widget = None # 用于动态宽度
        self.setup_logging()
        self.create_widgets()
        self.setup_layout()
        self.setup_styles()
        self.show_placeholders()
        logging.info("状态: 等待启动...")

    def setup_logging(self):
        self.log_handler = QtLogHandler()
        self.log_handler.new_log_record.connect(self.update_status_bar, Qt.ConnectionType.QueuedConnection)
        logging.getLogger().addHandler(self.log_handler)

    def create_widgets(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.start_button = QPushButton("启动")
        self.calib_button = QPushButton("校准")
        self.finish_calib_button = QPushButton("完成")
        self.exp_start_button = QPushButton("实验")
        self.exp_end_button = QPushButton("结束")

        # 设置按钮的尺寸策略，使其水平扩展
        for button in [self.start_button, self.calib_button, self.finish_calib_button, self.exp_start_button, self.exp_end_button]:
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        
        ### ==================================================== ###
        ### v4.0 修复: 按钮点击现在只发射信号
        ### ==================================================== ###
        self.start_button.clicked.connect(self.start_hardware)
        self.calib_button.clicked.connect(self.toggle_calibration)
        self.finish_calib_button.clicked.connect(self.finish_calibration)
        
        # 实验按钮的连接移至 on_tracker_initialized
        
        self.scene_cam_label = QLabel()
        self.left_eye_label = QLabel()
        self.right_eye_label = QLabel()
        for label in [self.scene_cam_label, self.left_eye_label, self.right_eye_label]:
            label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("background-color: #333333; color: white;")
            label.setText("未连接")
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.calib_button.setEnabled(False)
        self.finish_calib_button.setEnabled(False)
        self.exp_start_button.setEnabled(False)
        self.exp_end_button.setEnabled(False)

    def setup_layout(self):
        main_layout = QVBoxLayout(self.central_widget)
        control_layout = QHBoxLayout()
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.calib_button)
        control_layout.addWidget(self.finish_calib_button)
        control_layout.addWidget(self.exp_start_button)
        control_layout.addWidget(self.exp_end_button)
        main_layout.addLayout(control_layout)
        video_container_layout = QHBoxLayout()
        video_container_layout.setContentsMargins(0, 0, 0, 0)
        video_container_layout.setSpacing(0)
        video_container_layout.addWidget(self.scene_cam_label, 2) # 场景占 2/3

        self.eye_container_widget = QWidget()
        # 修改：初始宽度基于新的窗口大小
        eye_container_width = self.width() // 3
        self.eye_container_widget.setFixedWidth(eye_container_width) # 初始宽度

        eye_layout = QVBoxLayout()
        eye_layout.setContentsMargins(0, 0, 0, 0)
        eye_layout.setSpacing(0)
        eye_layout.addWidget(self.left_eye_label)
        eye_layout.addWidget(self.right_eye_label)
        self.eye_container_widget.setLayout(eye_layout)

        video_container_layout.addWidget(self.eye_container_widget, 1) # 眼睛占 1/3
        main_layout.addLayout(video_container_layout)

    def setup_styles(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #F0F0F0; }
            QPushButton { background-color: #808080; color: white; font-weight: bold; padding: 5px; border: none; border-radius: 3px; }
            QPushButton:hover { background-color: #909090; }
            QPushButton:pressed { background-color: #ADD8E6; }
            QPushButton:disabled { background-color: #C0C0C0; color: #A9A9A9; }
            QStatusBar { font-size: 10px; color: #555; }
            QLabel { background-color: #333333; color: white; }
        """)

    def start_hardware(self):
        logging.info("正在初始化眼动追踪器...")
        self.start_button.setEnabled(False)
        self.start_button.setText("启动中...")

        # 检查旧线程
        if self.tracker_thread and self.tracker_thread.isRunning():
            logging.warning("旧线程仍在运行，尝试停止...")
            if self.tracker_worker: self.tracker_worker.stop()
            self.tracker_thread.quit()
            if not self.tracker_thread.wait(1500):
                logging.error("无法停止旧线程。")
                self.start_button.setEnabled(True); self.start_button.setText("启动")
                QMessageBox.critical(self, "错误", "无法停止之前的追踪器线程，请重启应用。")
                return
            else:
                logging.info("旧线程已停止。")
                self.tracker_thread = None
                self.tracker_worker = None

        self.tracker_thread = QThread(self)
        self.tracker_worker = TrackerWorker()
        self.tracker_worker.moveToThread(self.tracker_thread)
        self.tracker_worker.new_frames.connect(self.update_gui, Qt.ConnectionType.QueuedConnection)
        self.tracker_worker.tracker_initialized.connect(self.on_tracker_initialized, Qt.ConnectionType.QueuedConnection)
        self.tracker_thread.finished.connect(self.on_tracker_thread_finished)
        self.tracker_thread.started.connect(self.tracker_worker.run)
        self.tracker_thread.start()

    def on_tracker_initialized(self, success, message):
        logging.info(f"Tracker initialized: success={success}, message={message}")
        self.start_button.setText("启动")
        if success:
            self.calib_button.setEnabled(True)
            self.start_button.setEnabled(False)
            logging.info("初始化成功，请校准。")
            
            ### ==================================================== ###
            ### v4.0 修复: 连接所有信号和槽
            ### ==================================================== ###
            try:
                # 先断开旧的连接 (如果有的话)
                self.exp_start_button.clicked.disconnect()
                self.exp_end_button.clicked.disconnect()
                self.request_calibration.disconnect()
                self.request_add_point.disconnect()
                self.request_finish_calibration.disconnect()
                self.request_start_tracking.disconnect()
                self.tracker_worker.tracker_error.disconnect()
                self.tracker_worker.tracker_info.disconnect()
            except TypeError: 
                pass # 忽略断开连接失败
            
            if self.tracker_worker:
                # 实验按钮 -> Worker 槽
                self.exp_start_button.clicked.connect(self.tracker_worker.start_experiment)
                self.exp_end_button.clicked.connect(self.tracker_worker.end_experiment)
                
                # GUI 请求信号 -> Worker 槽
                self.request_calibration.connect(self.tracker_worker.on_request_calibration)
                self.request_add_point.connect(self.tracker_worker.on_request_add_point)
                self.request_finish_calibration.connect(self.tracker_worker.on_request_finish_calibration)
                self.request_start_tracking.connect(self.tracker_worker.on_request_start_tracking)

                # Worker 反馈信号 -> GUI 槽
                self.tracker_worker.tracker_error.connect(self.on_tracker_error)
                self.tracker_worker.tracker_info.connect(self.on_tracker_info)
            ### ==================================================== ###
            
            # 动态调整眼部容器宽度
            QTimer.singleShot(100, self.adjust_eye_container_width)
        else:
            QMessageBox.critical(self, "错误", message)
            self.start_button.setEnabled(True)
            if self.tracker_thread and self.tracker_thread.isRunning():
                if self.tracker_worker: self.tracker_worker.stop()
                self.tracker_thread.quit()
                self.tracker_thread.wait(500)
            self.tracker_thread = None
            self.tracker_worker = None

    def adjust_eye_container_width(self):
        """延迟执行的宽度调整函数"""
        if self.tracker_worker and self.tracker_worker.tracker and self.eye_container_widget:
            try:
                eye_w = self.tracker_worker.tracker.eye_cam_width
                scene_cam = self.tracker_worker.tracker.scene_cam
                if scene_cam:
                    scene_w = scene_cam.width
                    total_cam_width = eye_w + scene_w
                    video_layout_item = self.centralWidget().layout().itemAt(1)
                    if video_layout_item:
                        video_layout = video_layout_item.layout()
                        available_width = video_layout.geometry().width() if video_layout else self.width()

                        if total_cam_width > 0 and available_width > 0:
                            eye_container_width = int(available_width * (eye_w / total_cam_width))
                            eye_container_width = max(50, eye_container_width)
                            self.eye_container_widget.setFixedWidth(eye_container_width)
                            logging.info(f"动态调整眼部容器宽度为: {eye_container_width}")
                        else:
                            self.eye_container_widget.setFixedWidth(self.width() // 3)
                    else:
                        self.eye_container_widget.setFixedWidth(self.width() // 3)
                else:
                    self.eye_container_widget.setFixedWidth(self.width() // 3)
            except Exception as e:
                logging.exception("动态调整眼部容器宽度时出错:")
                self.eye_container_widget.setFixedWidth(self.width() // 3)

    def on_tracker_thread_finished(self):
        logging.info("追踪器线程已结束。")
        if self.tracker_worker:
            try: self.tracker_worker.new_frames.disconnect(self.update_gui)
            except TypeError: pass
            try: self.tracker_worker.tracker_initialized.disconnect(self.on_tracker_initialized)
            except TypeError: pass
            ### v4.0 修复: 断开所有连接
            try:
                self.tracker_worker.tracker_error.disconnect(self.on_tracker_error)
                self.tracker_worker.tracker_info.disconnect(self.on_tracker_info)
            except TypeError: pass
            
        self.tracker_thread = None
        self.tracker_worker = None
        if self.isVisible():
            if hasattr(self, 'start_button') and self.start_button:
                self.start_button.setEnabled(True)
                self.start_button.setText("启动")
                logging.info("启动按钮已重新启用。")
            self.update_button_states()
            self.show_placeholders()

    ### ==================================================================== ###
    ### 步骤1 优化：删除 start_experiment, end_experiment, 
    ###          和 record_experiment_data
    ### 它们已被移至 TrackerWorker
    ### ==================================================================== ###
    # def start_experiment(self):
    #     ...
    # def end_experiment(self):
    #     ...
    # def record_experiment_data(self, all_frames):
    #     ...
    ### ==================================================================== ###

    ### ==================================================================== ###
    ### v4.0 修复: 添加用于 worker 反馈的槽函数
    ### ==================================================================== ###
    @pyqtSlot(str, str)
    def on_tracker_error(self, title, message):
        QMessageBox.critical(self, title, message)

    @pyqtSlot(str, str)
    def on_tracker_info(self, title, message):
        QMessageBox.information(self, title, message)
    ### ==================================================================== ###

    def toggle_calibration(self):
        """ (移植自 complex_app.py) """
        if not self.tracker_worker or not self.tracker_worker.tracker:
            QMessageBox.critical(self, "错误", "请先点击启动。")
            return
            
        ### ==================================================================== ###
        ### v4.0 修复: 不再直接调用 tracker, 而是发射信号
        ### 我们仍然需要读取状态来决定发射哪个信号
        ### ==================================================================== ###
        tracker = self.tracker_worker.tracker # 读取状态是相对安全的
        state = tracker.state
        try:
            if state == TrackerState.CALIBRATING:
                if not tracker.is_collecting_samples:
                    self.request_add_point.emit()
                # else: (如果正在采集, 按钮是禁用的, 不会到这里)
            
            elif state == TrackerState.VALIDATING: 
                self.request_start_tracking.emit()
            
            elif state == TrackerState.IDLE: 
                self.request_calibration.emit()
            
            elif state == TrackerState.TRACKING: 
                self.request_calibration.emit()
                
        except Exception as e:
            logging.error(f"发射校准信号时出错: {e}")
            QMessageBox.critical(self, "错误", f"操作失败: {e}")
        
        # update_button_states() 将由 worker 线程的下一帧自动触发
        # self.update_button_states() # 立即更新是可选的，但可能导致状态不同步
        ### ==================================================================== ###


    def finish_calibration(self):
        """ (移植自 complex_app.py) """
        if not self.tracker_worker or not self.tracker_worker.tracker: 
            return
            
        ### ==================================================================== ###
        ### v4.0 修复: 不再直接调用 tracker, 而是发射信号
        ### ==================================================================== ###
        if self.tracker_worker.tracker.state == TrackerState.CALIBRATING:
            if not self.tracker_worker.tracker.is_collecting_samples:
                self.request_finish_calibration.emit()
        ### ==================================================================== ###


    @pyqtSlot(dict)
    def update_gui(self, all_frames):
        """ 
        (移植自 complex_app.py，带旋转逻辑) 
        ### 步骤1 优化：all_frames 现在只包含 'main', 'left', 'right'
        """
        if all_frames is None: return
        scene_display_frame = all_frames.get("main")
        left_eye_frame = all_frames.get("left")
        right_eye_frame = all_frames.get("right")
        
        # 旋转左眼 (物理右眼, CAM ID 0)
        #if left_eye_frame is not None:
            #try: left_eye_frame = cv2.rotate(left_eye_frame, cv2.ROTATE_180)
            #except cv2.error as rot_e: logging.warning(f"旋转左眼帧失败: {rot_e}")
        #不旋转右眼(物理左眼, CAM ID 2)
        if right_eye_frame is not None:
            try: right_eye_frame = cv2.rotate(right_eye_frame, cv2.ROTATE_180)
            except cv2.error as rot_e: logging.warning(f"旋转右眼帧失败: {rot_e}")
            
        ### ==================================================== ###
        ### 步骤1 优化：删除数据记录调用
        ### ==================================================== ###
        # if self.is_experiment_running: self.record_experiment_data(all_frames)
        ### ==================================================== ###
        
        self.update_image_label(self.scene_cam_label, scene_display_frame)
        self.update_image_label(self.left_eye_label, left_eye_frame)
        self.update_image_label(self.right_eye_label, right_eye_frame)
        
        ### ==================================================================== ###
        ### v4.0 修复: 这是唯一安全更新按钮状态的地方
        ### ==================================================================== ###
        self.update_button_states()


    def update_button_states(self):
        """ 
        (移植自 complex_app.py) 
        v4.0: 此函数现在是唯一从主线程读取 tracker 状态的地方,
        并且只在 update_gui (由 worker 线程的 new_frames 信号触发) 时调用。
        """
        start_enabled, calib_enabled, finish_enabled, exp_start_enabled, exp_end_enabled = True, False, False, False, False
        calib_text = "校准"
        
        is_exp_running = False
        if self.tracker_worker:
            # 安全地读取 worker 的属性
            is_exp_running = self.tracker_worker.is_experiment_running

        if self.tracker_worker and self.tracker_worker.tracker:
            # 安全地读取 tracker 的属性
            tracker = self.tracker_worker.tracker 
            state, is_trained, is_collecting = tracker.state, tracker.is_calibration_finished(), tracker.is_collecting_samples
            
            start_enabled = False # 一旦启动，启动按钮就禁用
            
            if state == TrackerState.CALIBRATING:
                calib_enabled, finish_enabled = True, True
                if is_collecting:
                    calib_text, calib_enabled, finish_enabled = f"采集中({len(tracker.feature_buffer)}/{tracker.SAMPLES_TO_COLLECT})", False, False
                else:
                    if tracker.current_target_index >= len(tracker.CALIBRATION_TARGET_IDS): calib_text, calib_enabled = f"已采完({tracker.calibration_point_count})", False
                    else: calib_text = f"采点 {tracker.CALIBRATION_TARGET_IDS[tracker.current_target_index]} ({tracker.calibration_point_count})"
            elif state == TrackerState.VALIDATING: 
                calib_text, calib_enabled = "开始追踪", True
            elif state == TrackerState.TRACKING:
                calib_text, calib_enabled = "重新校准", True
                if is_trained: exp_start_enabled, exp_end_enabled = not is_exp_running, is_exp_running
            else: calib_enabled = (self.tracker_worker is not None) # IDLE
        
        # 允许重启
        start_enabled = not (self.tracker_thread and self.tracker_thread.isRunning())
        if self.tracker_thread and self.tracker_thread.isRunning():
            if self.start_button.text() == "启动": self.start_button.setText("运行中")
        else:
            if self.start_button.text() != "启动": self.start_button.setText("启动")

        if is_exp_running: calib_enabled, finish_enabled = False, False
        
        self.start_button.setEnabled(start_enabled)
        self.calib_button.setEnabled(calib_enabled); self.calib_button.setText(calib_text)
        self.finish_calib_button.setEnabled(finish_enabled)
        self.exp_start_button.setEnabled(exp_start_enabled)
        self.exp_end_button.setEnabled(exp_end_enabled)

    def show_placeholders(self):
        """ (移植自 complex_app.py) """
        placeholder_text = "等待连接..."
        for label in [self.scene_cam_label, self.left_eye_label, self.right_eye_label]:
            pixmap = QPixmap(label.size())
            if pixmap.isNull(): pixmap = QPixmap(160, 120)
            pixmap.fill(Qt.GlobalColor.darkGray)
            label.setPixmap(pixmap)
            label.setText(placeholder_text)

    def update_image_label(self, label, frame):
        """ (移植自 complex_app.py，带 rgbSwapped) """
        if frame is None:
            if not label.pixmap() or label.pixmap().isNull() or label.text() == "等待连接...":
                pixmap = QPixmap(label.size())
                if pixmap.isNull():
                    default_w, default_h = (320, 240) if label == self.scene_cam_label else (160, 120)
                    pixmap = QPixmap(default_w, default_h)
                pixmap.fill(Qt.GlobalColor.darkGray)
                label.setPixmap(pixmap)
                label.setText("无信号")
            return
        try:
            if label.text(): label.setText("")
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            if ch == 3: # Color image
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_format = QImage.Format.Format_RGB888
            else: # Grayscale
                img_format = QImage.Format.Format_Grayscale8
            
            qt_image = QImage(frame.data, w, h, bytes_per_line, img_format).copy()


            pixmap = QPixmap.fromImage(qt_image)
            
            target_width, target_height = label.width(), label.height()
            if target_width > 0 and target_height > 0:
                scaled_pixmap = pixmap.scaled(target_width, target_height, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.FastTransformation) # <-- v5.0: 优化
                label.setPixmap(scaled_pixmap)
            else:
                default_w, default_h = (320, 240) if label == self.scene_cam_label else (160, 120)
                label.setPixmap(pixmap.scaled(default_w, default_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)) # <-- v5.0: 优化
        except Exception as e:
            logging.exception(f"更新图像标签时出错")
            try: 
                pixmap = QPixmap(label.size())
                if pixmap.isNull(): pixmap = QPixmap(160, 120)
                pixmap.fill(Qt.GlobalColor.red)
                label.setPixmap(pixmap)
                label.setText("渲染错误")
            except Exception: pass


    @pyqtSlot(str)
    def update_status_bar(self, message):
        if self.status_bar:
            self.status_bar.showMessage(message, 3000)

    def closeEvent(self, event):
        """ (移植自 complex_app.py) """
        logging.info("收到关闭请求...")
        try:
            for button in [self.start_button, self.calib_button, self.finish_calib_button, self.exp_start_button, self.exp_end_button]:
                if button: button.setEnabled(False)
            if self.status_bar: self.status_bar.showMessage("正在关闭...")
            QApplication.processEvents()
        except Exception as ui_e:
            logging.error(f"关闭前禁用 UI 出错: {ui_e}")

        ### ==================================================== ###
        ### 步骤1 优化：删除此处的 end_experiment() 调用
        ### 它已自动移至 TrackerWorker.run 的 finally 块中
        ### ==================================================== ###
        # if self.is_experiment_running:
        #     logging.info("结束实验...")
        #     try:
        #
        # _button         self.end_experiment()
        #         QApplication.processEvents()
        #     except Exception as exp_e:
        #         logging.exception("结束实验时出错")
        ### ==================================================== ###

        if self.tracker_thread and self.tracker_thread.isRunning():
            logging.info("请求线程停止...")
            try:
                if self.tracker_worker: self.tracker_worker.stop()
                self.tracker_thread.quit()
                logging.info("等待线程结束...")
                # ### 步骤 2A: 停止 worker 会导致 tracker.stop() 被调用
                # ### tracker.stop() 会 join 子进程，这可能需要更长时间
                if not self.tracker_thread.wait(5000): # ### 增加到 5 秒
                    logging.warning("线程在5秒内未停止。")
                else: 
                    logging.info("线程已停止。")
            except Exception as thread_e:
                logging.exception("停止追踪器线程时出错:")
        
        self.tracker_worker, self.tracker_thread = None, None

        try:
            model_path = 'calibration_model.pkl'
            if os.path.exists(model_path):
                try: os.remove(model_path); logging.info(f"已删除模型: {model_path}")
                except Exception as e_remove: logging.error(f"删除模型出错: {e_remove}")
        except Exception as e_cleanup: logging.error(f"清理模型出错: {e_cleanup}")

        logging.info("关闭完成。")
        event.accept()

# =============================================================================
# 8. MAIN EXECUTION
# =============================================================================
if __name__ == '__main__':
    """ (移植自 complex_app.py，带日志和异常钩子) """
    
    # ### 步骤 2A: 确保多进程在 Windows/macOS 上安全启动
    # ### (此调用必须在 main 启动的早期)
    try:
        multiprocessing.freeze_support()
    except Exception:
        pass # 在 Linux 上可能不需要
    # --- 关键检查：AI 模型文件是否存在 ---
    if not os.path.exists(ONNX_MODEL_PATH):
        print(f"!!! 致命错误: 瞳孔检测模型 '{ONNX_MODEL_PATH}' 未找到。")
        print(f"请确保 'best.int8.onnx' 文件与此脚本位于同一目录中。")
        # 尝试显示一个GUI错误
        try:
            app_check = QApplication(sys.argv)
            QMessageBox.critical(None, "致命错误", f"模型文件 '{ONNX_MODEL_PATH}' 未找到。\n程序即将退出。")
        except Exception:
            pass
        sys.exit(-1)
    # --- 检查结束 ---

    log_format = '%(asctime)s - %(levelname)s - [%(threadName)s] %(filename)s:%(lineno)d - %(message)s'
    from logging.handlers import RotatingFileHandler
    log_filename = 'eyetracker_app.log'
    max_log_size, backup_count = 5*1024*1024, 2
    root_logger = logging.getLogger(); root_logger.setLevel(logging.INFO)
    for handler in root_logger.handlers[:]: root_logger.removeHandler(handler); handler.close()
    file_handler = RotatingFileHandler(log_filename, maxBytes=max_log_size, backupCount=backup_count, encoding='utf-8')
    file_formatter = logging.Formatter(log_format)
    file_handler.setFormatter(file_formatter); root_logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(); console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter); root_logger.addHandler(console_handler)
    logging.info("应用程序启动...")

    app = QApplication(sys.argv)
    
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt): sys.__excepthook__(exc_type, exc_value, exc_traceback); return
        logging.error("捕获到未处理异常:", exc_info=(exc_type, exc_value, exc_traceback))
        try:
            QMessageBox.critical(None, "严重错误", f"发生未捕获的异常: {exc_value}\n请查看日志文件 eyetracker_app.log 获取详细信息。")
        except Exception:
            pass
        logging.info("因未处理异常退出程序。")
        sys.exit(1)

    sys.excepthook = handle_exception

    try:
        main_window = EyeTrackerPiGUI()
        main_window.show()
        exit_code = app.exec()
        logging.info(f"应用程序退出，代码: {exit_code}")
        sys.exit(exit_code)
    except Exception as start_e:
        logging.exception("启动或运行期间出错")
        try: QMessageBox.critical(None, "启动错误", f"启动失败: {start_e}")
        except Exception: pass
        sys.exit(1)