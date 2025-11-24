import cv2
import numpy as np
import onnxruntime as ort
import time

# --- 1. 配置参数 ---
MODEL_PATH = "best.int8.onnx"  # 确保你的 .onnx 文件和这个脚本在同一个文件夹
INPUT_SIZE = 320          # 必须和你训练时的 imgsz=320 一致!
CONF_THRESHOLD = 0.45     # 置信度阈值，只显示大于45%可能的框
NMS_THRESHOLD = 0.4       # 非极大值抑制，用于过滤重叠的框

# --- 2. 加载 ONNX 模型 ---
try:
    session = ort.InferenceSession(MODEL_PATH)
    print("ONNX 模型加载成功。")
    # 获取输入输出的名称
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"输入名称: {input_name}, 输出名称: {output_name}")
except Exception as e:
    print(f"!!! 错误：无法加载 ONNX 模型: {e}")
    exit()

# --- 3. 预处理函数 (信我的，这比YOLOv8官方的还简单) ---
def preprocess(frame):
    # a. 调整图像大小到 320x320
    img_resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    
    # b. 归一化 (0-255 -> 0.0-1.0)
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # c. 调整维度 HWC -> NCHW (1, 3, 320, 320)
    #    (Height, Width, Channel) -> (Batch, Channel, Height, Width)
    img_transposed = np.transpose(img_normalized, (2, 0, 1))
    
    # d. 增加一个 'Batch' 维度
    img_batch = np.expand_dims(img_transposed, axis=0)
    
    return img_batch

# --- 4. 打开摄像头 ---
# 提示: 0 可能是你的USB摄像头, 1 可能是你的IR摄像头。多试试
cap = cv2.VideoCapture(0) 
if not cap.isOpened():
    print("!!! 错误：无法打开摄像头。")
    exit()

print("摄像头已启动... 按 'q' 退出。")
print("-----------------------------------")

# --- 5. 实时循环 ---
while True:
    start_time = time.time() # 记录开始时间
    
    ret, frame = cap.read()
    if not ret:
        break
        
    frame_height, frame_width = frame.shape[:2]

    # --- 6. 图像预处理 ---
    input_tensor = preprocess(frame)

    # --- 7. 执行推理 ---
    # outputs[0] 的形状是 (1, 5, 8400) -> [x, y, w, h, confidence]
    outputs = session.run([output_name], {input_name: input_tensor})
    
    # --- 8. 后处理 ---
    # (这部分是YOLOv8输出的标准解析流程)
    detections = outputs[0][0].T  # 转置为 (8400, 5)
    
    boxes = []
    confidences = []

    # 过滤掉低置信度的检测
    for det in detections:
        confidence = det[4]
        if confidence > CONF_THRESHOLD:
            # 坐标是 (cx, cy, w, h)，且是 0-320 范围的
            cx, cy, w, h = det[:4]
            
            # 换算回原始图像 (frame) 的坐标
            x1 = int((cx - w / 2) * frame_width / INPUT_SIZE)
            y1 = int((cy - h / 2) * frame_height / INPUT_SIZE)
            x2 = int((cx + w / 2) * frame_width / INPUT_SIZE)
            y2 = int((cy + h / 2) * frame_height / INPUT_SIZE)

            boxes.append([x1, y1, x2, y2])
            confidences.append(float(confidence))

    # 使用 NMS 过滤重叠的框 (瞳孔只有一个，但保险起见)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
    
    best_roi = None
    if len(indices) > 0:
        # 我们只取置信度最高的那个
        best_index = indices[0]
        x1, y1, x2, y2 = boxes[best_index]
        
        # --- 关键！这是你的ROI (Region of Interest) ---
        # 坐标 [x1, y1, x2, y2]
        best_roi = (x1, y1, x2, y2) 
        
        # --- 9. 可视化 (画框) ---
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Pupil: {confidences[best_index]:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 计算并显示 FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # 显示结果
    cv2.imshow("Pupil Detector (AI)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
print("程序退出。")