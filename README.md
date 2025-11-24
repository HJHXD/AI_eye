# AI Eye Tracking System

这是一个基于深度学习和计算机视觉的眼动追踪系统，专为树莓派（Raspberry Pi）或其他 Linux 环境设计。该项目结合了 AI 瞳孔检测、多摄像头支持和 PyQt6 图形界面，提供实时的视线估计和数据记录功能。

## ✨ 主要特性

*   **AI 瞳孔检测**: 集成 ONNX 深度学习模型 (`best.int8.onnx`)，实现高精度的瞳孔识别。
*   **多摄像头支持**: 同时支持左眼、右眼和场景（前视）三个摄像头输入。
*   **高性能架构**:
    *   **多进程 AI 推理**: 利用 `multiprocessing` 并行化 AI 检测，避免阻塞主界面。
    *   **多线程 I/O**: 数据记录和视频流捕获在独立线程中运行。
    *   **ROI 追踪**: 智能感兴趣区域（ROI）追踪算法，大幅降低检测延迟。
*   **九点校准**: 内置基于 ArUco 标记的九点校准程序，使用多项式回归实现精确的视线映射。
*   **图形用户界面 (GUI)**: 基于 PyQt6 的现代化界面，实时显示眼部图像、检测框和校准状态。
*   **数据记录**: 自动将视线数据（Gaze Data）记录为 CSV 文件，便于后续分析。

## 🛠️ 环境要求

*   **硬件**:
    *   树莓派 4B/5 或其他 Linux 计算机
    *   3个 USB 摄像头（左眼、右眼、场景）
*   **操作系统**: Linux (推荐 Raspberry Pi OS)
*   **Python**: Python 3.8+

## 📦 安装指南

1.  **克隆项目**
    ```bash
    git clone <repository_url>
    cd AI_eye
    ```

2.  **安装依赖库**
    请确保安装了以下 Python 库：
    ```bash
    pip install opencv-python opencv-contrib-python numpy onnxruntime PyQt6 scikit-learn joblib
    ```
    *注意: 在树莓派上，某些库（如 PyQt6 或 opencv）可能需要通过 `apt` 安装系统包或使用特定的预编译版本。*

## 🚀 使用方法

### 1. 运行主程序
启动完整的眼动追踪系统（GUI）：
```bash
python main.py
```
*   **界面操作**:
    *   程序启动后会自动加载模型并打开摄像头。
    *   点击界面上的按钮进行校准或开始/停止记录。
    *   校准过程中，请注视屏幕上显示的 ArUco 标记。

### 2. 运行测试脚本
如果你只想测试摄像头和 AI 模型是否正常工作，可以运行轻量级的测试脚本：
```bash
python eye.py
```
*   该脚本会打开默认摄像头（ID 0），并显示瞳孔检测结果。
*   按 `q` 键退出。

## ⚙️ 配置说明

主要配置位于 `main.py` 文件的开头部分（或 `config.py`，如果存在）。你可以根据实际硬件修改以下参数：

*   **摄像头 ID**:
    ```python
    CAMERA_CONFIG = {
        "left_eye": 0,   # 左眼摄像头ID
        "right_eye": 2,  # 右眼摄像头ID
        "scene": 4       # 场景摄像头ID
    }
    ```
*   **模型路径**: `ONNX_MODEL_PATH = "best.int8.onnx"`
*   **校准参数**: 可调整 `CALIBRATION_SAMPLES_TO_COLLECT` 等参数。

## 📂 文件结构

*   `main.py`: 主程序入口，包含 GUI、多进程逻辑和校准算法。
*   `eye.py`: 独立的 AI 模型测试脚本。
*   `best.int8.onnx`: 量化后的瞳孔检测 AI 模型。
*   `best.onnx`: 原始 AI 模型。
*   `2025-xx-xx_.../`: 自动生成的日志文件夹，包含 `gaze_data.csv` 数据文件。

## 📝 注意事项

*   **摄像头顺序**: 如果程序启动报错或画面错乱，请检查 `CAMERA_CONFIG` 中的摄像头 ID 是否与物理连接一致。
*   **光照条件**: 良好的红外照明（对于眼部摄像头）有助于提高瞳孔检测的准确性。
