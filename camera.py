import cv2
import numpy as np

class Camera:
    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480):
        """
        初始化摄像头。
        - camera_id: 摄像头设备编号，默认 0
        - width/height: 请求的分辨率
        """
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            print("错误: 无法打开摄像头")
            exit(1)
        
        # 设置分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        self.width = width
        self.height = height

    def read_frame(self) -> np.ndarray | None:
        """
        读取一帧图像。
        - 返回: BGR 格式的 numpy 数组 (height, width, 3)，失败返回 None
        """
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # 水平镜像翻转
        frame = cv2.flip(frame, 1)
        return frame

    def release(self) -> None:
        """释放摄像头资源。"""
        if self.cap.isOpened():
            self.cap.release()
