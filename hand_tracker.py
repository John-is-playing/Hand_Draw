from dataclasses import dataclass
from typing import Optional
import numpy as np
import cv2

@dataclass
class HandState:
    """手部状态数据结构"""
    index_finger_tip: tuple[float, float]  # 食指指尖归一化坐标 (x, y)，范围 0~1
    is_detected: bool                       # 是否检测到手
    all_landmarks: list[tuple[float, float, float]]  # 所有关键点（备用）

class HandTracker:
    def __init__(self, max_hands: int = 1, min_detection_confidence: float = 0.7):
        """
        初始化手部检测器（使用 OpenCV 颜色追踪）。
        - max_hands: 最多检测的手数，默认 1（MVP 只需单手）
        - min_detection_confidence: 检测置信度阈值
        """
        self.max_hands = max_hands
        self.min_detection_confidence = min_detection_confidence
        
        # 定义皮肤颜色范围（HSV）
        self.lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    def process(self, frame: np.ndarray) -> tuple[HandState, np.ndarray]:
        """
        处理一帧图像，返回手部状态和处理后的帧。
        - frame: BGR 格式图像
        - 返回: (HandState, 处理后的帧)，HandState 包含食指指尖坐标和检测状态
        """
        # 复制帧以避免修改原始图像
        processed_frame = frame.copy()
        
        # 转换为 HSV 颜色空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 提取皮肤颜色区域
        mask = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
        
        # 形态学操作，去除噪声
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 找到最大的轮廓（假设是手）
            max_contour = max(contours, key=cv2.contourArea)
            
            # 计算轮廓面积
            area = cv2.contourArea(max_contour)
            
            # 如果面积太小，不认为是手
            if area < 5000:
                return HandState(
                    index_finger_tip=(0, 0),
                    is_detected=False,
                    all_landmarks=[]
                ), processed_frame
            
            # 找到轮廓的凸包
            hull = cv2.convexHull(max_contour)
            
            # 找到最高点（假设是食指指尖）
            # 按 y 坐标排序，最小的点就是最高点
            hull_points = hull.reshape(-1, 2)
            if len(hull_points) > 0:
                # 找到最高点（y 坐标最小）
                highest_point = hull_points[np.argmin(hull_points[:, 1])]
                
                # 归一化坐标
                h, w = frame.shape[:2]
                norm_x = highest_point[0] / w
                norm_y = highest_point[1] / h
                
                # 在处理后的帧上绘制手部轮廓和指尖
                cv2.drawContours(processed_frame, [max_contour], -1, (0, 255, 0), 2)
                cv2.drawContours(processed_frame, [hull], -1, (0, 0, 255), 2)
                cv2.circle(processed_frame, tuple(highest_point), 10, (255, 0, 0), -1)
                
                return HandState(
                    index_finger_tip=(norm_x, norm_y),
                    is_detected=True,
                    all_landmarks=[(norm_x, norm_y, 0.0)]  # 简化处理，只返回指尖点
                ), processed_frame
        
        # 未检测到手
        return HandState(
            index_finger_tip=(0, 0),
            is_detected=False,
            all_landmarks=[]
        ), processed_frame

    def release(self) -> None:
        """释放资源。"""
        pass


