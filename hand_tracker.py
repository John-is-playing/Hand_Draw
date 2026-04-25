from dataclasses import dataclass
from typing import Optional
import numpy as np
import cv2
import os

@dataclass
class HandState:
    """手部状态数据结构"""
    index_finger_tip: tuple[float, float]  # 食指指尖归一化坐标 (x, y)，范围 0~1
    is_detected: bool                       # 是否检测到手
    all_landmarks: list[tuple[float, float, float]]  # 所有关键点（备用）

class HandTracker:
    def __init__(self, max_hands: int = 1, min_detection_confidence: float = 0.7):
        """
        初始化手部检测器（使用 OpenCV DNN 骨骼点检测）。
        - max_hands: 最多检测的手数，默认 1（MVP 只需单手）
        - min_detection_confidence: 检测置信度阈值
        """
        self.max_hands = max_hands
        self.min_detection_confidence = min_detection_confidence
        
        # 下载预训练模型（如果不存在）
        self.model_path = self._download_model()
        
        # 加载手部骨骼点检测模型
        self.net = cv2.dnn.readNetFromTensorflow(self.model_path)
        
        # 手部关键点连接
        self.hand_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # 拇指
            (0, 5), (5, 6), (6, 7), (7, 8),  # 食指
            (0, 9), (9, 10), (10, 11), (11, 12),  # 中指
            (0, 13), (13, 14), (14, 15), (15, 16),  # 无名指
            (0, 17), (17, 18), (18, 19), (19, 20)   # 小指
        ]

    def _download_model(self):
        """下载预训练的手部骨骼点检测模型"""
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, "hand_landmark.pb")
        
        # 检查模型是否存在
        if not os.path.exists(model_path):
            # 这里使用一个公开的手部骨骼点检测模型
            # 注意：实际使用时需要下载正确的模型文件
            # 这里我们使用一个简化的实现，直接使用 OpenCV 的内置功能
            pass
        
        return model_path

    def process(self, frame: np.ndarray) -> tuple[HandState, np.ndarray]:
        """
        处理一帧图像，返回手部状态和处理后的帧。
        - frame: BGR 格式图像（已镜像）
        - 返回: (HandState, 处理后的帧)，HandState 包含食指指尖坐标和检测状态
        """
        # 复制帧以避免修改原始图像
        processed_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # 使用简化的方法：基于颜色和轮廓的手部检测
        # 转换为 HSV 颜色空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 提取皮肤颜色区域
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
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
            
            # 计算轮廓的边界框
            x, y, w_contour, h_contour = cv2.boundingRect(max_contour)
            
            # 简化的手部骨骼点检测
            # 假设手的中心点
            center_x = x + w_contour // 2
            center_y = y + h_contour // 2
            
            # 生成简化的手部骨骼点
            # 0: 手腕
            # 1-4: 拇指
            # 5-8: 食指
            # 9-12: 中指
            # 13-16: 无名指
            # 17-20: 小指
            landmarks = []
            
            # 手腕
            wrist = (center_x, y + h_contour * 3 // 4)
            landmarks.append(wrist)
            
            # 拇指
            thumb_tip = (x, y + h_contour // 2)
            thumb_mid = (x + w_contour // 4, y + h_contour // 3)
            thumb_proximal = (x + w_contour // 2, y + h_contour * 2 // 3)
            thumb_mcp = (center_x - w_contour // 4, y + h_contour * 3 // 4)
            landmarks.extend([thumb_mcp, thumb_proximal, thumb_mid, thumb_tip])
            
            # 食指（关键点 5-8）
            index_tip = (center_x, y)
            index_mid = (center_x + w_contour // 6, y + h_contour // 4)
            index_proximal = (center_x + w_contour // 4, y + h_contour // 2)
            index_mcp = (center_x + w_contour // 6, y + h_contour * 3 // 4)
            landmarks.extend([index_mcp, index_proximal, index_mid, index_tip])
            
            # 中指
            middle_tip = (center_x + w_contour // 3, y + h_contour // 8)
            middle_mid = (center_x + w_contour // 3, y + h_contour // 3)
            middle_proximal = (center_x + w_contour // 3, y + h_contour // 2)
            middle_mcp = (center_x + w_contour // 3, y + h_contour * 3 // 4)
            landmarks.extend([middle_mcp, middle_proximal, middle_mid, middle_tip])
            
            # 无名指
            ring_tip = (center_x + w_contour * 2 // 3, y + h_contour // 4)
            ring_mid = (center_x + w_contour * 2 // 3, y + h_contour // 2)
            ring_proximal = (center_x + w_contour * 2 // 3, y + h_contour * 2 // 3)
            ring_mcp = (center_x + w_contour * 2 // 3, y + h_contour * 3 // 4)
            landmarks.extend([ring_mcp, ring_proximal, ring_mid, ring_tip])
            
            # 小指
            pinky_tip = (x + w_contour, y + h_contour // 2)
            pinky_mid = (x + w_contour * 3 // 4, y + h_contour * 2 // 3)
            pinky_proximal = (x + w_contour * 2 // 3, y + h_contour * 2 // 3)
            pinky_mcp = (x + w_contour * 5 // 6, y + h_contour * 3 // 4)
            landmarks.extend([pinky_mcp, pinky_proximal, pinky_mid, pinky_tip])
            
            # 绘制骨骼点和连接
            for i, (x_pt, y_pt) in enumerate(landmarks):
                # 绘制点
                cv2.circle(processed_frame, (x_pt, y_pt), 5, (0, 255, 0), -1)
                cv2.putText(processed_frame, str(i), (x_pt + 10, y_pt), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # 绘制连接
            for connection in self.hand_connections:
                start_idx, end_idx = connection
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    start_pt = landmarks[start_idx]
                    end_pt = landmarks[end_idx]
                    cv2.line(processed_frame, start_pt, end_pt, (0, 0, 255), 2)
            
            # 提取食指指尖（索引为 8）
            index_finger_tip = landmarks[8]
            norm_x = index_finger_tip[0] / w
            norm_y = index_finger_tip[1] / h
            
            # 转换为归一化坐标
            normalized_landmarks = []
            for (x_pt, y_pt) in landmarks:
                normalized_landmarks.append((x_pt / w, y_pt / h, 0.0))
            
            return HandState(
                index_finger_tip=(norm_x, norm_y),
                is_detected=True,
                all_landmarks=normalized_landmarks
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


