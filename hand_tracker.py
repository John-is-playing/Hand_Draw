from dataclasses import dataclass
from typing import Optional
import numpy as np
import cv2
import mediapipe as mp

@dataclass
class HandState:
    """手部状态数据结构"""
    index_finger_tip: tuple[float, float]  # 食指指尖归一化坐标 (x, y)，范围 0~1
    is_detected: bool                       # 是否检测到手
    all_landmarks: list[tuple[float, float, float]]  # 所有关键点（备用）

class HandTracker:
    def __init__(self, max_hands: int = 1, min_detection_confidence: float = 0.7):
        """
        初始化 MediaPipe Hands。
        - max_hands: 最多检测的手数，默认 1（MVP 只需单手）
        - min_detection_confidence: 检测置信度阈值
        """
        self.max_hands = max_hands
        self.min_detection_confidence = min_detection_confidence
        
        # 初始化 MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5
        )
        
        # 用于绘制手部关键点
        self.mp_drawing = mp.solutions.drawing_utils

    def process(self, frame: np.ndarray) -> tuple[HandState, np.ndarray]:
        """
        处理一帧图像，返回手部状态和处理后的帧。
        - frame: BGR 格式图像（已镜像）
        - 返回: (HandState, 处理后的帧)，HandState 包含食指指尖坐标和检测状态
        """
        # 复制帧以避免修改原始图像
        processed_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # 将 BGR 图像转换为 RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 处理图像
        results = self.hands.process(rgb_frame)
        
        # 初始化返回值
        index_finger_tip = (0, 0)
        is_detected = False
        all_landmarks = []
        
        # 检查是否检测到手
        if results.multi_hand_landmarks:
            # 只处理第一只手
            hand_landmarks = results.multi_hand_landmarks[0]
            is_detected = True
            
            # 提取食指指尖坐标（索引为 8）
            index_finger_landmark = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_finger_tip = (index_finger_landmark.x, index_finger_landmark.y)
            
            # 提取所有关键点
            for landmark in hand_landmarks.landmark:
                all_landmarks.append((landmark.x, landmark.y, landmark.z))
            
            # 绘制手部关键点和连接
            self.mp_drawing.draw_landmarks(
                processed_frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS
            )
        
        return HandState(
            index_finger_tip=index_finger_tip,
            is_detected=is_detected,
            all_landmarks=all_landmarks
        ), processed_frame

    def release(self) -> None:
        """释放 MediaPipe 资源。"""
        if hasattr(self, 'hands'):
            self.hands.close()


