from dataclasses import dataclass
from typing import Optional
import numpy as np
import cv2

# 尝试导入 MediaPipe
try:
    import mediapipe as mp
    
    # 尝试不同的导入方式
    try:
        # 现代版本 MediaPipe
        from mediapipe.solutions import hands
    except ImportError:
        try:
            # 较旧版本 MediaPipe
            from mediapipe import hands
        except ImportError:
            # 非常旧的版本 MediaPipe
            hands = mp.hands
except ImportError:
    # MediaPipe 未安装
    raise ImportError("MediaPipe 未安装或版本不兼容。请运行: pip install mediapipe --upgrade")

except AttributeError:
    # MediaPipe 版本太旧
    raise ImportError("MediaPipe 版本太旧。请运行: pip install mediapipe --upgrade")

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
        self.hands = hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence
        )

    def process(self, frame: np.ndarray) -> HandState:
        """
        处理一帧图像，返回手部状态。
        - frame: BGR 格式图像
        - 返回: HandState，包含食指指尖坐标和检测状态
        """
        # 转换为 RGB 格式（MediaPipe 要求）
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 处理图像
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            # 取第一只手
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # 提取食指指尖（索引为8）
            index_finger_tip = hand_landmarks.landmark[8]
            tip_coords = (index_finger_tip.x, index_finger_tip.y)
            
            # 提取所有关键点
            all_landmarks = []
            for landmark in hand_landmarks.landmark:
                all_landmarks.append((landmark.x, landmark.y, landmark.z))
            
            return HandState(
                index_finger_tip=tip_coords,
                is_detected=True,
                all_landmarks=all_landmarks
            )
        else:
            # 未检测到手
            return HandState(
                index_finger_tip=(0, 0),
                is_detected=False,
                all_landmarks=[]
            )

    def release(self) -> None:
        """释放 MediaPipe 资源。"""
        self.hands.close()


