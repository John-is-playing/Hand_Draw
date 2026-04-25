from abc import ABC, abstractmethod
import pygame

class BaseBrush(ABC):
    """所有画笔效果的抽象基类"""

    def __init__(self, canvas: pygame.Surface):
        """
        - canvas: Pygame 画布 Surface 对象
        """
        self.canvas = canvas
        self.last_pos: tuple[int, int] | None = None  # 上一帧手指位置
        self.is_drawing: bool = False  # 是否正在绘制

    def update(self, finger_pos: tuple[int, int], is_detected: bool) -> None:
        """
        每帧调用，更新画笔状态并渲染。
        - finger_pos: 手指在画布上的像素坐标 (x, y)
        - is_detected: 是否检测到手
        """
        if not is_detected:
            self.is_drawing = False
            self.last_pos = None
            return

        if not self.is_drawing:
            self.is_drawing = True
            self.last_pos = finger_pos
            return

        # 计算运动速度（两帧之间的距离）
        self._render_stroke(self.last_pos, finger_pos)
        self.last_pos = finger_pos

    @abstractmethod
    def _render_stroke(self, from_pos: tuple[int, int], to_pos: tuple[int, int]) -> None:
        """在两个位置之间渲染一段笔画效果。子类必须实现。"""
        pass

    def reset(self) -> None:
        """重置画笔状态（切换画笔时调用）。"""
        self.last_pos = None
        self.is_drawing = False
