import pygame
from .base import BaseBrush

class RainbowBrush(BaseBrush):
    """彩虹画笔效果 — 手指轨迹产生彩色渐变线条"""

    def __init__(self, canvas: pygame.Surface):
        super().__init__(canvas)
        self.hue: float = 0.0           # 当前色相
        self.hue_speed: float = 2.0     # 色相变化速度（度/帧）
        self.line_width: int = 6        # 线条宽度

    def _render_stroke(self, from_pos, to_pos):
        """
        实现要点：
        1. 使用 pygame.draw.line 在 from_pos 和 to_pos 之间画线
        2. 颜色通过 HSV 色相循环生成：
           - hue 从 0~360 循环
           - 每帧递增 hue_speed
           - S=100%, V=100%（纯色）
        3. 线条宽度固定为 line_width
        4. 可选：在线条两侧加一层更粗的半透明同色线条，增加柔和感
        """
        # 更新色相
        self.hue = (self.hue + self.hue_speed) % 360
        
        # 将 HSV 转换为 RGB
        color = pygame.Color(0)
        color.hsva = (self.hue, 100, 100, 100)
        
        # 绘制外层半透明线条，增加柔和感
        outer_surface = pygame.Surface(self.canvas.get_size(), pygame.SRCALPHA)
        # 创建半透明版本的颜色
        outer_color = color.copy()
        outer_color.a = 100
        pygame.draw.line(outer_surface, outer_color, from_pos, to_pos, self.line_width + 4)
        self.canvas.blit(outer_surface, (0, 0))
        
        # 绘制内层线条
        pygame.draw.line(self.canvas, color, from_pos, to_pos, self.line_width)
