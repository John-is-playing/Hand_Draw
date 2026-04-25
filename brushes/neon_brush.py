import pygame
import math
from .base import BaseBrush

class NeonBrush(BaseBrush):
    """霓虹光带效果 — 手指轨迹产生发光拖尾"""

    def __init__(self, canvas: pygame.Surface):
        super().__init__(canvas)
        self.trail: list[tuple[int, int]] = []  # 轨迹点列表
        self.max_trail_length: int = 50          # 最大轨迹长度
        self.color_hue: float = 0.0              # 色相值，随时间变化

    def _render_stroke(self, from_pos, to_pos):
        """
        实现要点：
        1. 将新点加入 trail 列表
        2. 如果 trail 超过 max_trail_length，移除最旧的点
        3. 绘制效果：
           a. 先用较粗的半透明线条画外发光层（模拟 glow）
           b. 再用细的亮色线条画核心
           c. 颜色色相随时间缓慢变化（HSL 色相循环）
        4. 轨迹越旧的点越透明（alpha 递减）
        """
        # 添加新点到轨迹
        self.trail.append(to_pos)
        
        # 限制轨迹长度
        if len(self.trail) > self.max_trail_length:
            self.trail.pop(0)
        
        # 更新色相
        self.color_hue = (self.color_hue + 1.0) % 360
        
        # 绘制轨迹
        for i in range(len(self.trail) - 1):
            # 计算透明度（越旧的点越透明）
            alpha = int(255 * (i + 1) / len(self.trail))
            
            # 计算当前点的颜色
            color = pygame.Color(0)
            color.hsva = (self.color_hue, 100, 100, 100)  # 基础颜色不透明（A 值范围 0-100）
            
            # 绘制外发光层
            outer_surface = pygame.Surface(self.canvas.get_size(), pygame.SRCALPHA)
            # 外发光层使用较低的透明度
            glow_color = color.copy()
            glow_color.a = 100
            pygame.draw.line(outer_surface, glow_color, self.trail[i], self.trail[i+1], 12)
            self.canvas.blit(outer_surface, (0, 0))
            
            # 绘制核心线条
            core_color = color.copy()
            core_color.a = alpha
            pygame.draw.line(self.canvas, core_color, self.trail[i], self.trail[i+1], 4)
