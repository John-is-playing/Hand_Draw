import pygame
import random
import math
from .base import BaseBrush

class FireParticle:
    """单个火焰粒子"""
    def __init__(self, x: float, y: float):
        self.x: float = x
        self.y: float = y
        # 随机向上偏移 + 轻微水平扩散
        self.vx: float = random.uniform(-1.0, 1.0)
        self.vy: float = random.uniform(-3.0, -1.0)
        self.life: float = 1.0  # 剩余生命值 0~1
        self.size: float = random.uniform(3, 8)  # 粒子大小

class FireBrush(BaseBrush):
    """火焰粒子效果 — 手指轨迹伴随火焰粒子"""

    def __init__(self, canvas: pygame.Surface):
        super().__init__(canvas)
        self.particles: list[FireParticle] = []
        self.max_particles: int = 500

    def _render_stroke(self, from_pos, to_pos):
        """
        实现要点：
        1. 在 from_pos 到 to_pos 的路径上，每隔一定距离生成若干粒子
        2. 每个粒子的初始属性：
           - 位置: 路径上的随机点
           - 速度: 随机向上偏移 + 轻微水平扩散
           - 生命: 1.0，每帧递减
           - 大小: 随机 3~8 像素
        3. 每帧更新所有粒子：
           - 位置 += 速度
           - vy 略微增加（模拟火焰上升减速）
           - life -= decay_rate
           - 移除 life <= 0 的粒子
        4. 渲染每个粒子：
           - 颜色映射: life 高→亮黄/白，life 中→橙色，life 低→暗红
           - 大小随 life 缩小
           - 使用圆形绘制，带 alpha 混合
        """
        # 计算两点之间的距离
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        distance = math.sqrt(dx*dx + dy*dy)
        
        # 根据距离生成粒子
        num_particles = max(1, int(distance / 10))
        for _ in range(num_particles):
            if len(self.particles) < self.max_particles:
                # 在路径上随机生成点
                t = random.random()
                x = from_pos[0] + t * dx
                y = from_pos[1] + t * dy
                self.particles.append(FireParticle(x, y))

    def update(self, finger_pos, is_detected):
        """重写 update，增加粒子物理更新逻辑。"""
        super().update(finger_pos, is_detected)
        self._update_particles()  # 即使不绘制，已有粒子也要继续运动和消散

    def _update_particles(self) -> None:
        """更新所有粒子的物理状态和渲染。"""
        # 创建一个新的表面用于绘制粒子
        particle_surface = pygame.Surface(self.canvas.get_size(), pygame.SRCALPHA)
        
        # 更新和渲染粒子
        i = 0
        while i < len(self.particles):
            particle = self.particles[i]
            
            # 更新位置
            particle.x += particle.vx
            particle.y += particle.vy
            
            # 模拟重力（火焰上升减速）
            particle.vy += 0.1
            
            # 减少生命值
            particle.life -= 0.02
            
            if particle.life <= 0:
                # 移除死亡粒子
                self.particles.pop(i)
                continue
            
            # 根据生命值计算颜色
            if particle.life > 0.7:
                # 亮黄/白
                color = pygame.Color(255, 255, 100)
            elif particle.life > 0.4:
                # 橙色
                color = pygame.Color(255, 165, 0)
            else:
                # 暗红
                color = pygame.Color(139, 0, 0)
            
            # 计算透明度和大小
            alpha = int(particle.life * 255)
            size = particle.size * particle.life
            
            # 绘制粒子
            if size > 0:
                particle_color = color.copy()
                particle_color.a = alpha
                pygame.draw.circle(particle_surface, particle_color, 
                                 (int(particle.x), int(particle.y)), int(size))
            
            i += 1
        
        # 将粒子绘制到画布
        self.canvas.blit(particle_surface, (0, 0))
