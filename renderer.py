import pygame
import cv2

class Renderer:
    def __init__(self, width: int = 1280, height: int = 720):
        """
        初始化 Pygame 窗口和画布。
        - width/height: 窗口尺寸
        """
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("手势魔法画板 ✨ Gesture Magic Canvas")
        
        # 创建画布 Surface
        self.canvas = pygame.Surface((width, height))
        self.clear_canvas()
        
        # 字体初始化
        self.font = pygame.font.Font(None, 36)

    def clear_canvas(self) -> None:
        """清空画布（填充黑色背景）。"""
        self.canvas.fill((0, 0, 0))

    def get_canvas(self) -> pygame.Surface:
        """获取画布 Surface 对象。"""
        return self.canvas

    def draw_ui(self, brush_name: str, fps: float) -> None:
        """
        绘制 UI 叠加层。
        - 左上角: 当前画笔名称
        - 右上角: FPS 数值
        - 底部居中: 操作提示 "按 1/2/3 切换画笔 | ESC 退出"
        """
        # 绘制画笔名称
        brush_text = self.font.render(f"画笔: {brush_name}", True, (255, 255, 255))
        self.screen.blit(brush_text, (20, 20))
        
        # 绘制 FPS
        fps_text = self.font.render(f"FPS: {fps:.1f}", True, (255, 255, 255))
        fps_rect = fps_text.get_rect()
        fps_rect.topright = (self.width - 20, 20)
        self.screen.blit(fps_text, fps_rect)
        
        # 绘制操作提示
        hint_text = self.font.render("按 1/2/3 切换画笔 | ESC 退出", True, (255, 255, 255))
        hint_rect = hint_text.get_rect()
        hint_rect.center = (self.width // 2, self.height - 30)
        self.screen.blit(hint_text, hint_rect)

    def flip(self, camera_frame: None = None) -> None:
        """刷新显示。
        - camera_frame: 摄像头帧（可选），如果提供则显示在左上角
        """
        # 将画布内容绘制到屏幕
        self.screen.blit(self.canvas, (0, 0))
        
        # 如果提供了摄像头帧，显示在左上角
        if camera_frame is not None:
            # 将 OpenCV 帧转换为 Pygame 表面
            # 转换颜色空间从 BGR 到 RGB
            frame_rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
            # 调整大小为小窗口
            frame_resized = cv2.resize(frame_rgb, (320, 240))
            # 创建 Pygame 表面
            frame_surface = pygame.surfarray.make_surface(frame_resized.swapaxes(0, 1))
            # 绘制到屏幕左上角
            self.screen.blit(frame_surface, (10, 10))
        
        # 刷新显示
        pygame.display.flip()

    def should_quit(self) -> bool:
        """检查是否收到退出事件。"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return True
        return False

    def get_key(self) -> int | None:
        """获取当前按下的键。"""
        # 注意：这里需要重新获取事件，因为 should_quit() 已经处理了一部分事件
        # 为了避免事件丢失，我们需要在主循环中统一处理事件
        # 这里只作为辅助方法
        keys = pygame.key.get_pressed()
        for key in range(pygame.K_0, pygame.K_9 + 1):
            if keys[key]:
                return key
        return None
