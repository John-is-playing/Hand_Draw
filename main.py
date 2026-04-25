import time
import pygame
from camera import Camera
from hand_tracker import HandTracker
from renderer import Renderer
from brushes import NeonBrush, FireBrush, RainbowBrush

def main():
    """
    主函数 — 程序入口。

    初始化流程：
    1. 创建 Camera 实例
    2. 创建 HandTracker 实例
    3. 创建 Renderer 实例
    4. 创建三种画笔实例
    5. 设置默认画笔为 NeonBrush

    主循环：
    while not renderer.should_quit():
        1. 读取摄像头帧
        2. 将帧传入 HandTracker.process() 获取 HandState
        3. 将手指归一化坐标映射到画布像素坐标
           - 注意：摄像头画面是镜像的，x 坐标需要翻转 (1 - x)
           - 映射公式: canvas_x = int((1 - norm_x) * canvas_width)
                        canvas_y = int(norm_y * canvas_height)
        4. 将像素坐标传入当前画笔的 update() 方法
        5. 处理键盘事件：
           - 按 '1': 切换到 NeonBrush
           - 按 '2': 切换到 FireBrush
           - 按 '3': 切换到 RainbowBrush
        6. 调用 renderer.draw_ui() 绘制 UI
        7. 调用 renderer.flip() 刷新显示
        8. 计算 FPS

    清理流程：
    1. camera.release()
    2. hand_tracker.release()
    3. pygame.quit()
    """
    # 初始化模块
    camera = Camera()
    hand_tracker = HandTracker()
    renderer = Renderer()
    
    # 创建画笔实例
    canvas = renderer.get_canvas()
    neon_brush = NeonBrush(canvas)
    fire_brush = FireBrush(canvas)
    rainbow_brush = RainbowBrush(canvas)
    
    # 设置默认画笔
    current_brush = neon_brush
    brush_name = "霓虹光带"
    
    # FPS 计算
    fps = 0
    frame_count = 0
    start_time = time.time()
    
    # 主循环
    while not renderer.should_quit():
        # 读取摄像头帧
        frame = camera.read_frame()
        if frame is None:
            continue
        
        # 处理手势识别
        hand_state, processed_frame = hand_tracker.process(frame)
        
        # 映射坐标
        if hand_state.is_detected:
            norm_x, norm_y = hand_state.index_finger_tip
            # 注意：摄像头画面是镜像的，x 坐标需要翻转 (1 - x)
            canvas_x = int((1 - norm_x) * renderer.width)
            canvas_y = int(norm_y * renderer.height)
            finger_pos = (canvas_x, canvas_y)
        else:
            finger_pos = (0, 0)
        
        # 更新画笔
        current_brush.update(finger_pos, hand_state.is_detected)
        
        # 处理键盘事件
        keys = pygame.key.get_pressed()
        if keys[pygame.K_1]:
            current_brush = neon_brush
            brush_name = "霓虹光带"
        elif keys[pygame.K_2]:
            current_brush = fire_brush
            brush_name = "火焰粒子"
        elif keys[pygame.K_3]:
            current_brush = rainbow_brush
            brush_name = "彩虹画笔"
        
        # 计算 FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1.0:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
        
        # 绘制 UI
        renderer.draw_ui(brush_name, fps)
        
        # 刷新显示（传入处理后的摄像头帧以显示画中画）
        renderer.flip(camera_frame=processed_frame)
    
    # 清理流程
    camera.release()
    hand_tracker.release()
    pygame.quit()

if __name__ == "__main__":
    main()
