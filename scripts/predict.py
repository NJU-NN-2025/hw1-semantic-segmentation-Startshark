import cv2
import numpy as np
from ultralytics import YOLO
import argparse
from pathlib import Path

def predict_custom(model_path, image_path, output_path=None, conf=0.25):
    """
    专门用于在彩色图片上测试在灰度数据集上训练的模型。
    1. 读取彩色图片
    2. 转换为灰度图（模拟训练数据分布）
    3. 进行推理
    4. 将推理结果（Mask）叠加回原彩色图片上
    """
    
    # 1. 读取原始图片
    original_img = cv2.imread(image_path)
    if original_img is None:
        print(f"错误: 无法读取图片 {image_path}")
        return

    print(f"正在处理图片: {image_path}")
    print(f"原始尺寸: {original_img.shape}")

    # 2. 预处理：转换为灰度图 (模拟训练数据)
    # 训练数据是 312x312 的灰度图 (3通道相同)
    # 我们先转为灰度
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    # 再转回BGR (3通道，但每个通道值相同)，因为YOLO通常期望3通道输入
    input_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    print("已转换为灰度图进行推理 (模拟训练数据分布)...")

    # 3. 加载模型并推理
    model = YOLO(model_path)
    results = model.predict(source=input_img, conf=conf, verbose=False)
    result = results[0]

    # 4. 处理结果
    if result.masks is None:
        print("未检测到任何对象。")
        # 即使没有检测到，也保存/显示原图
        final_img = original_img
    else:
        print(f"检测到 {len(result.masks)} 个对象！")
        
        # 获取掩膜 (Masks)
        # result.masks.data 是 (N, H, W) 的 tensor
        # 我们需要将其调整回原图大小
        masks = result.masks.data.cpu().numpy()
        
        # 创建一个彩色遮罩层
        mask_overlay = np.zeros_like(original_img)
        
        # 为每个检测到的对象分配随机颜色
        for i, mask in enumerate(masks):
            # 调整 mask 大小以匹配原图
            # 注意: result.masks.data 的尺寸可能与模型输入尺寸一致 (312x312)，需要resize回原图
            # 但 ultralytics 的 result.plot() 会自动处理，我们这里手动处理是为了叠加到原图
            
            # 更简单的方法：使用 result.plot() 生成带掩膜的图，但那是基于灰度输入的
            # 我们想要的是：原图(彩色) + 掩膜
            
            # 获取多边形坐标 (相对于原图尺寸)
            # result.masks.xy 是一个列表，包含每个mask的轮廓坐标
            contours = result.masks.xy[i].astype(np.int32)
            
            # 随机颜色
            color = np.random.randint(0, 255, (3,), dtype=np.uint8).tolist()
            
            # 填充轮廓
            cv2.fillPoly(mask_overlay, [contours], color)

        # 混合原图和掩膜
        alpha = 0.5
        final_img = cv2.addWeighted(original_img, 1, mask_overlay, alpha, 0)
        
        # 绘制边界框和标签
        if result.boxes is not None:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                label = f"{result.names[cls]} {conf:.2f}"
                
                cv2.rectangle(final_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(final_img, label, (x1, y1 - 10), cv2.LINE_AA, 0.5, (0, 255, 0), 2)

    # 5. 保存或显示
    if output_path:
        cv2.imwrite(output_path, final_img)
        print(f"结果已保存至: {output_path}")
    else:
        # 如果没有指定输出路径，尝试显示窗口
        # 注意：如果图片太大，缩放一下显示
        display_img = final_img.copy()
        h, w = display_img.shape[:2]
        if h > 1000 or w > 1000:
            scale = 1000 / max(h, w)
            display_img = cv2.resize(display_img, (0, 0), fx=scale, fy=scale)
        
        cv2.imshow("Result (Press any key to exit)", display_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="在彩色图片上测试灰度训练的模型")
    parser.add_argument("--source", type=str, required=True, help="输入图片路径")
    parser.add_argument("--model", type=str, default="runs/segment/antarctic_yolo/weights/best.pt", help="模型路径")
    parser.add_argument("--output", type=str, default=None, help="输出图片路径 (可选)")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    
    args = parser.parse_args()
    
    predict_custom(args.model, args.source, args.output, args.conf)
