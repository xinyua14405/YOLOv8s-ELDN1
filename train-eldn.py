import torch
from ultralytics import YOLO
from eldn_modules import C2f_EMSC, SPPF_LSKA, DySample  # 导入自定义模块


# NWD 损失函数实现 [cite: 447, 455]
def nwd_loss(pred_boxes, gt_boxes, constant=12.0):
    # 将 Bbox 建模为二维高斯分布并计算 Wasserstein 距离 [cite: 432, 452]
    b1_xy = pred_boxes[..., :2]
    b1_wh_half = pred_boxes[..., 2:] / 2
    b2_xy = gt_boxes[..., :2]
    b2_wh_half = gt_boxes[..., 2:] / 2

    # 公式 (10): Wasserstein 距离计算 [cite: 452]
    w2_dist = torch.norm(b1_xy - b2_xy, dim=-1) ** 2 + torch.norm(b1_wh_half - b2_wh_half, dim=-1) ** 2
    # 公式 (11): NWD 归一化 [cite: 455]
    return torch.exp(-torch.sqrt(w2_dist) / constant)


# 注册模块到 YOLO 映射字典（确保 YAML 能够识别）
# 注意：在 Ultralytics 内部代码中通常需要手动修改 tasks.py 注册，这里是脚本化调用的逻辑说明

def main():
    # 1. 加载模型结构
    model = YOLO("yolov8s-eldn.yaml")

    # 2. 设置训练超参数
    model.train(
        data="your_dataset.yaml",  # 需要你自己准备数据集路径文件
        epochs=500,  # [cite: 474]
        imgsz=512,  # [cite: 473]
        batch=12,  # [cite: 473]
        lr0=0.002,  # 初始学习率 [cite: 474]
        lrf=0.01,  # 最终学习率倍率 (0.00002 / 0.002) [cite: 474]
        momentum=0.937,  # [cite: 474]
        weight_decay=0.0005,  # [cite: 474]
        optimizer='SGD',  # [cite: 474]
        device=0,  # 指定 GTX 1650 [cite: 464]
        close_mosaic=10,  # 最后 10 个 epoch 关闭 Mosaic [cite: 474]
        # 在 Ultralytics 损失中启用 NWD 的方法通常需要修改 loss.py
        # 如果不修改库文件，默认会使用 CIoU，建议在 Ultralytics 源码中添加 nwd_loss 调用
    )


if __name__ == '__main__':
    main()