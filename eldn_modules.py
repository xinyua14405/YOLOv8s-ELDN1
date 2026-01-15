import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv


# 1. Efficient Multi-Scale Convolution (EMSC) [cite: 275-283]
class EMSC(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        # 按照 50%, 25%, 25% 的比例分割通道 [cite: 281]
        self.c1 = c1 // 2
        self.c2 = c1 // 4
        self.c3 = c1 - self.c1 - self.c2

        self.cv3x3 = Conv(self.c2, self.c2, 3)  # 捕捉高频局部细节 [cite: 279, 344]
        self.cv5x5 = Conv(self.c3, self.c3, 5)  # 捕捉低频全局上下文 [cite: 280, 345]
        self.cv_fuse = Conv(c1, c2, 1)  # 1x1 融合 [cite: 283]

    def forward(self, x):
        x1, x2, x3 = torch.split(x, [self.c1, self.c2, self.c3], dim=1)
        return self.cv_fuse(torch.cat((x1, self.cv3x3(x2), self.cv5x5(x3)), dim=1))


# 2. C2f-EMSC 模块 [cite: 65, 289-291]
class C2f_EMSC(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv0 = EMSC(c1, c2)
        self.cv1 = Conv(c2, c2, 1, 1)
        self.m = nn.ModuleList(nn.Sequential(Conv(self.c, self.c, 3), EMSC(self.c, self.c)) for _ in range(n))

    def forward(self, x):
        y = list(self.cv0(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv1(torch.cat(y, 1))


# 3. LSKA 注意力机制 [cite: 67, 356-358]
class LSKA(nn.Module):
    def __init__(self, c, k=5, d=1):
        super().__init__()
        # 大核分离卷积实现 [cite: 356]
        self.conv0 = nn.Conv2d(c, c, (1, 2 * d - 1), padding=(0, d - 1), groups=c)
        self.conv1 = nn.Conv2d(c, c, (2 * d - 1, 1), padding=(d - 1, 0), groups=c)
        self.conv2 = nn.Conv2d(c, c, (1, k // d), padding=(0, (k // d) // 2), groups=c, dilation=d)
        self.conv3 = nn.Conv2d(c, c, (k // d, 1), padding=((k // d) // 2, 0), groups=c, dilation=d)
        self.conv4 = nn.Conv2d(c, c, 1)

    def forward(self, x):
        attn = self.conv4(self.conv3(self.conv2(self.conv1(self.conv0(x)))))
        return x * attn


# 4. SPPF-LSKA 模块 [cite: 67, 357]
class SPPF_LSKA(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.lska = LSKA(c_ * 4)  # 在池化融合后加入 LSKA [cite: 358]

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(self.lska(torch.cat((x, y1, y2, self.m(y2)), 1)))


# 5. DySample 动态上采样 [cite: 68, 404-408]
class DySample(nn.Module):
    def __init__(self, in_channels, scale=2):
        super().__init__()
        self.scale = scale
        self.offset_conv = nn.Conv2d(in_channels, 2 * scale * scale, 1)

    def forward(self, x):
        # 简单实现版，实际推理时通过偏移量调整采样点位置 [cite: 406]
        return nn.functional.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)