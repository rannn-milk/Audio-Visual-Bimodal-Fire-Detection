import torch
import torch.nn as nn

# 独立的视觉/音频分支 残差块
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = nn.ReLU(inplace=True)(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.ReLU(inplace=True)(out)
        return out

# 双分支 ResNet18 —— 视觉 + 音频 完全独立
class ResNet18(nn.Module):
    def __init__(self, num_classes=2, fusion_mode="OGM-GE"):
        super(ResNet18, self).__init__()
        self.fusion_mode = fusion_mode

        # ==================== 视觉分支 ====================
        self.vis_conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.vis_bn = nn.BatchNorm2d(64)
        self.vis_layer1 = self._make_layer(64, 64, 2, stride=1)
        self.vis_layer2 = self._make_layer(64, 128, 2, stride=2)
        self.vis_layer3 = self._make_layer(128, 256, 2, stride=2)
        self.vis_layer4 = self._make_layer(256, 512, 2, stride=2)

        # ==================== 音频分支 ====================
        self.aud_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.aud_bn = nn.BatchNorm2d(64)
        self.aud_layer1 = self._make_layer(64, 64, 2, stride=1)
        self.aud_layer2 = self._make_layer(64, 128, 2, stride=2)
        self.aud_layer3 = self._make_layer(128, 256, 2, stride=2)
        self.aud_layer4 = self._make_layer(256, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_planes, planes, blocks, stride):
        layers = [BasicBlock(in_planes, planes, stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(planes, planes, 1))
        return nn.Sequential(*layers)

    def forward(self, vis, aud):
        # 视觉
        x = nn.ReLU()(self.vis_bn(self.vis_conv(vis)))
        x = self.vis_layer1(x)
        x = self.vis_layer2(x)
        x = self.vis_layer3(x)
        x = self.vis_layer4(x)
        x = self.avgpool(x).flatten(1)

        # 音频
        a = nn.ReLU()(self.aud_bn(self.aud_conv(aud)))
        a = self.aud_layer1(a)
        a = self.aud_layer2(a)
        a = self.aud_layer3(a)
        a = self.aud_layer4(a)
        a = self.avgpool(a).flatten(1)

        # 4种融合方式
        if self.fusion_mode == "OGM-GE":
            f = x * torch.sigmoid(a) + a * torch.sigmoid(x)
        elif self.fusion_mode == "MSLR":
            f = torch.max(x, a)
        elif self.fusion_mode == "PMR":
            f = x + a
        elif self.fusion_mode == "AGM":
            f = (x + a) / 2
        else:
            f = x + a

        return self.fc(f)

# 对外接口
def resnet18(num_classes=2, fusion_mode="OGM-GE"):
    return ResNet18(num_classes=num_classes, fusion_mode=fusion_mode)