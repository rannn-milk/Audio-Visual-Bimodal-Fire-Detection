import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models, transforms
from torchvision.models.resnet import ResNet

# -------------------------- 1. 配置路径（你只需要改这里）--------------------------
# 你的图片绝对路径
IMAGE_PATH = "/tmp/tmp/pycharm_project_599/datasum/frames/1/1-2-595s.jpg"  # 替换成真实图片名
# 热力图保存路径
SAVE_HEATMAP_PATH = "/tmp/tmp/pycharm_project_599/my/attention_heatmap.jpg"

# -------------------------- 2. 加载预训练模型（ResNet50，通用视觉模型）--------------------------
# 使用GPU（如果有），没有就用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 加载预训练ResNet50（适用于各类图片注意力可视化）
model = models.resnet50(pretrained=True).to(device)
model.eval()  # 推理模式


# -------------------------- 3. 图片预处理 --------------------------
def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    return img_tensor, img


# -------------------------- 4. Grad-CAM 注意力热力图核心代码 --------------------------
class GradCAM:
    def __init__(self, model: ResNet, target_layer: str = "layer4"):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        # 注册梯度和激活值钩子
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        def forward_hook(module, input, output):
            self.activations = output

        # 找到目标层并注册钩子
        target_module = dict(self.model.named_modules())[self.target_layer]
        target_module.register_forward_hook(forward_hook)
        target_module.register_backward_hook(backward_hook)

    def generate(self, input_tensor):
        # 前向传播
        output = self.model(input_tensor)
        pred_class = output.argmax(dim=1).item()

        # 反向传播获取梯度
        self.model.zero_grad()
        output[0, pred_class].backward()

        # 计算权重
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        # 加权求和得到热力图
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam)  # 只保留正贡献

        # 归一化到0-1
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = cam.squeeze().cpu().detach().numpy()

        return cam, pred_class


# -------------------------- 5. 生成并绘制热力图 --------------------------
def visualize_heatmap(img_path, save_path):
    # 预处理图片
    img_tensor, original_img = preprocess_image(img_path)
    original_img = np.array(original_img)

    # 初始化Grad-CAM
    grad_cam = GradCAM(model)
    # 生成热力图
    cam, pred_class = grad_cam.generate(img_tensor)

    # 调整热力图大小匹配原图
    heatmap = cv2.resize(cam, (original_img.shape[1], original_img.shape[0]))
    # 转换为彩色热力图（JET色映射）
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    # 热力图叠加原图
    superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap_colored, 0.4, 0)

    # 绘图展示
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(original_img)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap="jet")
    plt.title("Attention Heatmap")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.title("Image + Heatmap")
    plt.axis("off")

    plt.tight_layout()
    # 保存热力图
    cv2.imwrite(save_path, superimposed_img)
    plt.show()
    print(f"✅ 注意力热力图已保存至：{save_path}")


# -------------------------- 运行 --------------------------
if __name__ == "__main__":
    visualize_heatmap(IMAGE_PATH, SAVE_HEATMAP_PATH)