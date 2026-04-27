import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models, transforms
from torchvision.models.resnet import ResNet
import tqdm  # 用于显示进度条

# -------------------------- 1. 配置路径（只改这里！）--------------------------
VIDEO_PATH    = "/tmp/tmp/pycharm_project_599/my/cut_done.mp4"    # 输入视频
OUTPUT_VIDEO  = "/tmp/tmp/pycharm_project_599/my/heatmap_video.mp4"   # 输出热力图视频

# -------------------------- 2. 加载模型 --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True).to(device)
model.eval()

# -------------------------- 3. 图像预处理 --------------------------
def preprocess_frame(frame):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(frame).unsqueeze(0).to(device)
    return img_tensor

# -------------------------- 4. Grad-CAM 核心（和你原来一样） --------------------------
class GradCAM:
    def __init__(self, model: ResNet, target_layer="layer4"):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]
        def forward_hook(module, input, output):
            self.activations = output

        target_module = dict(self.model.named_modules())[self.target_layer]
        target_module.register_forward_hook(forward_hook)
        target_module.register_backward_hook(backward_hook)

    def generate(self, input_tensor):
        output = self.model(input_tensor)
        pred_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, pred_class].backward()

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = cam.squeeze().cpu().detach().numpy()
        return cam

# -------------------------- 5. 视频热力图处理 --------------------------
def process_video_heatmap(video_path, output_path):
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    grad_cam = GradCAM(model)
    print(f"🎬 视频处理中，总帧数：{total_frames}，FPS：{fps}")

    # 逐帧处理
    for _ in tqdm.tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        # 预处理 + 生成CAM
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = preprocess_frame(frame_rgb)
        cam = grad_cam.generate(img_tensor)

        # 热力图叠加
        heatmap = cv2.resize(cam, (width, height))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        combined = cv2.addWeighted(frame, 0.6, heatmap_colored, 0.4, 0)

        out.write(combined)

    cap.release()
    out.release()
    print(f"✅ 热力图视频已保存：{output_path}")

# -------------------------- 运行 --------------------------
if __name__ == "__main__":
    process_video_heatmap(VIDEO_PATH, OUTPUT_VIDEO)