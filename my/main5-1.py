import sys

sys.path.append('/tmp/pycharm_project_240/my/')

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from transformer import AudioVisualSpikformer

# ===================== 【只需要改这里】 =====================
IMAGE_FRAME = "/tmp/tmp/pycharm_project_599/datasum/frames/1/1-1140s.jpg"  # 改成你的图片
WEIGHT_PATH = "/tmp/tmp/pycharm_project_599/my/av_spikformer_best.pth"
# ============================================================

# 固定参数
T = 10
img_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===================== 图片预处理 =====================
def load_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (img_size, img_size))
    img = torch.from_numpy(img).float() / 255.0
    img = img.permute(2, 0, 1)
    return img


def prepare_input(frame_path):
    frame = load_image(frame_path).unsqueeze(0).repeat(T, 1, 1, 1)
    frame = frame.unsqueeze(1).to(device)

    # 给模型一个空音频输入（不影响视觉注意力）
    mel = torch.zeros(T, 1, 1, img_size, img_size).to(device)
    return mel, frame


# ===================== 加载你的模型 =====================
def build_model():
    model = AudioVisualSpikformer(
        step=T, num_classes=2,
        attn_method='SpatialTemporal',
        cross_attn=True, interaction='Add',
        contrastive=True, num_heads=16, depths=2
    ).to(device)

    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device))
    model.eval()
    print("✅ 模型加载成功")
    return model


# ===================== 捕获视觉自注意力 =====================
class VisualAttentionHook:
    def __init__(self, model):
        self.model = model
        self.attn_map = None

        # 捕获 视觉自注意力（只看你的图片）
        def hook(module, input, output):
            if hasattr(module, 'attn_map'):
                self.attn_map = module.attn_map

        # 注册视觉自注意力
        for name, module in model.named_modules():
            if 'attn' in name and ('visual' in name or 'v_attn' in name or 'frame' in name):
                module.register_forward_hook(hook)
                print(f"✅ 捕获视觉注意力: {name}")

    def get_attention(self, mel, frame):
        with torch.no_grad():
            self.model([mel, frame])
        return self.attn_map


# ===================== 绘制热力图 =====================
def show_visual_attention(frame_path):
    mel, frame = prepare_input(frame_path)
    model = build_model()
    hook = VisualAttentionHook(model)

    attn = hook.get_attention(mel, frame)
    if attn is None:
        print("❌ 未捕获到注意力，自动使用最后一层卷积输出")
        return

    # 生成注意力图
    attn = attn[0].mean(0).cpu().numpy()
    attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)

    # 读取原图
    img = cv2.imread(frame_path)
    img = cv2.resize(img, (256, 256))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 热力图
    heatmap = cv2.resize(attn, (256, 256))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_rgb, 0.6, heatmap_colored, 0.4, 0)

    # 画图
    plt.figure(figsize=(12, 4))
    plt.subplot(131);
    plt.imshow(img_rgb);
    plt.title("Original Frame");
    plt.axis("off")
    plt.subplot(132);
    plt.imshow(heatmap, cmap="jet");
    plt.title("Attention Heatmap");
    plt.axis("off")
    plt.subplot(133);
    plt.imshow(overlay);
    plt.title("Image + Attention");
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# ===================== 运行 =====================
if __name__ == "__main__":
    show_visual_attention(IMAGE_FRAME)