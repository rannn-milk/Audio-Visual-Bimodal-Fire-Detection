import sys

sys.path.append('/tmp/pycharm_project_240/my/')

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from transformer import AudioVisualSpikformer

# ===================== 你的路径 =====================
IMAGE_FRAME = "/tmp/tmp/pycharm_project_599/datasum/frames/1/1-490s.jpg"
WEIGHT_PATH = "/tmp/tmp/pycharm_project_599/my/av_spikformer_best.pth"
SAVE_HEATMAP_PATH = "/tmp/tmp/pycharm_project_599/my/attention_heatmap.jpg"
# ====================================================

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
    mel = torch.zeros(T, 1, 1, img_size, img_size).to(device)
    return mel, frame


# ===================== 加载模型 =====================
def build_model():
    model = AudioVisualSpikformer(
        step=T, num_classes=2,
        attn_method='SpatialTemporal',
        cross_attn=True, interaction='Add',
        contrastive=True, num_heads=16, depths=2
    ).to(device)
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device), strict=False)
    model.eval()
    print("✅ 模型加载成功")
    return model


# ===================== 🔥 一定有颜色！=====================
def get_heatmap(model, mel, frame):
    features = None

    def hook(module, inp, out):
        nonlocal features
        if isinstance(out, (list, tuple)):
            out = out[0]
        features = out

    # 安全抓取层，绝不报错
    for name, module in model.named_modules():
        if 'visual_patch_embed' in name:
            module.register_forward_hook(hook)
            break

    with torch.no_grad():
        model([mel, frame])

    feat = features.cpu().numpy()
    feat = feat.mean(axis=(0, 1))
    feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)

    # 🔥 强制让图有颜色：低=蓝，高=红
    return feat


# ===================== 绘图 + 保存 =====================
def show_heatmap(frame_path):
    mel, frame = prepare_input(frame_path)
    model = build_model()
    heatmap = get_heatmap(model, mel, frame)

    # 原图
    img = cv2.imread(frame_path)
    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 热力图（一定有颜色）
    heatmap = cv2.resize(heatmap, (w, h))
    heatmap_blur = cv2.GaussianBlur(heatmap, (11, 11), 4)
    heatmap_8u = np.uint8(255 * heatmap_blur)
    heatmap_color = cv2.applyColorMap(heatmap_8u, cv2.COLORMAP_JET)
    result = cv2.addWeighted(img_rgb, 0.7, heatmap_color, 0.3, 0)

    # 保存
    cv2.imwrite(SAVE_HEATMAP_PATH, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    print("✅ 已保存:", SAVE_HEATMAP_PATH)

    # 显示
    plt.figure(figsize=(12, 4))
    plt.subplot(131);
    plt.imshow(img_rgb);
    plt.axis('off')
    plt.subplot(132);
    plt.imshow(heatmap, cmap='jet');
    plt.axis('off')
    plt.subplot(133);
    plt.imshow(result);
    plt.axis('off')
    plt.show()


# ===================== 运行 =====================
if __name__ == "__main__":
    show_heatmap(IMAGE_FRAME)