import sys

sys.path.append('/tmp/pycharm_project_240/my/')

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from TSA transformer import AudioVisualSpikformer

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
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device))
    model.eval()
    print("✅ 模型加载成功")
    return model


# ===================== 🔥 清晰聚焦热力图（全自动，无报错） =====================
def get_clean_attention(model, mel, frame):
    features = None

    def hook(module, inp, out):
        nonlocal features
        if isinstance(out, (list, tuple)):
            out = out[0]
        features = out

    # 自动抓取视觉卷积层（最稳定、效果最好）
    target_layer = None
    for name, module in model.named_modules():
        if 'visual' in name and hasattr(module, 'weight') and len(module.weight.shape) == 4:
            target_layer = module
            print(f"✅ 已锁定最佳视觉层: {name}")
            break

    target_layer.register_forward_hook(hook)

    with torch.no_grad():
        model([mel, frame])

    # 🔥 核心：生成清晰、聚焦、有重点的热力图
    feat = features.float().cpu().numpy()
    feat = np.mean(feat, axis=(0, 1))  # 时序+通道平均

    # 增强对比度：只高亮模型真正关注的区域
    feat = np.maximum(feat, 0)
    feat = (feat - np.percentile(feat, 60)) / (np.percentile(feat, 99) - np.percentile(feat, 60) + 1e-8)
    feat = np.clip(feat, 0, 1)

    return feat


# ===================== 绘图 + 保存 =====================
# ===================== 绘图 + 保存 =====================
def show_heatmap(frame_path):
    mel, frame = prepare_input(frame_path)
    model = build_model()
    heatmap = get_clean_attention(model, mel, frame)

    # 原图
    img = cv2.imread(frame_path)
    ori_h, ori_w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 热力图处理（清晰、不糊、聚焦强）
    heatmap = cv2.resize(heatmap, (ori_w, ori_h))
    heatmap = cv2.GaussianBlur(heatmap, (15, 15), 3)  # 柔和但聚焦
    heatmap_8u = np.uint8(255 * heatmap)

    # ==============================================
    # 🔥 只改这一行：解决偏红！换成不发红的专业配色
    heatmap_color = cv2.applyColorMap(255 - heatmap_8u, cv2.COLORMAP_JET)    # ==============================================

    # 叠加显示
    result = cv2.addWeighted(img_rgb, 0.6, heatmap_color, 0.5, 0)

    # 保存
    cv2.imwrite(SAVE_HEATMAP_PATH, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    print(f"✅ 清晰热力图已保存：{SAVE_HEATMAP_PATH}")

    # 展示
    plt.figure(figsize=(12, 4))
    plt.subplot(131);
    plt.imshow(img_rgb);
    plt.axis('off');
    plt.title('Original')
    plt.subplot(132);
    plt.imshow(heatmap, cmap='jet');
    plt.axis('off');
    plt.title('Attention Heatmap')
    plt.subplot(133);
    plt.imshow(result);
    plt.axis('off');
    plt.title('Result')
    plt.tight_layout()
    plt.show()

# ===================== 运行 =====================
if __name__ == "__main__":
    show_heatmap(IMAGE_FRAME)