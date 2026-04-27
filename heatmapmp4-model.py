import sys
sys.path.append('/tmp/pycharm_project_240/my/')

import torch
import cv2
import numpy as np
from tqdm import tqdm
from transformer import AudioVisualSpikformer
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 关闭matplotlib后端（避免服务器报错）
plt.switch_backend('agg')

from transformer import AudioVisualSpikformer
# ===================== 【只改这里】 =====================
VIDEO_PATH    = "/tmp/tmp/pycharm_project_599/my/cut_done.mp4"    # 输入视频
SAVE_VIDEO_PATH  = "/tmp/tmp/pycharm_project_599/my/heatmap_video1.mp4"   # 输出热力图视频
WEIGHT_PATH   = "/tmp/tmp/pycharm_project_599/my/av_spikformer_best.pth"
# ========================================================


T = 10
img_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fps = 25


# ===================== 预处理 =====================
def load_image(img):
    img = cv2.resize(img, (img_size, img_size))
    img = torch.from_numpy(img).float() / 255.0
    img = img.permute(2, 0, 1)
    return img


def prepare_input(frame):
    frame = load_image(frame).unsqueeze(0).repeat(T, 1, 1, 1)
    frame = frame.unsqueeze(1).to(device)
    mel = torch.zeros(T, 1, 1, img_size, img_size).to(device)
    return mel, frame


# ===================== 模型 =====================
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


# ===================== 热力图 =====================
def get_clean_attention(model, mel, frame):
    features = None

    def hook(module, inp, out):
        nonlocal features
        if isinstance(out, (list, tuple)):
            out = out[0]
        features = out

    target_layer = None
    for name, module in model.named_modules():
        if 'visual' in name and hasattr(module, 'weight') and len(module.weight.shape) == 4:
            target_layer = module
            break

    target_layer.register_forward_hook(hook)

    with torch.no_grad():
        model([mel, frame])

    feat = features.float().cpu().numpy()
    feat = np.mean(feat, axis=(0, 1))

    feat = np.maximum(feat, 0)
    feat = (feat - np.percentile(feat, 60)) / (np.percentile(feat, 99) - np.percentile(feat, 60) + 1e-8)
    feat = np.clip(feat, 0, 1)

    return feat


# ===================== 只生成视频 =====================
def process_video_heatmap(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("无法打开视频")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(SAVE_VIDEO_PATH, fourcc, fps, (width, height))

    model = build_model()
    pbar = tqdm(total=total_frames, desc="处理进度")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        mel, frame_input = prepare_input(frame)
        heatmap = get_clean_attention(model, mel, frame_input)

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        heatmap = cv2.resize(heatmap, (width, height))
        heatmap = cv2.GaussianBlur(heatmap, (15, 15), 3)
        heatmap_8u = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(255 - heatmap_8u, cv2.COLORMAP_JET)

        result = cv2.addWeighted(img_rgb, 0.6, heatmap_color, 0.5, 0)
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

        out.write(result_bgr)
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()
    print(f"\n✅ 热力图视频已保存：{SAVE_VIDEO_PATH}")


if __name__ == "__main__":
    process_video_heatmap(VIDEO_PATH)