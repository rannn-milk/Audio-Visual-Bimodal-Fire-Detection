import sys
definitions_path = '/tmp/pycharm_project_240/my/'
sys.path.append(definitions_path)
import torch.nn.functional as F
import cv2
from transformer import AudioVisualSpikformer
import torch
import torch.nn as nn
import os
import glob
import random
from timm.utils import AverageMeter, accuracy
import re
import matplotlib.pyplot as plt
import numpy as np

# ====================== 数据加载
def load_all_data1(root_dir):
    data = []
    pattern = re.compile(r'(\d+)s\.[jpg|png|jpeg]', re.I)
    for label in ["0", "1"]:
        mel_dir = os.path.join(root_dir, "mels", label)
        frame_dir = os.path.join(root_dir, "frames", label)
        if not os.path.exists(mel_dir) or not os.path.exists(frame_dir):
            continue
        for mel_path in glob.glob(os.path.join(mel_dir, "*.*")):
            mel_name = os.path.basename(mel_path)
            match = pattern.search(mel_name)
            if not match:
                continue
            sec = match.group(1)
            frame_files = glob.glob(os.path.join(frame_dir, f"*-{sec}s.*"))
            if not frame_files:
                continue
            frame_path = frame_files[0]
            data.append((frame_path, mel_path, int(label)))
    random.shuffle(data)
    split = int(0.8 * len(data))
    print(f"总数据量：{len(data)}")
    return data[:split], data[split:]

def load_image(path, is_gray=False, img_size=128):
    if is_gray:
        img = cv2.imread(path, 0)
    else:
        img = cv2.imread(path)
    img = cv2.resize(img, (img_size, img_size))
    img = torch.from_numpy(img).float() / 255.0
    if not is_gray:
        img = img.permute(2, 0, 1)
    else:
        img = img.unsqueeze(0)
    return img

def make_batch(batch_data, T=10):
    frames = []
    mels = []
    labels = []
    for fpath, mpath, lab in batch_data:
        f = load_image(fpath, is_gray=False).unsqueeze(0).repeat(T,1,1,1)
        m = load_image(mpath, is_gray=True).unsqueeze(0).repeat(T,1,1,1)
        frames.append(f)
        mels.append(m)
        labels.append(torch.tensor(lab))
    return torch.stack(frames).permute(1,0,2,3,4), torch.stack(mels).permute(1,0,2,3,4), torch.stack(labels)

def class_contrastive_loss(feature_1, feature_2, label, batch_size, temperature=0.01):
    T, _, _ = feature_1.shape

    feature_1 = feature_1.mean(0)
    feature_2 = feature_2.mean(0)

    class_matrix = label.unsqueeze(0)
    class_matrix = class_matrix.repeat(class_matrix.shape[1], 1)
    class_matrix = (class_matrix == label.unsqueeze(-1)).float()
    score = torch.mm(feature_1, feature_2.transpose(0, 1)) / temperature
    loss = -torch.mean(torch.mean(F.log_softmax(score, dim=-1) * class_matrix, dim=-1))
    return loss

# ====================== 统一训练测试函数
# ====================== 统一训练测试函数（终极版）
def run_exp(attn_method, exp_name):
    data_root = "/tmp/tmp/pycharm_project_599/datasum"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    T = 10
    num_classes = 2
    batch_size = 4
    epochs = 30
    lr = 1e-4

    train_data, test_data = load_all_data1(data_root)

    print("\n" + "=" * 60)
    print(f" Running: {exp_name} ")
    print(f" attn_method = {attn_method} ")
    print("=" * 60)

    model = AudioVisualSpikformer(
        step=T, num_classes=num_classes,
        attn_method=attn_method,
        cross_attn=True,
        interaction='Add',
        contrastive=True, num_heads=16, depths=2
    ).to(device)
    model = torch.nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    ce_loss = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 训练
    model.train()
    for epoch in range(epochs):
        losses_m = AverageMeter()
        top1_m = AverageMeter()
        random.shuffle(train_data)

        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i + batch_size]
            video, audio, labels = make_batch(batch, T=T)
            video, audio, labels = video.to(device), audio.to(device), labels.to(device)

            output, af, vf = model([audio, video])
            if output.size(0) != labels.size(0):
                output = output[:labels.size(0)]

            # 核心分类损失
            loss_ce = ce_loss(output, labels)

            # ====================== ✅ 关键：CMCI 不计算对比损失
            if attn_method == "CMCI":
                total_loss = loss_ce  # 只用分类损失
            else:
                # 其他注意力：使用对比损失
                loss_con = class_contrastive_loss(af, vf, labels, labels.size(0))
                total_loss = loss_ce + 0.5 * loss_con

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            acc1, _ = accuracy(output, labels, topk=(1, 5))
            losses_m.update(total_loss.item(), labels.size(0))
            top1_m.update(acc1.item(), labels.size(0))

        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:2d} | Loss {losses_m.avg:.4f} | Acc {top1_m.avg:.2f}%")

    # 测试
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for sample in test_data:
            video, audio, label = make_batch([sample], T=T)
            video, audio, label = video.to(device), audio.to(device), label.to(device)
            output, _, _ = model([audio, video])
            output = output.mean(dim=0, keepdim=True)
            pred = output.argmax(dim=1)
            correct += (pred == label).item()
            total += 1

    acc = 100 * correct / total
    print(f"\n✅ {exp_name}  测试准确率 = {acc:.2f}%\n")
    return acc
# ====================== 🔥 运行4组消融实验（画图用）
if __name__ == "__main__":
    # 这里是你要的 4 组实验
    exp_list = [
        ("SpatialTemporal", "Spatiotemporal Attention"),
        ("CMCI",            "Concat"),
        ("Spatial",         "Spatial-only Attention"),
        ("Temporal",        "Temporal-only Attention"),

    ]

    results = []
    names = []

    for attn, name in exp_list:
        acc = run_exp(attn, name)
        results.append(acc)
        names.append(name)

    # ====================== 📊 输出论文表格
    print("\n" + "="*70)
    print("           📊 消融实验结果（时空注意力有效性）           ")
    print("="*70)
    for n, acc in zip(names, results):
        print(f"{n:30s} |  Accuracy = {acc:.2f}%")

    # ====================== 📊 画论文柱状图
    plt.figure(figsize=(12, 6))
    bars = plt.bar(names, results, color = ['#888888', '#FFA500', '#1E90FF', '#FF4500'])
    plt.ylim(min(results)-5, max(results)+5)
    plt.title("Ablation Study: Spatial-Temporal Attention Fusion")
    plt.ylabel("Accuracy (%)")
    plt.xticks(rotation=10, fontsize=10)

    # 显示数值
    for b, v in zip(bars, results):
        plt.text(b.get_x()+b.get_width()/2, b.get_height()+0.3, f"{v:.1f}%", ha="center", fontsize=12)

    plt.tight_layout()
    plt.savefig("/tmp/tmp/pycharm_project_599/my/ablation_attention.png", dpi=300)
    plt.show()