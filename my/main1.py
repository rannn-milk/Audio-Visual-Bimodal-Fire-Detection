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
import numpy as np
import matplotlib.pyplot as plt
from timm.utils import AverageMeter, accuracy
import re

# 读取数据集（mels/0、mels/1 → 标签0/1）
def load_all_data(root_dir):
    data = []
    for label in ["0", "1"]:
        mel_dir = os.path.join(root_dir, "mels", label)
        frame_dir = os.path.join(root_dir, "frames", label)
        if not os.path.exists(mel_dir) or not os.path.exists(frame_dir):
            continue

        for mel_path in glob.glob(os.path.join(mel_dir, "*.jpg")):
            name = os.path.basename(mel_path).replace("mel_", "").replace("s.jpg", "")
            frame_path = os.path.join(frame_dir, f"{name}s.jpg")
            if os.path.exists(frame_path):
                data.append((frame_path, mel_path, int(label)))

    random.shuffle(data)
    split = int(0.8 * len(data))
    print(len(data))
    return data[:split], data[split:]


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

# 图像预处理
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

# 造批次
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
    loss = 0.
    class_matrix = label.unsqueeze(0)
    class_matrix = class_matrix.repeat(class_matrix.shape[1], 1)
    class_matrix = class_matrix == label.unsqueeze(-1)
    class_matrix = class_matrix.float()
    score = torch.mm(feature_1, feature_2.transpose(0, 1)) / temperature
    loss += -torch.mean(torch.mean(F.log_softmax(score, dim=-1) * class_matrix, dim=-1))
    return loss


# ============================
# 训练 + 测试 函数（方便多次调用）
# ============================
def run_experiment(train_data, test_data, alpha=1.0, num_heads=8, epochs=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = 8
    num_classes = 2
    batch_size = 4
    lr = 1e-4

    model = AudioVisualSpikformer(
        step=T, num_classes=num_classes,
        attn_method='SpatialTemporal',
        cross_attn=True, interaction='Add',
        contrastive=True, num_heads=num_heads, depths=2
    ).to(device)

    model = torch.nn.DataParallel(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    ce_loss = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for epoch in range(epochs):
        losses_m = AverageMeter()
        top1_m = AverageMeter()
        random.shuffle(train_data)

        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i + batch_size]
            video, audio, labels = make_batch(batch, T=T)
            video = video.to(device)
            audio = audio.to(device)
            labels = labels.to(device)

            output, af, vf = model([audio, video])

            if output.size(0) != labels.size(0):
                output = output[:labels.size(0)]

            loss_ce = ce_loss(output, labels)
            loss_con = class_contrastive_loss(af, vf, labels, labels.size(0))
            total_loss = loss_ce + alpha * loss_con  # ✅ alpha 在这里

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            acc1, _ = accuracy(output, labels, topk=(1, 5))
            losses_m.update(total_loss.item(), labels.size(0))
            top1_m.update(acc1.item(), labels.size(0))

        scheduler.step()
        print(f"[Epoch {epoch+1:2d}] Loss: {losses_m.avg:.4f} | Acc: {top1_m.avg:.2f}%")

    # 测试
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for sample in test_data:
            video, audio, label = make_batch([sample], T=T)
            video = video.to(device)
            audio = audio.to(device)
            label = label.to(device)
            output, _, _ = model([audio, video])
            output = output.mean(dim=0, keepdim=True)
            pred = output.argmax(dim=1)
            correct += (pred == label).item()
            total += 1

    test_acc = correct / total * 100
    print(f"\n✅ 实验结束 | alpha={alpha} heads={num_heads} | 测试准确率={test_acc:.2f}%\n")
    return test_acc


# ============================
# 主实验：超参数搜索
# ============================
if __name__ == "__main__":
    data_root = "/tmp/tmp/pycharm_project_599/datasum"
    train_data, test_data = load_all_data1(data_root)

    # ======================
    # 实验 1：残差系数 α
    # ======================
    '''alphas = [0.0,  0.05, 0.1, 0.5, 1.0]
    alpha_acc = []
    print("\n" + "="*50)
    print("开始实验 1：残差系数 α 对比")
    print("="*50)
    for a in alphas:
        acc = run_experiment(train_data, test_data, alpha=a, num_heads=8, epochs=30)
        alpha_acc.append(acc)

    # 绘图
    plt.figure(figsize=(8,5))
    plt.plot(alphas, alpha_acc, marker='o', linewidth=2, markersize=9, color='red')
    plt.xlabel("Residual Fusion Coefficient α", fontsize=12)
    plt.ylabel("Test Accuracy (%)", fontsize=12)
    plt.title("Effect of α on Model Performance", fontsize=14)
    plt.grid(linestyle='--', alpha=0.3)
    plt.xticks(alphas)
    for i, v in enumerate(alpha_acc):
        plt.text(alphas[i], v+0.3, f"{v:.1f}", ha='center', fontsize=11)
    plt.tight_layout()
    plt.savefig("/tmp/tmp/pycharm_project_599/my/alpha_experiment.png", dpi=300)
    plt.close()'''

    # ======================
    # 实验 2：注意力头数 num_heads
    # ======================
    '''heads_list = [4,8,16,32]
    head_acc = []
    print("\n" + "="*50)
    print("🔬 开始实验 2：注意力头数 对比")
    print("="*50)
    for h in heads_list:
        acc = run_experiment(train_data, test_data, alpha=0.1, num_heads=h, epochs=30)
        head_acc.append(acc)

    plt.figure(figsize=(8,5))
    plt.plot(heads_list, head_acc, marker='s', linewidth=2, markersize=9, color='blue')
    plt.xlabel("Number of Attention Heads", fontsize=12)
    plt.ylabel("Test Accuracy (%)", fontsize=12)
    plt.title("Effect of Num Heads on Model Performance", fontsize=14)
    plt.grid(linestyle='--', alpha=0.3)
    plt.xticks(heads_list)
    for i, v in enumerate(head_acc):
        plt.text(heads_list[i], v+0.3, f"{v:.1f}", ha='center', fontsize=11)
    plt.tight_layout()
    plt.savefig("/tmp/tmp/pycharm_project_599/my/num_heads_experiment.png", dpi=300)
    plt.close()'''

    T_list = [4, 6, 8, 10,12]  # 你可以自己改
    t_acc = []
    print("\n" + "=" * 50)
    print("🔬 开始实验 3：时间步 T 对比")
    print("=" * 50)
    for t_val in T_list:
        # 把 T 传入 run_experiment
        def run_experiment_T(train_data, test_data, alpha=0.1, num_heads=8, epochs=2, T=t_val):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            num_classes = 2
            batch_size = 4
            lr = 1e-4

            model = AudioVisualSpikformer(
                step=T, num_classes=num_classes,
                attn_method='SpatialTemporal',
                cross_attn=True, interaction='Add',
                contrastive=True, num_heads=num_heads, depths=2
            ).to(device)

            model = torch.nn.DataParallel(model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
            ce_loss = nn.CrossEntropyLoss()
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

            model.train()
            for epoch in range(epochs):
                losses_m = AverageMeter()
                top1_m = AverageMeter()
                random.shuffle(train_data)

                for i in range(0, len(train_data), batch_size):
                    batch = train_data[i:i + batch_size]
                    video, audio, labels = make_batch(batch, T=T)
                    video = video.to(device)
                    audio = audio.to(device)
                    labels = labels.to(device)

                    output, af, vf = model([audio, video])

                    if output.size(0) != labels.size(0):
                        output = output[:labels.size(0)]

                    loss_ce = ce_loss(output, labels)
                    loss_con = class_contrastive_loss(af, vf, labels, labels.size(0))
                    total_loss = loss_ce + alpha * loss_con

                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

                    acc1, _ = accuracy(output, labels, topk=(1, 5))
                    losses_m.update(total_loss.item(), labels.size(0))
                    top1_m.update(acc1.item(), labels.size(0))

                scheduler.step()
                print(f"[Epoch {epoch + 1:2d}] Loss: {losses_m.avg:.4f} | Acc: {top1_m.avg:.2f}%")

            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for sample in test_data:
                    video, audio, label = make_batch([sample], T=T)
                    video = video.to(device)
                    audio = audio.to(device)
                    label = label.to(device)
                    output, _, _ = model([audio, video])
                    output = output.mean(dim=0, keepdim=True)
                    pred = output.argmax(dim=1)
                    correct += (pred == label).item()
                    total += 1

            test_acc = correct / total * 100
            print(f"\n✅ T={t_val} 实验结束 | 测试准确率={test_acc:.2f}%\n")
            return test_acc


        acc = run_experiment_T(train_data, test_data, alpha=0.1, num_heads=16, epochs=30, T=t_val)
        t_acc.append(acc)

    # 绘制 T 实验折线图
    plt.figure(figsize=(8, 5))
    plt.plot(T_list, t_acc, marker='o', linewidth=2, markersize=9, color='green')
    plt.xlabel("Time Step T", fontsize=12)
    plt.ylabel("Test Accuracy (%)", fontsize=12)
    plt.title("Effect of Time Step T on Model Performance", fontsize=14)
    plt.grid(linestyle='--', alpha=0.3)
    plt.xticks(T_list)
    for i, v in enumerate(t_acc):
        plt.text(T_list[i], v + 0.3, f"{v:.1f}", ha='center', fontsize=11)
    plt.tight_layout()
    plt.savefig("/tmp/tmp/pycharm_project_599/my/T_experiment.png", dpi=300)
    plt.close()

    print("\n🎉 全部超参数实验完成！图表已保存：")
    print("1. alpha_experiment.png")
    print("2. num_heads_experiment.png")