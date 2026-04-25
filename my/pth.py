import sys
definitions_path = '/tmp/pycharm_project_240/my/'

sys.path.append(definitions_path)
import torch.nn.functional as F
import cv2
from transformer import AudioVisualSpikformer
import torch
import torch.nn as nn
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import glob
import random
from timm.utils import AverageMeter, accuracy
import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ============================
# 数据读取函数（不变）
# ============================
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
    loss = 0.
    class_matrix = label.unsqueeze(0)
    class_matrix = class_matrix.repeat(class_matrix.shape[1], 1)
    class_matrix = class_matrix == label.unsqueeze(-1)
    class_matrix = class_matrix.float()
    score = torch.mm(feature_1, feature_2.transpose(0, 1)) / temperature
    loss += -torch.mean(torch.mean(F.log_softmax(score, dim=-1) * class_matrix, dim=-1))
    return loss

# ============================
if __name__ == "__main__":
    data_root = "/tmp/tmp/pycharm_project_599/datasum"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    T = 10
    num_classes = 2
    batch_size = 4
    epochs = 35
    lr = 1e-4
    alpha = 0.5

    print("加载数据集...")
    train_data, test_data = load_all_data1(data_root)
    print(f"训练集：{len(train_data)} | 测试集：{len(test_data)}")

    model = AudioVisualSpikformer(
        step=T, num_classes=num_classes,
        attn_method='SpatialTemporal',
        cross_attn=True, interaction='Add',
        contrastive=True, num_heads=16, depths=2
    ).to(device)
    model = torch.nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    ce_loss = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_loss_history = []
    train_acc_history = []

    print("开始训练")
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

            loss = ce_loss(output, labels)
            loss_con = class_contrastive_loss(af, vf, labels, labels.size(0))
            total_loss = loss + alpha * loss_con

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            acc1, _ = accuracy(output, labels, topk=(1, 5))
            losses_m.update(loss.item(), labels.size(0))
            top1_m.update(acc1.item(), labels.size(0))

        scheduler.step()
        train_loss_history.append(losses_m.avg)
        train_acc_history.append(top1_m.avg)

        # ========== 每 5 个 epoch 打印一次 ==========
        if (epoch + 1) % 5 == 0:
            print(f"[Epoch {epoch+1:2d}] Loss: {losses_m.avg:.4f} | Acc: {top1_m.avg:.2f}%")

    # ============================
    # ✅ 绘图：每隔 5 个 epoch 打点
    # ============================
    epochs_range = list(range(1, epochs+1))
    epochs_plot = epochs_range[::5]
    loss_plot = train_loss_history[::5]
    acc_plot = train_acc_history[::5]

    plt.figure(figsize=(10, 5))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    ax1.plot(epochs_plot, loss_plot, 'b-o', linewidth=2, markersize=6, label='Train Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2.plot(epochs_plot, acc_plot, 'r-s', linewidth=2, markersize=6, label='Train Accuracy')
    ax2.set_ylabel('Accuracy (%)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    plt.title('Training Loss & Accuracy Curve')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("/tmp/tmp/pycharm_project_599/my/training_curve.png", dpi=300)
    plt.close()
    print("✅ 训练曲线已保存：/tmp/tmp/pycharm_project_599/my/training_curve.png")

    # 测试
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for single_sample in test_data:
            video, audio, label = make_batch([single_sample], T=T)
            video = video.to(device)
            audio = audio.to(device)
            label = label.to(device)
            output, _, _ = model([audio, video])
            output = output.mean(dim=0, keepdim=True)
            pred = output.argmax(dim=1)
            correct += (pred == label).item()
            total += 1

    test_acc = correct / total * 100
    print(f"测试准确率: {test_acc:.2f}%")

    # ==============================================
    # ✅【核心新增】训练结束后自动保存 .pth 模型文件
    # ==============================================
    save_path = "/tmp/tmp/pycharm_project_599/my/av_spikformer_best.pth"
    torch.save(model.module.state_dict(), save_path)  # 因为用了 DataParallel
    print(f"\n✅ 模型已保存至：{save_path}")