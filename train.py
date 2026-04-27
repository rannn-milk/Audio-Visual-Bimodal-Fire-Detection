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
    # 匹配时间：提取 数字s
    pattern = re.compile(r'(\d+)s\.[jpg|png|jpeg]', re.I)

    for label in ["0", "1"]:
        mel_dir = os.path.join(root_dir, "mels", label)
        frame_dir = os.path.join(root_dir, "frames", label)
        if not os.path.exists(mel_dir) or not os.path.exists(frame_dir):
            continue

        # 遍历所有梅尔图
        for mel_path in glob.glob(os.path.join(mel_dir, "*.*")):
            mel_name = os.path.basename(mel_path)

            # 提取 时间秒数（兼容 0-5s 和 0-2-5s 两种格式）
            match = pattern.search(mel_name)
            if not match:
                continue
            sec = match.group(1)  # 拿到 5、10、20 这种时间

            # 拼接帧路径（自动匹配对应秒数的帧）
            frame_files = glob.glob(os.path.join(frame_dir, f"*-{sec}s.*"))
            if not frame_files:
                continue

            frame_path = frame_files[0]
            data.append((frame_path, mel_path, int(label)))

    random.shuffle(data)
    split = int(0.8 * len(data))
    print(f"总数据量：{len(data)}")
    return data[:split], data[split:]


# 图像预处理（只转成模型需要的张量）
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


# 造一个批次
def make_batch(batch_data, T=10):
    frames = []
    mels = []
    labels = []
    for fpath, mpath, lab in batch_data:
        f = load_image(fpath, is_gray=False).unsqueeze(0).repeat(T, 1, 1, 1)
        m = load_image(mpath, is_gray=True).unsqueeze(0).repeat(T, 1, 1, 1)
        frames.append(f)
        mels.append(m)
        labels.append(torch.tensor(lab))
    return torch.stack(frames).permute(1, 0, 2, 3, 4), torch.stack(mels).permute(1, 0, 2, 3, 4), torch.stack(labels)


def class_contrastive_loss(feature_1, feature_2, label, batch_size, temperature=0.01):
    """
        input shape: [T, B, C]
    """
    T, _, _ = feature_1.shape

    feature_1 = feature_1.mean(0)  # B, C
    feature_2 = feature_2.mean(0)  # B, C

    loss = 0.

    class_matrix = label.unsqueeze(0)  # 标签升维 → [1,B]
    class_matrix = class_matrix.repeat(class_matrix.shape[1], 1)  # 重复行 → [B,B]
    class_matrix = class_matrix == label.unsqueeze(-1)
    class_matrix = class_matrix.float()

    score = torch.mm(feature_1, feature_2.transpose(0, 1)) / temperature
    loss += -torch.mean(torch.mean(F.log_softmax(score, dim=-1) * class_matrix, dim=-1))
    return loss


# ============================ 双卡训练主函数 ============================
if __name__ == "__main__":
    # 🔥 启用双卡（关键）
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    data_root = "/tmp/tmp/pycharm_project_599/datasum"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    T = 9
    num_classes = 2
    batch_size = 4
    epochs = 100
    lr = 1e-4

    print("加载数据集...")
    train_data, test_data = load_all_data1(data_root)
    print(f"训练集：{len(train_data)} | 测试集：{len(test_data)}")

    model = AudioVisualSpikformer(
        step=T, num_classes=num_classes,
        attn_method='SpatialTemporal',
        cross_attn=True, interaction='Add',
        contrastive=True, num_heads=8, depths=2
    )


    model = torch.nn.DataParallel(model)
    model = model.to(device)  # 再送入GPU

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    ce_loss = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

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
            total_loss = loss + 0.1 * loss_con

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            acc1, _ = accuracy(output, labels, topk=(1, 5))
            losses_m.update(total_loss.item(), labels.size(0))
            top1_m.update(acc1.item(), labels.size(0))

        scheduler.step()
        print(f"Epoch {epoch + 1:2d} | Loss: {losses_m.avg:.4f} | Acc@1: {top1_m.avg:.2f}%")

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