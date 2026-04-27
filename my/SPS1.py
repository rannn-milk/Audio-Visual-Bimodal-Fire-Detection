import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import os
from pathlib import Path


# ============================
def to_2tuple(x):
    return (x, x)


# ============================
#  LIF 神经元
# ============================
class MyNode(nn.Module):
    def __init__(self, step=10, tau=2.0, v_threshold=1.0):
        super().__init__()
        self.step = step
        self.tau = tau
        self.v_threshold = v_threshold
        self.membrane_potential = None

    def reset(self):
        self.membrane_potential = None

    def forward(self, x):
        if self.membrane_potential is None:
            self.membrane_potential = torch.zeros_like(x)

        self.membrane_potential = self.membrane_potential / self.tau + x
        spike = (self.membrane_potential >= self.v_threshold).float()
        self.membrane_potential = self.membrane_potential * (1 - spike)
        return spike



class SPS(nn.Module):
    def __init__(self, step=10, img_size_h=128, img_size_w=128, patch_size=4, in_channels=3, embed_dims=256):
        super().__init__()
        self.step = step
        self.image_size = [img_size_h, img_size_w]

        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W

        # 4层卷积
        self.proj_conv = nn.Conv2d(in_channels, embed_dims // 8, 3, 1, 1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims // 8)
        self.proj_lif = MyNode(step=step)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.proj_conv1 = nn.Conv2d(embed_dims // 8, embed_dims // 4, 3, 1, 1, bias=False)
        self.proj_bn1 = nn.BatchNorm2d(embed_dims // 4)
        self.proj_lif1 = MyNode(step=step)
        self.maxpool1 = nn.MaxPool2d(3, 2, 1)

        self.proj_conv2 = nn.Conv2d(embed_dims // 4, embed_dims // 2, 3, 1, 1, bias=False)
        self.proj_bn2 = nn.BatchNorm2d(embed_dims // 2)
        self.proj_lif2 = MyNode(step=step)
        self.maxpool2 = nn.MaxPool2d(3, 2, 1)

        self.proj_conv3 = nn.Conv2d(embed_dims // 2, embed_dims, 3, 1, 1, bias=False)
        self.proj_bn3 = nn.BatchNorm2d(embed_dims)
        self.proj_lif3 = MyNode(step=step)
        self.maxpool3 = nn.MaxPool2d(3, 2, 1)

        # 位置编码
        self.rpe_conv = nn.Conv2d(embed_dims, embed_dims, 3, 1, 1, bias=False)
        self.rpe_bn = nn.BatchNorm2d(embed_dims)
        self.rpe_lif = MyNode(step=step)

    def reset(self):
        self.proj_lif.reset()
        self.proj_lif1.reset()
        self.proj_lif2.reset()
        self.proj_lif3.reset()
        self.rpe_lif.reset()

    def forward(self, x):
        self.reset()
        T, B, C, H, W = x.shape

        x = self.proj_conv(x.flatten(0, 1))
        x = self.proj_bn(x).view(T, B, -1, H, W)
        x = self.proj_lif(x.flatten(0, 1))
        x = self.maxpool(x)

        x = self.proj_conv1(x)
        x = self.proj_bn1(x).view(T, B, -1, H // 2, W // 2)
        x = self.proj_lif1(x.flatten(0, 1))
        x = self.maxpool1(x)

        x = self.proj_conv2(x)
        x = self.proj_bn2(x).view(T, B, -1, H // 4, W // 4)
        x = self.proj_lif2(x.flatten(0, 1))
        x = self.maxpool2(x)

        x = self.proj_conv3(x)
        x = self.proj_bn3(x).view(T, B, -1, H // 8, W // 8)
        x = self.proj_lif3(x.flatten(0, 1))
        x = self.maxpool3(x)

        x_rpe = self.rpe_bn(self.rpe_conv(x)).view(T, B, -1, H // 16, W // 16)
        x_rpe = self.rpe_lif(x_rpe.flatten(0, 1))
        x = x + x_rpe
        x = x.view(T, B, -1, (H // 16) * (W // 16))

        return x


# ============================
# 图片 → 5维张量 (T,1,C,128,128)
# ============================
def to_5D(img_path, is_gray=False, T=10, img_size=128):
    if is_gray:
        img = cv2.imread(img_path, 0)
        img = cv2.resize(img, (img_size, img_size))
        img = torch.from_numpy(img).unsqueeze(0).float() / 255.0
    else:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (img_size, img_size))
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

    img = img.unsqueeze(0).unsqueeze(0)
    img = img.repeat(T, 1, 1, 1, 1)
    return img


# ============================
# 批量处理文件夹
# ============================
def process_folder(
        frame_folder="frames",
        mel_folder="mels",
        save_folder="sps_features"
):
    Path(save_folder).mkdir(exist_ok=True)
    Path(os.path.join(save_folder, "video")).mkdir(exist_ok=True)
    Path(os.path.join(save_folder, "audio")).mkdir(exist_ok=True)

    T = 10
    sps_video = SPS(step=T, in_channels=3, embed_dims=256)
    sps_audio = SPS(step=T, in_channels=1, embed_dims=256)

    for fname in os.listdir(frame_folder):
        if not fname.endswith(".jpg"):
            continue

        name = fname.replace("s.jpg", "")
        print(f"处理：{name}s")

        # 视频
        img_path = os.path.join(frame_folder, fname)
        x = to_5D(img_path, is_gray=False)
        feat = sps_video(x)
        #torch.save(feat, os.path.join(save_folder, "video", f"{name}.pt"))

        # 音频
        mel_path = os.path.join(mel_folder, f"mel_{name}s.jpg")
        x_mel = to_5D(mel_path, is_gray=True)
        feat_mel = sps_audio(x_mel)
        #torch.save(feat_mel, os.path.join(save_folder, "audio", f"{name}.pt"))

    print("处理完成！")


# ============================
# 运行
# ============================
if __name__ == "__main__":
    process_folder(
        frame_folder="/tmp/pycharm_project_240/my/data/frames",
        mel_folder="/tmp/pycharm_project_240/my/data/mels",
        save_folder="/tmp/pycharm_project_240/my/data/sps_features"
    )



'''
if __name__ == "__main__":
        # 1. 随便选一张图片测试（你可以改成自己的路径）
        test_img_path = "/tmp/pycharm_project_240/my/data/frames/20s.jpg"  # 视频帧
        test_mel_path = "/tmp/pycharm_project_240/my/data/mels/mel_20s.jpg"  # 梅尔谱

        # 2. 转为5维
        img_5d = to_5D(test_img_path, is_gray=False)
        mel_5d = to_5D(test_mel_path, is_gray=True)

        # 3. 初始化SPS
        sps_video = SPS(in_channels=3)
        sps_audio = SPS(in_channels=1)

        # 4. 送入SPS得到输出
        video_out = sps_video(img_5d)
        audio_out = sps_audio(mel_5d)

        # ============================
        # ✅ 这里就是看输出！！！
        # ============================
        print("=" * 50)
        print("第20秒")
        print("视频shape :", video_out.shape)
        print("音频shape :", audio_out.shape)
        print("=" * 50)
        print("视频输出的前5个值：", video_out.flatten()[:5])
        print("音频输出的前5个值：", audio_out.flatten()[:5])
        print("=" * 50)
        print("输出里有哪些值：", video_out.unique())
        '''
