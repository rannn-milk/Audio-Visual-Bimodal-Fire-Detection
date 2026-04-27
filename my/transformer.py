import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from timm.utils import AverageMeter, accuracy


class MyNode(nn.Module):
    def __init__(self, step=10, tau=2.0, v_threshold=1.0):
        super().__init__()
        self.step = step
        self.tau = tau
        self.v_threshold = v_threshold
        self.membrane_potential = None

    def n_reset(self):
        self.membrane_potential = None

    def forward(self, x):
        if self.membrane_potential is None:
            self.membrane_potential = torch.zeros_like(x)
        self.membrane_potential = self.membrane_potential / self.tau + x
        out = (self.membrane_potential >= self.v_threshold).float()
        self.membrane_potential[out == 1] = 0.0
        return out


# 脉冲mlp
class MLP(nn.Module):
    def __init__(self, in_features, step=10, encode_type='direct', hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # 卷积 + BN + 脉冲神经元
        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = MyNode(step=step, tau=2.0)

        self.fc2_conv = nn.Conv1d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = MyNode(step=step, tau=2.0)

        self.c_hidden = hidden_features
        self.c_output = out_features

        self.id = nn.Identity()

    def reset(self):
        for m in self.modules():
            if hasattr(m, 'n_reset'):
                m.n_reset()

    def forward(self, x):
        self.reset()

        T, B, C, N = x.shape

        x = self.id(x)

        x = self.fc1_conv(x.flatten(0, 1))  # 将(T, B, C, N)展平为(T×B, C, N)
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, N).contiguous()  # T B C N 时间步 批次 通道数 特征维度
        # 再次把(T, B, H, N) → (T×B, H, N)，原因和 Conv1d 一样：MyNode（脉冲神经元）接收 3 维输入
        x = self.fc1_lif(x.flatten(0, 1)).reshape(T, B, self.c_hidden, N).contiguous()

        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, C, N).contiguous()
        x = self.fc2_lif(x.flatten(0, 1)).reshape(T, B, C, N).contiguous()
        return x


class MLPBlock(nn.Module):
    def __init__(self, dim, step=10, mlp_ratio=4., drop=0.):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, step=step, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.mlp(x)
        return x


def to_2tuple(x):
    return (x, x)


class SPS(nn.Module):
    def __init__(self, step=10, encode_type='direct', img_size_h=64, img_size_w=64, patch_size=4, in_channels=2,
                 embed_dims=256, if_UCF=False):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]


        patch_size = to_2tuple(patch_size) #  将patch_size转为二维元组（4→(4,4)）
        self.patch_size = patch_size
        self.C = in_channels
        # 计算patch的高/宽：输入尺寸 ÷ patch大小
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W

        self.if_UCF = if_UCF

        # 4 层卷积逐步将通道数从in_channels提升到embed_dims（256），每层 MaxPool 将空间维度（H/W）减半，最终 H/W 变为原始的 1/16
        # 通道数 in_channels → embed_dims//8
        self.proj_conv = nn.Conv2d(in_channels, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims // 8)
        self.proj_lif = MyNode(step=step, tau=2.0)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)# 空间维度减半

        self.proj_conv1 = nn.Conv2d(embed_dims // 8, embed_dims // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn1 = nn.BatchNorm2d(embed_dims // 4)
        self.proj_lif1 = MyNode(step=step, tau=2.0)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv2 = nn.Conv2d(embed_dims // 4, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn2 = nn.BatchNorm2d(embed_dims // 2)
        self.proj_lif2 = MyNode(step=step, tau=2.0)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv3 = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn3 = nn.BatchNorm2d(embed_dims)
        self.proj_lif3 = MyNode(step=step, tau=2.0)
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        # 对最终卷积特征做位置编码
        self.rpe_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.rpe_bn = nn.BatchNorm2d(embed_dims)
        self.rpe_lif = MyNode(step=step, tau=2.0)

    def reset(self):
        self.membrane_potential = None


    def forward(self, x):
        self.reset()

        T, B, C, H, W = x.shape

        # UCF101DVS
        if self.if_UCF:
            x = F.adaptive_avg_pool2d(x.flatten(0, 1), output_size=(64, 64)).reshape(T, B, C, 64, 64)
            T, B, C, H, W = x.shape

        # 四次处理C-》embed_dims
        x = self.proj_conv(x.flatten(0, 1))  #(T×B,C,H,W)→(T×B, embed_dims//8, H, W)（通道数提升，空间维度不变）
        x = self.proj_bn(x).reshape(T, B, -1, H, W).contiguous() # (T×B,32,H,W)→(T,B,32,H,W)
        x = self.proj_lif(x.flatten(0, 1)).contiguous()
        x = self.maxpool(x)

        x = self.proj_conv1(x)
        x = self.proj_bn1(x).reshape(T, B, -1, H // 2, W // 2).contiguous()
        x = self.proj_lif1(x.flatten(0, 1)).contiguous()
        x = self.maxpool1(x)

        x = self.proj_conv2(x)
        x = self.proj_bn2(x).reshape(T, B, -1, H // 4, W // 4).contiguous()
        x = self.proj_lif2(x.flatten(0, 1)).contiguous()
        x = self.maxpool2(x)

        x = self.proj_conv3(x)
        x = self.proj_bn3(x).reshape(T, B, -1, H // 8, W // 8).contiguous()
        x = self.proj_lif3(x.flatten(0, 1)).contiguous()
        x = self.maxpool3(x)

        # 卷积+BN：(T×B,256,4,4)→(T×B,256,4,4)，reshape恢复为(T,B,256,4,4)
        x_rpe = self.rpe_bn(self.rpe_conv(x)).reshape(T, B, -1, H // 16, W // 16).contiguous()
        # 脉冲神经元：(T,B,256,4,4)→(T×B,256,4,4)，转换为脉冲序列
        x_rpe = self.rpe_lif(x_rpe.flatten(0, 1)).contiguous()
        x = x + x_rpe
        x = x.reshape(T, B, -1, (H // 16) * (W // 16)).contiguous()

        return x  # T B C N



class AVattention(nn.Module):
    def __init__(self, channel=512, av_attn_channel=64):
        super().__init__()
        self.d = av_attn_channel
        self.fc = nn.Linear(channel, self.d)  # 降维
        self.fcs = nn.ModuleList([])
        for i in range(2):  # 升维，需要为音频（x）、视觉（y）各生成 1 组注意力权重
            self.fcs.append(nn.Linear(self.d, channel))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        """
            x, y: [T, B, C]
            return [B, C]
        """
        x, y = x.mean(0), y.mean(0)  # # [T,B,C] → [B,C]：对时间步T求平均，融合所有时间步的脉冲特征

        b, c = x.size()  # # 获取批次B、通道C

        ### fuse
        U = x + y

        ### reduction channel
        Z = self.fc(U)  # 降维C-》d，维度变为B, d

        ### calculate attention weight
        weights = []
        for fc in self.fcs:
            weight = fc(Z)
            weights.append(weight.view(b, c))  # b, c
        attention_weights = torch.stack(weights, 0)  # k,b,c
        # 将 2 组权重从列表转为张量，维度从[B,C]×2变为[k（2）,B,C] ，每个模态、每个样本做sigmoid
        attention_weights = self.sigmoid(attention_weights)

        ### fuse 融合特征
        V = attention_weights[0] * x + attention_weights[1] * y
        return V


# SpatialAudioVisualSSA（空间注意力）和TemporalAudioVisualSSA（时间注意力）的联合模块
class SpatialTemporalAudioVisualSSA(nn.Module):
    def __init__(self, dim, step=10, encode_type='direct', num_heads=16, TIM_alpha=0.5, qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()



        self.spatial_attn = SpatialAudioVisualSSA(dim, step=step, TIM_alpha=TIM_alpha, num_heads=num_heads,
                                                  qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                  attn_drop=attn_drop, sr_ratio=sr_ratio)

        self.temporal_attn = TemporalAudioVisualSSA(dim, step=step, TIM_alpha=TIM_alpha, num_heads=num_heads,
                                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                    attn_drop=attn_drop, sr_ratio=sr_ratio)

    def reset(self):
        pass
    def forward(self, x, y):
        a = self.spatial_attn(x, y)
        b = self.temporal_attn(x, y)

        T, B, C, N = a.shape
        a_reduced = a.mean(dim=3)  # Shape: (T, B, C)
        b_reduced = b.mean(dim=0)  # Shape: (B, C, N)

        a_expanded = a_reduced.unsqueeze(-1)  # Shape: (T, B, C, 1)
        a_expanded = a_expanded.expand(-1, -1, -1, N)  # Shape: (T, B, C, N)，-1表示保留原维度，仅扩展最后一维到N

        b_expanded = b_reduced.unsqueeze(0)  # Shape: (1, B, C, N)
        b_expanded = b_expanded.expand(T, -1, -1, -1)  # Shape: (T, B, C, N)

        output = a_expanded * b_expanded  # 逐元素相乘
        return output


class AudioVisualBlock(nn.Module):
    def __init__(self, dim, num_heads, step=10, TIM_alpha=0.5, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1, attn_method='Spatial', alpha=1.0,
                 contrastive=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn_method = attn_method

        # ===================== 注意力初始化（你已修正，我保留） =====================
        if attn_method == "Spatial":
            self.attn = SpatialAudioVisualSSA(dim, step=step, TIM_alpha=TIM_alpha, num_heads=num_heads,
                                              qkv_bias=qkv_bias, qk_scale=qk_scale,
                                              attn_drop=attn_drop, sr_ratio=sr_ratio)
        elif attn_method == "Temporal":
            self.attn = TemporalAudioVisualSSA(dim, step=step, TIM_alpha=TIM_alpha, num_heads=num_heads,
                                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                                               attn_drop=attn_drop, sr_ratio=sr_ratio)
        elif attn_method == "SpatialTemporal":
            self.attn = SpatialTemporalAudioVisualSSA(dim, step=step, TIM_alpha=TIM_alpha, num_heads=num_heads,
                                                      qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                      attn_drop=attn_drop, sr_ratio=sr_ratio)
        elif attn_method == "WeightAttention":
            self.attn = WeightAttention(dim, step=step)
        elif attn_method == "SCA_AV":
            self.attn = SCA_AV(dim, TIM_alpha=TIM_alpha, num_heads=num_heads, qkv_bias=qkv_bias,
                               qk_scale=qk_scale, attn_drop=attn_drop, sr_ratio=sr_ratio)
        elif attn_method == "SCA_VA":
            self.attn = SCA_VA(dim, TIM_alpha=TIM_alpha, num_heads=num_heads, qkv_bias=qkv_bias,
                               qk_scale=qk_scale, attn_drop=attn_drop, sr_ratio=sr_ratio)
        elif attn_method == "CMCI":
            self.attn = CMCI(dim, step=step)

        self.norm2 = norm_layer(dim)
        self.alpha = alpha
        self.contrastive = contrastive

        if self.contrastive:
            self.trans_linear = nn.Linear(dim, dim)
            self.trans_lif = MyNode(step=step, tau=2.0, v_threshold=0.5)

    # 🔥 修复1：reset 函数正确定义（外部调用不报错）
    def reset(self):
        if hasattr(self.attn, 'reset'):
            self.attn.reset()

    # 🔥 修复2：forward 完全兼容所有注意力
    def forward(self, x, y):
        # 安全调用注意力
        SpatialTemporalBrach = self.attn(x, y)

        if self.attn_method == "CMCI":
            x = x

        elif self.attn_method == "WeightAttention" or self.attn_method == "SCA_AV" or self.attn_method == "SCA_VA":
            x = SpatialTemporalBrach  # 这三个直接赋值，不残差

        elif self.attn_method == "Spatial" or self.attn_method == "Temporal" or self.attn_method == "SpatialTemporal":
            if self.contrastive:
                T, B, C, N = SpatialTemporalBrach.shape
                SpatialTemporalBrach = self.trans_linear(SpatialTemporalBrach.transpose(2, 3).contiguous())
                self.trans_lif.n_reset()
                SpatialTemporalBrach = self.trans_lif(SpatialTemporalBrach.flatten(0, 1)).transpose(1,
                                                                                                    2).contiguous().reshape(
                    T, B, C, N)
            x = x + SpatialTemporalBrach * self.alpha

        return x, SpatialTemporalBrach


class SpatialAudioVisualSSA(nn.Module):
    def __init__(self, dim, step=10, encode_type='direct', num_heads=16, TIM_alpha=0.5, qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim

        self.num_heads = num_heads

        self.in_channels = dim // num_heads

        self.scale = 0.25

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MyNode(step=step, tau=2.0)

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MyNode(step=step, tau=2.0)

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MyNode(step=step, tau=2.0)

        self.attn_drop = nn.Dropout(0.2)
        self.res_lif = MyNode(step=step, tau=2.0)
        self.attn_lif = MyNode(step=step, tau=2.0, v_threshold=0.5, )

        # 输出投影层
        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = MyNode(step=step, tau=2.0, )

    def reset(self):
        pass
    def forward(self, x, y):
        self.reset()

        T, B, C, N = x.shape

        x_for_qkv = x.flatten(0, 1)  # [T,B,C,N] → [T×B,C,N]
        y_for_qkv = y.flatten(0, 1)  #

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, N).contiguous()
        q_conv_out = self.q_lif(q_conv_out.flatten(0, 1)).reshape(T, B, C, N).transpose(-2, -1)  # 转置为[T,B,N,C]
        q = q_conv_out.reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2,
                                                                                     4).contiguous()  # [T,B,num_heads,N,in_channels]

        k_conv_out = self.k_conv(y_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, N).contiguous()
        k_conv_out = self.k_lif(k_conv_out.flatten(0, 1)).reshape(T, B, C, N).transpose(-2, -1)
        k = k_conv_out.reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v_conv_out = self.v_conv(y_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T, B, C, N).contiguous()
        v_conv_out = self.v_lif(v_conv_out.flatten(0, 1)).reshape(T, B, C, N).transpose(-2, -1)
        v = v_conv_out.reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        # TIM on Q
        # q = self.TIM(q)

        # SSA
        attn = (q @ k.transpose(-2, -1))  # attn [T,B,num_heads,N,N]空间的注意力
        x = (attn @ v) * self.scale  #

        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x.flatten(0, 1))
        x = self.proj_lif(self.proj_bn(self.proj_conv(x))).reshape(T, B, C, N)

        return x


# 时间注意力，Q/K/V 的卷积 + BN + 脉冲神经元，音频生成 Q，视觉生成 K/V
class TemporalAudioVisualSSA(nn.Module):
    def __init__(self, dim, step=10, encode_type='direct', num_heads=16, TIM_alpha=0.5, qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim

        self.num_heads = num_heads

        self.in_channels = dim // num_heads

        self.scale = 0.25

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MyNode(step=step, tau=2.0)

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MyNode(step=step, tau=2.0)

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MyNode(step=step, tau=2.0)

        self.attn_drop = nn.Dropout(0.2)
        self.res_lif = MyNode(step=step, tau=2.0)
        self.attn_lif = MyNode(step=step, tau=2.0, v_threshold=0.5, )

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = MyNode(step=step, tau=2.0, )

    def reset(self):
        pass
    def forward(self, x, y):
        self.reset()

        T, B, C, N = x.shape
        # x =  x.permute(3, 1, 2, 0)  # N, B, C , T

        x_for_qkv = x.flatten(0, 1)
        y_for_qkv = y.flatten(0, 1)

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, N)
        q_conv_out = self.q_lif(q_conv_out.flatten(0, 1)).reshape(T, B, C, N).permute(3, 1, 0,
                                                                                      2).contiguous()  # T B C N -> N, B, T, C
        q = q_conv_out.reshape(N, B, T, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2,
                                                                                     4).contiguous()  # [N,B,num_heads,T,in_channels]

        k_conv_out = self.k_conv(y_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, N)
        k_conv_out = self.k_lif(k_conv_out.flatten(0, 1)).reshape(T, B, C, N).permute(3, 1, 0, 2).contiguous()
        k = k_conv_out.reshape(N, B, T, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v_conv_out = self.v_conv(y_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T, B, C, N)
        v_conv_out = self.v_lif(v_conv_out.flatten(0, 1)).reshape(T, B, C, N).permute(3, 1, 0, 2).contiguous()
        v = v_conv_out.reshape(N, B, T, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        # TIM on Q
        # q = self.TIM(q)

        # SSA
        attn = (q @ k.transpose(-2, -1))  # [N,B,num_heads,T,T]时间步的注意力
        x = (attn @ v) * self.scale

        x = x.reshape(N, B, T, C).permute(2, 1, 3, 0).contiguous()  # N B T C -> T, B, C, N
        x = self.attn_lif(x.flatten(0, 1))
        # x = self.proj_lif(self.proj_bn(self.proj_conv(x))).reshape(N, B, C, T).permute(3, 1, 2, 0).contiguous()  # N, B, C, T -> T, B, C, N
        x = self.proj_lif(self.proj_bn(self.proj_conv(x))).reshape(T, B, C, N)  # N, B, C, T -> T, B, C, N
        return x


class AudioVisualSpikformer(nn.Module):
    def __init__(self, step=10, TIM_alpha=0.5, if_UCF=False,
                 img_size_h=64, img_size_w=64, patch_size=16, num_classes=10,
                 num_heads=16, mlp_ratios=4, qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=2, sr_ratios=4, *args, **kwargs
                 ):  # embed_dims 默认是256
        super().__init__()
        self.T = step  # time step
        self.num_classes = num_classes
        self.depths = depths
        in_channels = kwargs['in_channels'] if 'in_channels' in kwargs else 2
        self.cross_attn = kwargs['cross_attn'] if 'cross_attn' in kwargs else False
        self.interaction = kwargs['interaction'] if 'interaction' in kwargs else False
        shallow_sps = kwargs['shallow_sps'] if 'shallow_sps' in kwargs else False
        contrastive = kwargs['contrastive'] if 'contrastive' in kwargs else False

        attn_method_list = ["init", "init"]
        attn_method = kwargs['attn_method'] if 'attn_method' in kwargs else None  # 下一步做成列表
        self.attn_method = attn_method
        if attn_method == "SCA":
            attn_method_list[0] = attn_method + "_AV"
            attn_method_list[1] = attn_method + "_VA"
        else:
            attn_method_list[0] = attn_method
            attn_method_list[1] = attn_method
        embed_dims = kwargs['embed_dims'] if 'embed_dims' in kwargs else 256
        alpha = kwargs['alpha'] if 'alpha' in kwargs else 1.0

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule


        # 初始化2层AudioVisualBlock
        block = nn.ModuleList([AudioVisualBlock(step=step, TIM_alpha=TIM_alpha,
                                                dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios,
                                                qkv_bias=qkv_bias,
                                                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=dpr[j],
                                                norm_layer=norm_layer, sr_ratio=sr_ratios,
                                                attn_method=attn_method_list[j], alpha=alpha, contrastive=contrastive)

                               for j in range(depths)])

        # 初始化2层MLPBlock
        mlp = nn.ModuleList([MLPBlock(dim=embed_dims, step=step, mlp_ratio=mlp_ratios, drop=drop_rate)
                             for j in range(depths)])

        av_attention = AVattention(channel=embed_dims, av_attn_channel=32)


        audio_patch_embed = SPS(step=step,
                                if_UCF=if_UCF,
                                img_size_h=img_size_h,
                                img_size_w=img_size_w,
                                patch_size=patch_size,
                                in_channels=1,
                                embed_dims=embed_dims)

        visual_patch_embed = SPS(step=step,
                                 if_UCF=if_UCF,
                                 img_size_h=img_size_h,
                                 img_size_w=img_size_w,
                                 patch_size=patch_size,
                                 in_channels=3,
                                 embed_dims=embed_dims)

        setattr(self, f"audio_patch_embed", audio_patch_embed)
        setattr(self, f"visual_patch_embed", visual_patch_embed)
        setattr(self, f"block", block)
        setattr(self, f"av_attention", av_attention)
        setattr(self, f"mlp", mlp)

        # classification head 分类头初始化
        if self.interaction == "Concat":
            embed_dims = 2 * embed_dims

        # CMCI特殊处理：3个分类头（辅助特征+音频+视觉）
        if self.attn_method == "CMCI":
            self.head = nn.ModuleList(
                [nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity() for j in range(3)])
        else:
            self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    @torch.jit.ignore
    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, audio, visual):

        block = getattr(self, f"block")
        audio_patch_embed = getattr(self, f"audio_patch_embed")
        visual_patch_embed = getattr(self, f"visual_patch_embed")
        mlp = getattr(self, f"mlp")

        # [T,B,C,H,W] → [T,B,embed_dims,N]
        audio = audio_patch_embed(audio)
        visual = visual_patch_embed(visual)

        audio_SpatialTemporalBrach = None
        visual_SpatialTemporalBrach = None

        if self.cross_attn:
            audio_feature, audio_SpatialTemporalBrach = block[0](audio, visual)  # 音频Q查视觉K/V
            visual_feature, visual_SpatialTemporalBrach = block[1](visual, audio)  # 视觉Q查音频K/V
            # 空间维度平均：[T,B,C,N] → [T,B,C]
            audio_SpatialTemporalBrach = audio_SpatialTemporalBrach.mean(-1)
            visual_SpatialTemporalBrach = visual_SpatialTemporalBrach.mean(-1)
        else:
            audio_feature = audio
            visual_feature = visual

        audio_feature = mlp[0](audio_feature)
        visual_feature = mlp[1](visual_feature)

        audio_feature = audio_feature.mean(-1)  # T B C N -> T B C
        visual_feature = visual_feature.mean(-1)  # T B C N -> T B C

        if self.interaction == "Add":
            fused_feature = audio_feature + visual_feature  # T B C
        elif self.interaction == "Concat":
            fused_feature = torch.cat((audio_feature, visual_feature), dim=-1)
        else:
            raise NotImplementedError
        #print(fused_feature.shape)
        fused_feature = fused_feature.mean(1)  # B C or B * 2C 时间平均
        audio_feature=audio_feature.permute(1,0,2)

        visual_feature=visual_feature.permute(1,0,2)


        if self.attn_method == "CMCI":
            return audio_SpatialTemporalBrach.mean(0), audio_feature.mean(0), visual_feature.mean(0)
        else:
            return fused_feature, audio_feature, visual_feature
            #return fused_feature, audio_SpatialTemporalBrach, visual_SpatialTemporalBrach

    def forward(self, x):
        audio, visual = x  # B T C H W
        audio = audio.permute(1, 0, 2, 3, 4)  # [B,T,C,H,W] → [T,B,C,H,W]

        visual = visual.permute(1, 0, 2, 3, 4)
        x, audio_feature, visual_feature = self.forward_features(audio, visual)  # T B C
        #print(x.shape)
        #print(audio_feature.shape)
        if self.attn_method == "CMCI":
            x = self.head[0](x)
            audio_feature = self.head[1](audio_feature)
            visual_feature = self.head[2](visual_feature)
        else:
            x = self.head(x)
        #print(x.shape)
        return x, audio_feature, visual_feature

    # 增量分类器，新增类别时，不重新训练原有类别权重
    def incremental_classifier(self, numclass):
        weight = self.head.weight.data
        bias = self.head.bias.data
        in_features = self.head.in_features
        out_features = self.head.out_features

        self.head = nn.Linear(in_features, numclass, bias=True)
        self.head.weight.data[:out_features] = weight
        self.head.bias.data[:out_features] = bias


class CMCI(nn.Module):
    def __init__(self, dim, step=10):
        super().__init__()
        self.dim = dim

        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.v_lif = MyNode(step=step, tau=2.0)
        self.mlp = MLPBlock(dim=dim, step=step, mlp_ratio=4, drop=0.)

    def reset(self):
        pass
    def forward(self, x, y):
        self.reset()

        T, B, C, N = x.shape

        x = self.fc1(x.transpose(-2, -1))  # T B N C
        y = self.fc2(y.transpose(-2, -1))  # T B N C

        current = (x + y).transpose(-2, -1)  # T B C N;

        x = self.v_lif(current)
        x = self.mlp(x)
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightAttention(nn.Module):
    def __init__(self, dim, step=10):
        super().__init__()
        self.dim = dim

        self.fc1 = nn.Linear(dim * 2, dim)
        self.fc2 = nn.Linear(dim, 1)

    def reset(self):
        pass
    def forward(self, x, y):
        self.reset()

        T, B, C, N = x.shape

        x_fea, y_fea = x.mean(-1), y.mean(-1) # 对音频/视觉特征在空间维度N上求平均 → 压缩为[T,B,C]

        # 1. Concatenate visual and auditory modality outputs
        concat_feat = torch.cat((x_fea, y_fea), dim=-1)  # [T,B,C] + [T,B,C] → [T,B,2*C]

        # 2. Pass concatenated features through FC layer to process them
        fusion_feat = self.fc1(concat_feat)      # [T, B, C]

        # 3. Estimate attention weights for both modalities (no softmax applied here)
        w = self.fc2(fusion_feat) # [T, B, 1]

        x = w.unsqueeze(-1) * x # [T,B,C,N]） × w（[T,B,1,1]

        return x



# 视觉（V）引导的音频（A）空间通道注意力
class SCA_VA(nn.Module):
    def __init__(self, dim, step=10, encode_type='direct', num_heads=16, TIM_alpha=0.5, qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim

        self.num_heads = num_heads

        self.in_channels = dim // num_heads

        self.scale = 0.25

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MyNode(step=step, tau=2.0)

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MyNode(step=step, tau=2.0)

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MyNode(step=step, tau=2.0)

        self.attn_drop = nn.Dropout(0.2)
        self.res_lif = MyNode(step=step, tau=2.0)
        self.attn_lif = MyNode(step=step, tau=2.0, v_threshold=0.5, )

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = MyNode(step=step, tau=2.0, )

        self.rpb_lif = MyNode(step=step, tau=2.0, )

    # 🔥 修复1：添加空 reset 方法
    def reset(self):
        pass

    def compute_RPB_matrix(self, N):
        H = W = int(N ** 0.5)
        M = H + 1
        P = torch.randn((2 * M - 1), (2 * M - 1))
        B = torch.zeros((N, N))
        for i in range(H):
            for j in range(W):
                i_x = i
                i_y = j
                for k in range(H):
                    for l in range(W):
                        j_x = k
                        j_y = l
                        f_val = i_x - j_x
                        g_val = i_y - j_y
                        idx = f_val + M - 1
                        idy = g_val + M - 1
                        if 0 <= idx < (2 * M - 1) and 0 <= idy < (2 * M - 1):
                            B[i * W + j, k * W + l] = P[idx, idy]
        return B

    def forward(self, x, y):
        # 🔥 修复2：删掉这行报错代码！
        # if hasattr(self.attn, 'reset'):
        #     self.reset()

        T, B, C, N = x.shape

        x_for_qkv = x.flatten(0, 1)
        y_for_qkv = y.flatten(0, 1)

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, N).contiguous()
        q_conv_out = self.q_lif(q_conv_out.flatten(0, 1)).reshape(T, B, C, N).transpose(-2, -1)
        q = q_conv_out.reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        k_conv_out = self.k_conv(y_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, N).contiguous()
        k_conv_out = self.k_lif(k_conv_out.flatten(0, 1)).reshape(T, B, C, N).transpose(-2, -1)
        k = k_conv_out.reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v_conv_out = self.v_conv(y_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T, B, C, N).contiguous()
        v_conv_out = self.v_lif(v_conv_out.flatten(0, 1)).reshape(T, B, C, N).transpose(-2, -1)
        v = v_conv_out.reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        attn = (q @ k.transpose(-2, -1))

        B_matrix = self.compute_RPB_matrix(N)
        B_matrix = B_matrix.reshape(1, 1, 1, N, N).to(attn.device)

        attn = (attn + B_matrix).reshape(T*B, -1, N*N)
        attn = self.rpb_lif(attn)
        attn = attn.reshape(T, B, -1, N, N)

        x = (attn @ v) * self.scale

        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x.flatten(0, 1))
        x = self.proj_lif(self.proj_bn(self.proj_conv(x))).reshape(T, B, C, N)

        return x

class SCA_AV(nn.Module):
    def __init__(self, dim, step=10, encode_type='direct', num_heads=16, TIM_alpha=0.5, qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim

        self.num_heads = num_heads

        self.in_channels = dim // num_heads

        self.scale = 0.25

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MyNode(step=step, tau=2.0)

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MyNode(step=step, tau=2.0)

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MyNode(step=step, tau=2.0)

        self.attn_drop = nn.Dropout(0.2)
        self.res_lif = MyNode(step=step, tau=2.0)
        self.attn_lif = MyNode(step=step, tau=2.0, v_threshold=0.5, )

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = MyNode(step=step, tau=2.0, )

        self.rpb_lif = MyNode(step=step, tau=2.0, )

    # 🔥 修复1：添加一个空的 reset 方法，解决调用报错
    def reset(self):
        pass

    # 🔥 修复2：forward 里删掉错误代码
    def forward(self, x, y):
        T, B, C, N = x.shape

        x_for_qkv = x.flatten(0, 1)
        y_for_qkv = y.flatten(0, 1)

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, N).contiguous()
        q_conv_out = self.q_lif(q_conv_out.flatten(0, 1)).reshape(T, B, C, N).transpose(-2, -1)
        q = q_conv_out.reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        k_conv_out = self.k_conv(y_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, N).contiguous()
        k_conv_out = self.k_lif(k_conv_out.flatten(0, 1)).reshape(T, B, C, N).transpose(-2, -1)
        k = k_conv_out.reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v_conv_out = self.v_conv(y_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T, B, C, N).contiguous()
        v_conv_out = self.v_lif(v_conv_out.flatten(0, 1)).reshape(T, B, C, N).transpose(-2, -1)
        v = v_conv_out.reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        # SSA
        attn = (q @ k.transpose(-2, -1))

        x = (attn @ v) * self.scale

        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x.flatten(0, 1))
        x = self.proj_lif(self.proj_bn(self.proj_conv(x))).reshape(T, B, C, N)

        return x
