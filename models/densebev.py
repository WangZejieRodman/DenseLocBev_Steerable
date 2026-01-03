import torch
import torch.nn as nn
import numpy as np

try:
    import e2cnn
    from e2cnn import gspaces
    from e2cnn import nn as enn
except ImportError:
    raise ImportError("请先安装 e2cnn 库: pip install e2cnn")


class InvariantMagnitude(nn.Module):
    """
    不变性投影层：将几何特征转换为旋转不变的标量特征。
    """

    def __init__(self, in_type):
        super(InvariantMagnitude, self).__init__()
        self.in_type = in_type
        self.out_channels = len(in_type.representations)

    def forward(self, x):
        assert isinstance(x, enn.GeometricTensor)
        assert x.type == self.in_type
        tensor = x.tensor
        b, c, h, w = tensor.shape

        output_slices = []
        current_idx = 0
        for i, r in enumerate(self.in_type.representations):
            dim = r.size
            slice_x = tensor[:, current_idx: current_idx + dim, :, :]
            if dim == 1:
                output_slices.append(slice_x)
            else:
                norm = torch.sqrt(torch.sum(slice_x ** 2, dim=1, keepdim=True) + 1e-8)
                output_slices.append(norm)
            current_idx += dim
        return torch.cat(output_slices, dim=1)


class DenseSteerableBottleneck(nn.Module):
    def __init__(self, in_type, out_type, kernel_size, stride=1):
        super(DenseSteerableBottleneck, self).__init__()
        mid_type = out_type
        padding = kernel_size // 2

        self.conv1 = enn.SequentialModule(
            enn.R2Conv(in_type, mid_type, kernel_size=1, bias=False),
            enn.GNormBatchNorm(mid_type),
            enn.NormNonLinearity(mid_type, function='n_relu')
        )
        self.conv2 = enn.SequentialModule(
            enn.R2Conv(mid_type, mid_type, kernel_size=kernel_size,
                       padding=padding, stride=stride, bias=False),
            enn.GNormBatchNorm(mid_type),
            enn.NormNonLinearity(mid_type, function='n_relu')
        )
        self.conv3 = enn.SequentialModule(
            enn.R2Conv(mid_type, out_type, kernel_size=1, bias=False),
            enn.GNormBatchNorm(out_type)
        )

        if in_type != out_type or stride != 1:
            self.shortcut = enn.SequentialModule(
                enn.R2Conv(in_type, out_type, kernel_size=1, stride=stride, bias=False),
                enn.GNormBatchNorm(out_type)
            )
        else:
            self.shortcut = nn.Identity()
        self.relu = enn.NormNonLinearity(out_type, function='n_relu')

    def forward(self, x):
        identity = x
        if not isinstance(self.shortcut, nn.Identity):
            identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        return self.relu(out + identity)


class DenseBEVBackbone(nn.Module):
    def __init__(self, in_channels=32, out_channels=256):
        super(DenseBEVBackbone, self).__init__()
        print("Initializing Steerable DenseBEV Backbone (SO(2) Equivariant) - WIDE & BALANCED Ver.")

        # 1. 定义几何空间
        self.r2_act = gspaces.Rot2dOnR2(N=-1, maximum_frequency=8)

        # 2. 输入类型 (32 标量)
        self.in_type = enn.FieldType(self.r2_act, in_channels * [self.r2_act.trivial_repr])

        # 3. 特征配方 (修改点 A: 增加标量比例到 50%)
        def get_mixed_type(total_channels):
            # 50% 标量 (快速收敛)
            n_scalar = int(total_channels * 0.5)
            remaining = total_channels - n_scalar
            # 剩余的一半分配给向量(30%)和张量(20%)
            # 注意: 向量/张量占 2 个通道
            n_vector = int((remaining * 0.6) / 2)
            n_tensor = int((remaining * 0.4) / 2)

            types = [self.r2_act.trivial_repr] * n_scalar + \
                    [self.r2_act.irrep(1)] * n_vector + \
                    [self.r2_act.irrep(2)] * n_tensor
            return enn.FieldType(self.r2_act, types)

        # 4. 定义各阶段宽度 (修改点 B: 通道数增加 50%~100% 以增加容量)
        # 原始: 64 -> 128 -> 256
        # 现在: 96 -> 192 -> 384 (增加容量，弥补参数共享带来的参数量减少)
        self.type_c1 = get_mixed_type(96)
        self.type_c2 = get_mixed_type(192)
        self.type_c3 = get_mixed_type(384)
        # 输出保持大容量，经过不变性投影后再降维
        self.type_c4 = get_mixed_type(384)

        print(f"  Input Type: {self.in_type.size}")
        print(f"  Stage 1 Type: {self.type_c1.size} (Increased)")
        print(f"  Stage 2 Type: {self.type_c2.size} (Increased)")
        print(f"  Stage 3 Type: {self.type_c3.size} (Increased)")

        # 5. 网络层
        self.block1 = nn.Sequential(
            DenseSteerableBottleneck(self.in_type, self.type_c1, kernel_size=11),
            enn.PointwiseAvgPool(self.type_c1, kernel_size=3, stride=2, padding=1),
            DenseSteerableBottleneck(self.type_c1, self.type_c1, kernel_size=11)
        )
        self.block2 = nn.Sequential(
            DenseSteerableBottleneck(self.type_c1, self.type_c2, kernel_size=7),
            enn.PointwiseAvgPool(self.type_c2, kernel_size=3, stride=2, padding=1),
            DenseSteerableBottleneck(self.type_c2, self.type_c2, kernel_size=7)
        )
        self.block3 = nn.Sequential(
            DenseSteerableBottleneck(self.type_c2, self.type_c3, kernel_size=5),
            enn.PointwiseAvgPool(self.type_c3, kernel_size=3, stride=2, padding=1),
            DenseSteerableBottleneck(self.type_c3, self.type_c3, kernel_size=5)
        )
        self.block4 = enn.R2Conv(self.type_c3, self.type_c4, kernel_size=3, padding=1)

        # 6. 不变性投影 + 最终适配
        self.invariant_map = InvariantMagnitude(self.type_c4)
        inv_channels = self.invariant_map.out_channels

        self.final_projection = nn.Sequential(
            nn.Conv2d(inv_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        print(f"  Invariant Projection: {self.type_c4.size} geom -> {inv_channels} inv scalars")

    def forward(self, x):
        x = enn.GeometricTensor(x, self.in_type)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.invariant_map(x)
        x = self.final_projection(x)
        return x