import warnings
# 过滤 e2cnn 库的弃用警告
warnings.filterwarnings("ignore", message=".*masked_fill_.*torch.uint8.*")
warnings.filterwarnings("ignore", message=".*indexing with dtype torch.uint8.*")

import torch
import numpy as np
from models.densebev import DenseBEVBackbone
from models.layers.pooling_wrapper_dense import PoolingWrapper
from models.denseloc import DenseLoc


def test_rotation_invariance():
    print("正在验证模型的旋转不变性...")

    # 1. 实例化模型 (使用默认参数)
    backbone = DenseBEVBackbone(in_channels=32, out_channels=256)
    # 使用 GeM 池化 (无参数需要学习，适合测试)
    pooling = PoolingWrapper(pool_method='GeM', in_dim=256, output_dim=256)
    model = DenseLoc(backbone, pooling)
    model.eval()

    # 2. 创建一个伪造的 BEV 输入 (Batch=2, Channels=32, H=64, W=64)
    # 使用随机数模拟
    x = torch.randn(2, 32, 64, 64)

    # 3. 创建旋转后的输入 (逆时针旋转 90 度)
    # 注意: e2cnn 是连续等变的，这里用 90 度离散旋转做简单验证
    x_rot = torch.rot90(x, k=1, dims=[2, 3])

    with torch.no_grad():
        # 4. 分别计算特征
        out = model({'features': x})['global']  # (B, 256)
        out_rot = model({'features': x_rot})['global']  # (B, 256)

    # 5. 比较差异
    # 计算两个特征向量的余弦相似度或欧氏距离
    diff = torch.norm(out - out_rot)
    similarity = torch.nn.functional.cosine_similarity(out, out_rot).mean()

    print(f"\n原始输出范数: {out.norm().item():.4f}")
    print(f"旋转输出范数: {out_rot.norm().item():.4f}")
    print(f"特征差异 (Euclidean Distance): {diff.item():.6f}")
    print(f"余弦相似度 (Cosine Similarity): {similarity.item():.6f}")

    if similarity > 0.99:
        print("\n✅ 验证成功！模型具备旋转不变性。")
    else:
        print("\n❌ 验证失败。特征随旋转发生了显著变化，请检查代码。")


if __name__ == "__main__":
    test_rotation_invariance()