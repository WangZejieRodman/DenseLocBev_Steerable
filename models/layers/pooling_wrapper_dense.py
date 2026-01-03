import torch.nn as nn
from models.layers.pooling_dense import MAC, SPoC, GeM, NetVLADWrapper


class PoolingWrapper(nn.Module):
    """Dense版本的PoolingWrapper"""
    def __init__(self, pool_method, in_dim, output_dim):
        super().__init__()

        self.pool_method = pool_method
        self.in_dim = in_dim
        self.output_dim = output_dim

        if pool_method == 'MAC':
            assert in_dim == output_dim
            self.pooling = MAC(input_dim=in_dim)
        elif pool_method == 'SPoC':
            assert in_dim == output_dim
            self.pooling = SPoC(input_dim=in_dim)
        elif pool_method == 'GeM':
            assert in_dim == output_dim
            self.pooling = GeM(input_dim=in_dim)
        elif self.pool_method == 'netvlad':
            self.pooling = NetVLADWrapper(feature_size=in_dim, output_dim=output_dim, gating=False)
        elif self.pool_method == 'netvladgc':
            self.pooling = NetVLADWrapper(feature_size=in_dim, output_dim=output_dim, gating=True)
        else:
            raise NotImplementedError('Unknown pooling method: {}'.format(pool_method))

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) dense tensor
        Returns:
            (B, output_dim) tensor
        """
        return self.pooling(x)