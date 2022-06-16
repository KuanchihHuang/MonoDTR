import torch.nn as nn
import torch
import torch
from torch.nn import Module, Dropout

class DepthAwareTransformer(nn.Module):
    def __init__(self, output_channel_num):
        super().__init__()
        self.output_channel_num = output_channel_num
        self.encoder = TransEncoderLayer(self.output_channel_num)
        self.decoder = TransDecoderLayer(self.output_channel_num)

    def forward(self, depth_feat, context_feat, depth_pos=None):
        
        # context_feat: N, L, C
        # depth_feat: N, L, C
        # depth_pos: N, L, C

        context_feat = context_feat + depth_pos
        context_feat = self.encoder(context_feat)
        integrated_feat = self.decoder(depth_feat, context_feat)
        return integrated_feat


class TransEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=2048,
                 nhead=8,
                 attention='linear'):
        super(TransEncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention() 
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and drop_path
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop_path = nn.Identity()

    def forward(self, x):
        
        bs = x.size(0)
        query, key, value = x, x, x

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        
        x = x + self.drop_path(self.norm1(message))
        x = x + self.drop_path(self.norm2(self.mlp(x)))
        return x

def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1

class LinearAttention(Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values):
        """ Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, L, H, D]
            values: [N, L, H, D]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (L,D)' @ L,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()


class TransDecoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead=8,
                 attention='linear'):
        super(TransDecoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj0 = nn.Linear(d_model, d_model, bias=False)
        self.k_proj0 = nn.Linear(d_model, d_model, bias=False)
        self.v_proj0 = nn.Linear(d_model, d_model, bias=False)
        self.attention0 = LinearAttention() 
        self.merge0 = nn.Linear(d_model, d_model, bias=False)

        # multi-head attention
        self.q_proj1 = nn.Linear(d_model, d_model, bias=False)
        self.k_proj1 = nn.Linear(d_model, d_model, bias=False)
        self.v_proj1 = nn.Linear(d_model, d_model, bias=False)
        self.attention1 = LinearAttention()
        self.merge1 = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm0 = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop_path = nn.Identity()

    def forward(self, x, source):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, L, C]
        """
        
        bs = x.size(0)

        #Self-Attentiion for x (depth_feat)
        query, key, value = x, x, x

        query = self.q_proj0(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj0(key).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        value = self.v_proj0(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention0(query, key, value)  # [N, L, (H, D)]
        message = self.merge0(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        
        x = x + self.drop_path(self.norm0(message))

        #Cross-Attentiion for x and source (depth_feat & context_feat)
        query, key, value = x, source, source

        query = self.q_proj1(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj1(key).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        value = self.v_proj1(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention1(query, key, value)  # [N, L, (H, D)]
        message = self.merge1(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        
        x = x + self.drop_path(self.norm1(message))
        x = x + self.drop_path(self.norm2(self.mlp(x)))
        
        return x

