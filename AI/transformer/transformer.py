import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
import numpy as np
from random  import random
import torch.nn.functional as F
import math
    
class FeedForward(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, act_layer=nn.GELU):
        super(FeedForward, self).__init__()
        hidden_channels =  hidden_channels if hidden_channels else in_channels//4
        self.feed_layer = nn.Sequential(
                    nn.LayerNorm(in_channels, eps=1e-5),
                    nn.Linear(in_channels, hidden_channels, bias=False),
                    act_layer(),
                    nn.Linear(hidden_channels, in_channels, bias=False)
        )
    def forward(self, x):
        return self.feed_layer(x)

class attention(nn.Module):
    def __init__(self, in_channels):
        super(attention, self).__init__()
        self.in_channels = in_channels
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x, e=1e-12):
        q, k, v = x.split(split_size=self.in_channels,dim=1)
        k_t = k.transpose(1, 2).contiguous()
        score = (q@k_t) / math.sqrt(k.size(-1))
        score = self.softmax(score)
        v = score @ v
        return v
    
class transformer_block(nn.Module):
    def __init__(self, in_channels, hidden_channels, act_layer):
        super(transformer_block, self).__init__()
        self.FeedForward = FeedForward(in_channels=in_channels, hidden_channels=hidden_channels, act_layer=act_layer)
        self.linear = nn.Linear(in_features=in_channels, out_features=in_channels*3, bias=False)
        self.attention = attention(in_channels=in_channels)
    def forward(self, x):
        x = self.linear(x).transpose(1, 2).contiguous()
        x = self.attention(x).transpose(1, 2).contiguous()
        x = x + self.FeedForward(x)
        return x
    
class Transformer(nn.Module):
    def __init__(self, feature_dimensions=2048, dim_reduction=4, num_head=2, depth=4, act_layer=nn.GELU):
        super(Transformer, self).__init__()
        self.num_head = num_head
        self.dim_reduction = nn.Sequential(
            nn.Linear(in_features=feature_dimensions, out_features=feature_dimensions*dim_reduction, bias=False),
            nn.LayerNorm(feature_dimensions*dim_reduction, eps=1e-5)
        )
        block = [transformer_block(in_channels=num_head, hidden_channels=None, act_layer=act_layer) for _ in range(depth)]
        self.transformer = nn.Sequential(*block)
        #Output
        self.pooling = nn.AdaptiveAvgPool1d(1)
        in_channels = feature_dimensions*dim_reduction//num_head
        self.final_FeedForward = FeedForward(in_channels, hidden_channels=None, act_layer=act_layer)
        self.classfier = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, 1)
        )
        #Weight Initialize
        self.apply(self._init_weights)

    def forward(self, x):
        B, _ = x.shape
        x = self.dim_reduction(x)
        x = x.view(B, -1, self.num_head)
        x = self.transformer(x)
        x = self.pooling(x).view(B, -1).contiguous()
        x = self.final_FeedForward(x)
        x = self.classfier(x)
        return x
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


if __name__ == '__main__':
    input = torch.randn(16, 2048)
    model = Transformer(feature_dimensions=2048, dim_reduction=2, num_head=16, depth=4, act_layer=nn.GELU)
    model = model.to(torch.device("cuda"))
    input = input.to(torch.device("cuda"))
    result = model(input)
    print(result.shape)
    print(result)