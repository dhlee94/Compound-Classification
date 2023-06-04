import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

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
        x = x + self.feed_layer(x)
        return x
    
class Linear(nn.Module):
    def __init__(self, in_channels, embedding_channels, hidden_channels, classes, depth, dim_reduction=False):
        super(Linear, self).__init__()
        tmp_channels = embedding_channels if dim_reduction else in_channels
        self.hidden_channels = hidden_channels
        self.dimension_layer = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=tmp_channels) if dim_reduction else nn.Identity(),
            nn.LayerNorm(tmp_channels))
        feed_forward = [FeedForward(in_channels=tmp_channels, hidden_channels=hidden_channels, act_layer=nn.GELU) for _ in range(depth)]
        self.feed_forward = nn.Sequential(*feed_forward)
        self.dropout_layer = nn.Dropout(p=0.2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(tmp_channels),
            nn.Linear(in_features=tmp_channels, out_features=classes)
        )
        self.apply(self._init_weights)
        
    def forward(self, x):
        x = self.dimension_layer(x)
        x = self.feed_forward(x)
        x = self.classifier(self.dropout_layer(x))
        return x
      
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
class BasicLinear(nn.Module):
    def __init__(self, in_channels, hidden_channels, classes):
        super(BasicLinear, self).__init__()
        self.layer1 = nn.Linear(in_features=in_channels, out_features=hidden_channels, bias=False)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.layer2 = nn.Linear(in_features=hidden_channels, out_features=hidden_channels, bias=False)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.classifier = nn.Linear(in_features=hidden_channels, out_features=classes, bias=False)
        self.apply(self._init_weights)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.norm1(x)
        x = x + self.layer2(x)
        x = self.norm2(x)
        x = self.classifier(x)
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
    model = Linear(in_channels=2048, embedding_channels=1024, hidden_channels=None, classes=1, depth=4, dim_reduction=True)
    result = model(input)
    print(result.shape)