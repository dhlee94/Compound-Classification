import torch
import torch.nn as nn
from torch.autograd import Variable
from timm.models.layers import trunc_normal_
class LSTM(nn.Module):
    def __init__(self, in_channels, embedding_channels, hidden_channels, classes, num_layers, dim_reduction=False):
        super(LSTM, self).__init__()
        tmp_channels = embedding_channels if dim_reduction else in_channels
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.dimension_layer = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=embedding_channels) if dim_reduction else nn.Identity(),
            nn.LayerNorm(tmp_channels))
        self.lstm_layer = nn.LSTM(input_size=tmp_channels, hidden_size=hidden_channels, num_layers=num_layers, bias=False)
        self.lstm_out_layer = nn.Sequential(
            nn.LayerNorm(hidden_channels),
            nn.Linear(in_features=hidden_channels, out_features=classes, bias=False)
        )
        self.dropout_layer = nn.Dropout(p=0.2)
        self.apply(self._init_weights)

    def forward(self, x, device):
        B, N = x.shape
        h_0 = Variable(torch.zeros(self.num_layers, B, self.hidden_channels).to(device))
        c_0 = Variable(torch.zeros(self.num_layers, B, self.hidden_channels).to(device))
        x = self.dimension_layer(x).unsqueeze(0)
        _, (ht, _) = self.lstm_layer(x, (h_0, c_0))
        x = self.dropout_layer(ht[-1])
        x = self.lstm_out_layer(x)
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
    model = LSTM(in_channels=2048, embedding_channels=1024, hidden_channels=512, classes=1, num_layers=3, dim_reduction=True)
    result = model(input, device=torch.device("cpu"))
    print(result.shape)