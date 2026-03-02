
import torch
import torch.nn as nn
from Transformer import *

class iDNA_SMP(nn.Module):
    def __init__(self, channels=32, r=4):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=4,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 32, 5, 1, 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        #
        self.conv3 = nn.Sequential(
            nn.Conv1d(32, 32, 5, 1, 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.posweight = nn.Sequential(
            nn.Conv1d(4, 32, 5, 1, 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.AdaptiveAvgPool1d(41)  
        )
        self.trans = Transformer(1, 41, 4, 100, 400, 64)　
        self.block1 = nn.Sequential(nn.Linear(1024, 256),
                                    nn.BatchNorm1d(256),
                                    nn.LeakyReLU(),
                                    )

        self.block2 = nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1),
        )

    def forward(self, x,xpos):
        x = x.permute(0, 2, 1)
        batch_size = x.size(0)
        xpos = xpos.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch, 4, 41]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        xpos=self.posweight(xpos)
        fused = torch.cat([x, xpos], dim=1)  # [batch, 64, 41]
        # fused = x+xpos
        x = self.trans(fused)
        self.attn=x
        x = x.view(x.size(0), -1)
        out = self.out(x)
        return self.block1(out)

    def trainModel(self, x,xpos):
        # with torch.no_grad():
        output = self.forward(x,xpos)
        return self.block2(output)






