import torch
import torch.nn as nn
import torch.nn.functional as F

class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        self.layer1 = nn.Linear(512, 32)
        self.layer2 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        return F.sigmoid(x)
        
class Net2(nn.Module): #87% with k = [2, 4, 8, 16, 32, 64], k_amount = 64
    def __init__(self):
        super(Net2, self).__init__()
        #dropout=.3, channels=16, kernels=[2, 4, 8, 16, 32, 64], classes=2
        dropout=.3
        channels=16
        kernels=[2, 4, 8, 16, 32, 64]
        classes=2
        
        self.conv1 = nn.ModuleList([nn.Conv1d(in_channels=1, out_channels=channels, kernel_size=k) for k in kernels])
        self.dropout = nn.Dropout(dropout)
        #self.norm = nn.BatchNorm1d()

        # self.conv1 = nn.Sequential(
        #     nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=750),
        #     nn.BatchNorm1d(256),
        #     nn.Dropout(dropout),
        # )
        self.lin = nn.Sequential(
            nn.Linear(channels * len(kernels), 32),
            nn.Linear(32,classes),
        )
    
    def forward(self, batch):
        x = torch.reshape(batch, (len(batch), 1, 512))
        x = [F.relu(conv(x)) for conv in self.conv1]
        x = [F.max_pool1d(l, l.size(2)) for l in x]
        x = torch.cat(x, dim=1).squeeze()
        x = self.dropout(x)
        #out = self.drop(out)
        #logits = self.fc(out.squeeze(2))
        logits = self.lin(x)
        return logits