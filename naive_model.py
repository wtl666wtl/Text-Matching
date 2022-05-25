import torch
from torch import nn
import transformers as tfs


class naive_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.mp = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(16 * 15 * 15, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        #print(x.size())
        in_size = x.size(0)
        x = x.unsqueeze(1) # 128 * 1 * 64 * 64
        #print(x.size())
        out = self.relu(self.mp(self.conv1(x))) # 1 * 64 * 64 -> 6 * 32 * 32
        out = self.relu(self.mp(self.conv2(out))) # 6 * 32 * 32 -> 16 * 15 * 15
        out = out.view(in_size, -1)
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out
