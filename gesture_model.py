import torch.nn as nn
import torch.nn.functional as F

class GestureClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(GestureClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=34, out_channels=64, kernel_size=3)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3)
        self.fc = nn.Linear(128, num_classes)  # Adjust if needed

    def forward(self, x):
        x = F.relu(self.conv1(x))  # Output: (B, 64, L-2)
        x = F.relu(self.conv2(x))  # Output: (B, 128, L-4)
        x = F.adaptive_avg_pool1d(x, 1)  # shape: (B, 128, 1)
        x = x.squeeze(2)  # shape: (B, 128)
        x = self.fc(x)
        return x
