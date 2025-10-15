import torch
import torch.nn as nn
class CNNLSTMTrackNet(nn.Module):
    def __init__(self, input_dim=5, cnn_dim=64, lstm_dim=128, emb_dim=64, num_classes=None, mode='classification'):
        super().__init__()
        self.mode = mode
        self.conv1 = nn.Conv1d(input_dim, cnn_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(cnn_dim, lstm_dim, batch_first=True)
        self.embedding = nn.Linear(lstm_dim, emb_dim)
        if self.mode == 'classification':
            assert num_classes is not None, "num_classes required for classification"
            self.classifier = nn.Linear(emb_dim, num_classes)
    def forward(self, x):
        x = x.transpose(1, 2)  
        x = self.relu(self.conv1(x))
        x = x.transpose(1, 2)  
        _, (h_n, _) = self.lstm(x)
        z = self.embedding(h_n[-1])  
        if self.mode == 'classification':
            return self.classifier(z), z
        else:
            return z
