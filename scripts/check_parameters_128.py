import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.lstm_models import CNNLSTMTrackNet, DilatedResidualTCN
def count_params(model):
	return sum(p.numel() for p in model.parameters())
def main():
	input_dim = 128
	hidden_dim = 128
	num_layers = 2
	num_classes = 10
	cnn = CNNLSTMTrackNet(
		input_dim=input_dim,
		hidden_dim=hidden_dim,
		num_layers=num_layers,
		num_classes=num_classes
	)
	tcn = DilatedResidualTCN(
		input_dim=input_dim,
		num_channels=[48, 96, 192],
		kernel_size=5,
		dropout=0.15,
		num_classes=num_classes
	)
	cnn_params = count_params(cnn)
	tcn_params = count_params(tcn)
	print(f"CNN-LSTM params (input_dim=128): {cnn_params:,}")
	print(f"DRC-TCN params (input_dim=128): {tcn_params:,}")
	ratio = tcn_params / cnn_params if cnn_params else float('inf')
	print(f"Ratio (DRC/CNN): {ratio:.2f}x")
if __name__ == "__main__":
	main()
