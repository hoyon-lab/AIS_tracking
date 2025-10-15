import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.lstm_models import CNNLSTMTrackNet, DilatedResidualTCN
def count_parameters(model):
    """모델의 파라미터 수를 계산"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params
def main():
    print("🔍 모델 파라미터 수 비교 분석")
    print("=" * 50)
    input_dim = 5
    hidden_dim = 128
    num_layers = 2
    num_classes = 10
    print("\n📊 CNN-LSTM 모델")
    print("-" * 30)
    cnn_lstm = CNNLSTMTrackNet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes
    )
    total_cnn, trainable_cnn = count_parameters(cnn_lstm)
    print(f"총 파라미터: {total_cnn:,}")
    print(f"학습 가능 파라미터: {trainable_cnn:,}")
    print("\n📊 DRC-TCN 모델")
    print("-" * 30)
    drc_tcn = DilatedResidualTCN(
        input_dim=input_dim,
        num_channels=[48, 96, 192],
        kernel_size=5,
        dropout=0.15,
        num_classes=num_classes
    )
    total_tcn, trainable_tcn = count_parameters(drc_tcn)
    print(f"총 파라미터: {total_tcn:,}")
    print(f"학습 가능 파라미터: {trainable_tcn:,}")
    print("\n📈 파라미터 수 비교")
    print("-" * 30)
    param_ratio = total_tcn / total_cnn
    print(f"DRC-TCN / CNN-LSTM 비율: {param_ratio:.2f}x")
    if param_ratio > 1:
        print(f"DRC-TCN이 CNN-LSTM보다 {param_ratio:.1f}배 더 많은 파라미터")
    else:
        print(f"DRC-TCN이 CNN-LSTM보다 {1/param_ratio:.1f}배 적은 파라미터")
    print("\n🔬 상세 파라미터 분석")
    print("-" * 30)
    print("CNN-LSTM 구조:")
    print("  - Conv1d: 5×64×3 + 64 = 1,024")
    print("  - BatchNorm1d: 64×2 = 128")
    print("  - LSTM Layer 1: 4×(64×128 + 128×128 + 128) = 98,816")
    print("  - LSTM Layer 2: 4×(128×128 + 128×128 + 128) = 131,584")
    print("  - Classifier: 128×64 + 64 + 64×10 + 10 = 8,906")
    print(f"  - 계산값: 1,024 + 128 + 98,816 + 131,584 + 8,906 = {1024+128+98816+131584+8906:,}")
    print("\nDRC-TCN 구조:")
    print("  - Level 1 (Dilation=1): 5×48×5 + 48 + 48×48×5 + 48 = 12,048")
    print("  - Level 2 (Dilation=2): 48×96×5 + 96 + 96×96×5 + 96 = 46,176")
    print("  - Level 3 (Dilation=4): 96×192×5 + 192 + 192×192×5 + 192 = 184,320")
    print("  - Classifier: 192×96 + 96 + 96×10 + 10 = 18,442")
    print(f"  - 계산값: 12,048 + 46,176 + 184,320 + 18,442 = {12048+46176+184320+18442:,}")
    print("\n✅ 파라미터 수 계산 완료!")
if __name__ == "__main__":
    main()
