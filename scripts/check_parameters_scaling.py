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
def analyze_parameter_scaling():
    print("🔍 입력 사이즈 변화에 따른 파라미터 수 분석")
    print("=" * 60)
    hidden_dim = 128
    num_layers = 2
    num_classes = 10
    input_dims = [5, 10, 20, 50, 100]
    print("\n📊 CNN-LSTM 모델 파라미터 변화")
    print("-" * 50)
    print("Input Dim | Total Params | Change | Ratio")
    print("-" * 50)
    cnn_lstm_params = []
    for input_dim in input_dims:
        cnn_lstm = CNNLSTMTrackNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes
        )
        total_params, _ = count_parameters(cnn_lstm)
        cnn_lstm_params.append(total_params)
        if input_dim == 5:
            change = 0
            ratio = 1.0
        else:
            change = total_params - cnn_lstm_params[0]
            ratio = total_params / cnn_lstm_params[0]
        print(f"{input_dim:9} | {total_params:12,} | {change:+7,} | {ratio:5.2f}x")
    print("\n📊 DRC-TCN 모델 파라미터 변화")
    print("-" * 50)
    print("Input Dim | Total Params | Change | Ratio")
    print("-" * 50)
    drc_tcn_params = []
    for input_dim in input_dims:
        drc_tcn = DilatedResidualTCN(
            input_dim=input_dim,
            num_channels=[48, 96, 192],
            kernel_size=5,
            dropout=0.15,
            num_classes=num_classes
        )
        total_params, _ = count_parameters(drc_tcn)
        drc_tcn_params.append(total_params)
        if input_dim == 5:
            change = 0
            ratio = 1.0
        else:
            change = total_params - drc_tcn_params[0]
            ratio = total_params / drc_tcn_params[0]
        print(f"{input_dim:9} | {total_params:12,} | {change:+7,} | {ratio:5.2f}x")
    print("\n🔬 상세 파라미터 변화 분석")
    print("-" * 50)
    print("CNN-LSTM 파라미터 변화 원인:")
    print("  - Conv1d 레이어만 입력 차원에 영향받음")
    print("  - LSTM과 분류기는 입력 차원과 무관")
    print("  - Conv1d: input_dim × 64 × 3 + 64")
    for i, input_dim in enumerate(input_dims):
        if i == 0:
            continue
        conv_params = input_dim * 64 * 3 + 64
        base_conv_params = input_dims[0] * 64 * 3 + 64
        conv_change = conv_params - base_conv_params
        print(f"  - Input {input_dims[0]}→{input_dim}: Conv1d +{conv_change:,} 파라미터")
    print("\nDRC-TCN 파라미터 변화 원인:")
    print("  - Level 1의 첫 번째 Conv1d만 입력 차원에 영향받음")
    print("  - 나머지 레이어들은 이전 레이어의 출력에 의존")
    print("  - Level 1 Conv1: input_dim × 48 × 5 + 48")
    for i, input_dim in enumerate(input_dims):
        if i == 0:
            continue
        level1_conv_params = input_dim * 48 * 5 + 48
        base_level1_conv_params = input_dims[0] * 48 * 5 + 48
        level1_change = level1_conv_params - base_level1_conv_params
        print(f"  - Input {input_dims[0]}→{input_dim}: Level 1 Conv1d +{level1_change:,} 파라미터")
    print("\n📈 모델 효율성 비교")
    print("-" * 50)
    for i, input_dim in enumerate(input_dims):
        if i == 0:
            continue
        cnn_ratio = cnn_lstm_params[i] / cnn_lstm_params[0]
        tcn_ratio = drc_tcn_params[i] / drc_tcn_params[0]
        print(f"Input {input_dims[0]}→{input_dim}:")
        print(f"  - CNN-LSTM: {cnn_ratio:.2f}x 증가")
        print(f"  - DRC-TCN: {tcn_ratio:.2f}x 증가")
        if cnn_ratio < tcn_ratio:
            print(f"  - CNN-LSTM이 더 효율적 (파라미터 증가율 낮음)")
        else:
            print(f"  - DRC-TCN이 더 효율적 (파라미터 증가율 낮음)")
        print()
    print("✅ 파라미터 스케일링 분석 완료!")
def main():
    analyze_parameter_scaling()
if __name__ == "__main__":
    main()
