import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.lstm_models import CNNLSTMTrackNet
def count_parameters(model):
    """모델의 총 파라미터 수와 학습 가능한 파라미터 수를 계산"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params
def calculate_cnn_lstm_params():
    """표에서 사용된 설정과 유사한 조건으로 CNN-LSTM 파라미터 수 계산"""
    print("=== CNN-LSTM 파라미터 수 계산 ===\n")
    configs = [
        {"input_dim": 5, "hidden_dim": 128, "num_layers": 2, "num_classes": 10, "desc": "기본 설정 (input_dim=5)"},
        {"input_dim": 10, "hidden_dim": 128, "num_layers": 2, "num_classes": 10, "desc": "중간 설정 (input_dim=10)"},
        {"input_dim": 20, "hidden_dim": 128, "num_layers": 2, "num_classes": 10, "desc": "표준 설정 (input_dim=20)"},
        {"input_dim": 50, "hidden_dim": 128, "num_layers": 2, "num_classes": 10, "desc": "높은 차원 (input_dim=50)"},
        {"input_dim": 128, "hidden_dim": 128, "num_layers": 2, "num_classes": 10, "desc": "최고 차원 (input_dim=128)"},
        {"input_dim": 256, "hidden_dim": 128, "num_layers": 2, "num_classes": 10, "desc": "초고 차원 (input_dim=256)"}
    ]
    print(f"{'설정':<25} | {'Input Dim':<10} | {'Hidden Dim':<10} | {'Layers':<7} | {'Parameters':<12}")
    print("-" * 80)
    for config in configs:
        model = CNNLSTMTrackNet(
            input_dim=config["input_dim"],
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            num_classes=config["num_classes"]
        )
        total_params, trainable_params = count_parameters(model)
        print(f"{config['desc']:<25} | {config['input_dim']:<10} | {config['hidden_dim']:<10} | {config['num_layers']:<7} | {trainable_params:<12,}")
    print("\n=== 상세 파라미터 분석 (input_dim=256 기준) ===")
    model = CNNLSTMTrackNet(input_dim=256, hidden_dim=128, num_layers=2, num_classes=10)
    print(f"\n모델 구조:")
    print(f"- CNN Conv1d: {model.conv1.in_channels} → {model.conv1.out_channels}, kernel_size={model.conv1.kernel_size}")
    print(f"- LSTM: {model.lstm.input_size} → {model.lstm.hidden_size}, layers={model.lstm.num_layers}")
    print(f"- Classifier: {model.classifier[0].in_features} → {model.classifier[0].out_features} → {model.classifier[-1].out_features}")
    conv_params = sum(p.numel() for p in model.conv1.parameters())
    bn_params = sum(p.numel() for p in model.bn1.parameters())
    lstm_params = sum(p.numel() for p in model.lstm.parameters())
    classifier_params = sum(p.numel() for p in model.classifier.parameters())
    print(f"\n레이어별 파라미터 수:")
    print(f"- Conv1d: {conv_params:,}")
    print(f"- BatchNorm1d: {bn_params:,}")
    print(f"- LSTM: {lstm_params:,}")
    print(f"- Classifier: {classifier_params:,}")
    total_params, trainable_params = count_parameters(model)
    print(f"\n총 파라미터 수: {total_params:,}")
    print(f"학습 가능한 파라미터 수: {trainable_params:,}")
    print(f"\n=== 표의 다른 모델들과 비교 ===")
    print(f"TCN: 403,834 파라미터")
    print(f"DRC-TCN: 405,178 파라미터")
    print(f"CNN-LSTM (input_dim=256): {trainable_params:,} 파라미터")
    if trainable_params < 403834:
        print(f"CNN-LSTM이 TCN보다 {403834 - trainable_params:,} 파라미터 적음")
    else:
        print(f"CNN-LSTM이 TCN보다 {trainable_params - 403834:,} 파라미터 많음")
    print(f"\n=== 파라미터 증가율 분석 ===")
    base_model = CNNLSTMTrackNet(input_dim=20, hidden_dim=128, num_layers=2, num_classes=10)
    base_params = sum(p.numel() for p in base_model.parameters())
    increase_ratio = trainable_params / base_params
    print(f"input_dim=20 → input_dim=256: {increase_ratio:.2f}x 증가")
    print(f"파라미터 증가량: {trainable_params - base_params:,}")
if __name__ == "__main__":
    calculate_cnn_lstm_params()
