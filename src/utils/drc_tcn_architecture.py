import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
def create_drc_tcn_architecture():
    """DRC-TCN 모델의 논문용 아키텍처 다이어그램 생성"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    colors = {
        'input': '#E8F4FD',
        'conv': '#FFE6E6',
        'bn': '#E6FFE6',
        'relu': '#FFF2E6',
        'dropout': '#F0E6FF',
        'chomp': '#FFE6F0',
        'residual': '#FFFFE6',
        'classifier': '#E6F0FF',
        'output': '#F0F0F0'
    }
    ax.text(8, 11.5, 'Dilated Residual Temporal Convolutional Network (DRC-TCN)', 
            fontsize=16, fontweight='bold', ha='center')
    ax.text(8, 11.2, 'AIS Trajectory Classification Architecture', 
            fontsize=12, ha='center', style='italic')
    input_box = FancyBboxPatch((0.5, 9.5), 2, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['input'], 
                               edgecolor='black', linewidth=1.5)
    ax.add_patch(input_box)
    ax.text(1.5, 10, 'Input\n(5×50)', ha='center', va='center', fontweight='bold')
    ax.arrow(2.5, 10, 0.5, 0, head_width=0.2, head_length=0.1, fc='black', ec='black')
    transpose_box = FancyBboxPatch((3, 9.5), 1.5, 1, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor=colors['input'], 
                                  edgecolor='black', linewidth=1)
    ax.add_patch(transpose_box)
    ax.text(3.75, 10, 'Transpose\n(50×5)', ha='center', va='center', fontsize=9)
    ax.arrow(4.5, 10, 0.5, 0, head_width=0.2, head_length=0.1, fc='black', ec='black')
    tcn_start_x = 5
    tcn_y = 9.5
    level1_box = FancyBboxPatch((tcn_start_x, tcn_y), 2.5, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['conv'], 
                                edgecolor='black', linewidth=1.5)
    ax.add_patch(level1_box)
    ax.text(tcn_start_x + 1.25, tcn_y + 0.5, 'Level 1\nDilation=1\nChannels=48', 
            ha='center', va='center', fontweight='bold')
    level2_box = FancyBboxPatch((tcn_start_x + 3, tcn_y), 2.5, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['conv'], 
                                edgecolor='black', linewidth=1.5)
    ax.add_patch(level2_box)
    ax.text(tcn_start_x + 4.25, tcn_y + 0.5, 'Level 2\nDilation=2\nChannels=96', 
            ha='center', va='center', fontweight='bold')
    level3_box = FancyBboxPatch((tcn_start_x + 6, tcn_y), 2.5, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['conv'], 
                                edgecolor='black', linewidth=1.5)
    ax.add_patch(level3_box)
    ax.text(tcn_start_x + 7.25, tcn_y + 0.5, 'Level 3\nDilation=4\nChannels=192', 
            ha='center', va='center', fontweight='bold')
    ax.arrow(tcn_start_x + 2.5, tcn_y + 0.5, 0.5, 0, head_width=0.2, head_length=0.1, fc='black', ec='black')
    ax.arrow(tcn_start_x + 5.5, tcn_y + 0.5, 0.5, 0, head_width=0.2, head_length=0.1, fc='black', ec='black')
    ax.arrow(tcn_start_x + 8.5, tcn_y + 0.5, 0.5, 0, head_width=0.2, head_length=0.1, fc='black', ec='black')
    gap_box = FancyBboxPatch((tcn_start_x + 9, tcn_y), 2, 1, 
                             boxstyle="round,pad=0.1", 
                             facecolor=colors['output'], 
                             edgecolor='black', linewidth=1.5)
    ax.add_patch(gap_box)
    ax.text(tcn_start_x + 10, tcn_y + 0.5, 'Global\nAverage\nPooling', 
            ha='center', va='center', fontweight='bold')
    ax.arrow(tcn_start_x + 11, tcn_y + 0.5, 0.5, 0, head_width=0.2, head_length=0.1, fc='black', ec='black')
    classifier_box = FancyBboxPatch((tcn_start_x + 11.5, tcn_y), 2.5, 1, 
                                   boxstyle="round,pad=0.1", 
                                   facecolor=colors['classifier'], 
                                   edgecolor='black', linewidth=1.5)
    ax.add_patch(classifier_box)
    ax.text(tcn_start_x + 12.75, tcn_y + 0.5, 'Classifier\nLinear(192→96)\nLinear(96→Classes)', 
            ha='center', va='center', fontweight='bold')
    ax.arrow(tcn_start_x + 14, tcn_y + 0.5, 0.5, 0, head_width=0.2, head_length=0.1, fc='black', ec='black')
    output_box = FancyBboxPatch((tcn_start_x + 14.5, tcn_y), 1.5, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['output'], 
                               edgecolor='black', linewidth=1.5)
    ax.add_patch(output_box)
    ax.text(tcn_start_x + 15.25, tcn_y + 0.5, 'Output\n(Classes)', 
            ha='center', va='center', fontweight='bold')
    detail_y = 7.5
    ax.text(6.25, 8.5, 'DilatedResidualBlock 상세 구조', 
            fontsize=12, fontweight='bold', ha='center')
    detail_input = FancyBboxPatch((0.5, detail_y), 1.5, 0.8, 
                                 boxstyle="round,pad=0.05", 
                                 facecolor=colors['input'], 
                                 edgecolor='black', linewidth=1)
    ax.add_patch(detail_input)
    ax.text(1.25, detail_y + 0.4, 'Input', ha='center', va='center', fontsize=9)
    conv1_box = FancyBboxPatch((2.2, detail_y), 1.2, 0.8, 
                               boxstyle="round,pad=0.05", 
                               facecolor=colors['conv'], 
                               edgecolor='black', linewidth=1)
    ax.add_patch(conv1_box)
    ax.text(2.8, detail_y + 0.4, 'Conv1d', ha='center', va='center', fontsize=9)
    chomp1_box = FancyBboxPatch((3.6, detail_y), 1.2, 0.8, 
                                boxstyle="round,pad=0.05", 
                                facecolor=colors['chomp'], 
                                edgecolor='black', linewidth=1)
    ax.add_patch(chomp1_box)
    ax.text(4.2, detail_y + 0.4, 'Chomp1d', ha='center', va='center', fontsize=9)
    bn1_box = FancyBboxPatch((5, detail_y), 1.2, 0.8, 
                             boxstyle="round,pad=0.05", 
                             facecolor=colors['bn'], 
                             edgecolor='black', linewidth=1)
    ax.add_patch(bn1_box)
    ax.text(5.6, detail_y + 0.4, 'BatchNorm', ha='center', va='center', fontsize=9)
    relu1_box = FancyBboxPatch((6.4, detail_y), 1.2, 0.8, 
                               boxstyle="round,pad=0.05", 
                               facecolor=colors['relu'], 
                               edgecolor='black', linewidth=1)
    ax.add_patch(relu1_box)
    ax.text(7, detail_y + 0.4, 'ReLU', ha='center', va='center', fontsize=9)
    dropout1_box = FancyBboxPatch((7.8, detail_y), 1.2, 0.8, 
                                  boxstyle="round,pad=0.05", 
                                  facecolor=colors['dropout'], 
                                  edgecolor='black', linewidth=1)
    ax.add_patch(dropout1_box)
    ax.text(8.4, detail_y + 0.4, 'Dropout', ha='center', va='center', fontsize=9)
    conv2_box = FancyBboxPatch((9.2, detail_y), 1.2, 0.8, 
                               boxstyle="round,pad=0.05", 
                               facecolor=colors['conv'], 
                               edgecolor='black', linewidth=1)
    ax.add_patch(conv2_box)
    ax.text(9.8, detail_y + 0.4, 'Conv1d', ha='center', va='center', fontsize=9)
    chomp2_box = FancyBboxPatch((10.6, detail_y), 1.2, 0.8, 
                                boxstyle="round,pad=0.05", 
                                facecolor=colors['chomp'], 
                                edgecolor='black', linewidth=1)
    ax.add_patch(chomp2_box)
    ax.text(11.2, detail_y + 0.4, 'Chomp1d', ha='center', va='center', fontsize=9)
    bn2_box = FancyBboxPatch((12, detail_y), 1.2, 0.8, 
                             boxstyle="round,pad=0.05", 
                             facecolor=colors['bn'], 
                             edgecolor='black', linewidth=1)
    ax.add_patch(bn2_box)
    ax.text(12.6, detail_y + 0.4, 'BatchNorm', ha='center', va='center', fontsize=9)
    relu2_box = FancyBboxPatch((13.4, detail_y), 1.2, 0.8, 
                               boxstyle="round,pad=0.05", 
                               facecolor=colors['relu'], 
                               edgecolor='black', linewidth=1)
    ax.add_patch(relu2_box)
    ax.text(14, detail_y + 0.4, 'ReLU', ha='center', va='center', fontsize=9)
    dropout2_box = FancyBboxPatch((14.8, detail_y), 1.2, 0.8, 
                                  boxstyle="round,pad=0.05", 
                                  facecolor=colors['dropout'], 
                                  edgecolor='black', linewidth=1)
    ax.add_patch(dropout2_box)
    ax.text(15.4, detail_y + 0.4, 'Dropout', ha='center', va='center', fontsize=9)
    residual_box = FancyBboxPatch((8.5, detail_y - 1.5), 2, 0.8, 
                                 boxstyle="round,pad=0.05", 
                                 facecolor=colors['residual'], 
                                 edgecolor='black', linewidth=1.5)
    ax.add_patch(residual_box)
    ax.text(9.5, detail_y - 1.1, 'Residual\nConnection', ha='center', va='center', fontweight='bold')
    ax.arrow(1.25, detail_y, 0, -0.7, head_width=0.1, head_length=0.05, fc='black', ec='black')
    ax.arrow(9.5, detail_y - 0.7, 0, 0.7, head_width=0.1, head_length=0.05, fc='black', ec='black')
    for i in range(5):
        start_x = 0.5 + i * 1.4
        end_x = 2.2 + i * 1.4
        ax.arrow(start_x + 1.5, detail_y + 0.4, end_x - start_x - 0.3, 0, 
                head_width=0.1, head_length=0.05, fc='black', ec='black')
    ax.arrow(15.4, detail_y + 0.4, 0.3, 0, head_width=0.1, head_length=0.05, fc='black', ec='black')
    features_y = 5.5
    ax.text(8, features_y + 1.5, 'DRC-TCN 핵심 특징', 
            fontsize=14, fontweight='bold', ha='center')
    features = [
        '• Dilated Convolution: 수용 영역을 지수적으로 확장 (1→2→4)',
        '• Residual Connection: 그래디언트 소실 문제 해결 및 안정적 학습',
        '• Chomp1d: 패딩으로 인한 시퀀스 길이 증가 방지',
        '• BatchNorm: 학습 안정성 및 수렴 속도 향상',
        '• Multi-level Architecture: 다양한 시간 스케일의 패턴 학습'
    ]
    for i, feature in enumerate(features):
        ax.text(0.5, features_y - i * 0.4, feature, fontsize=10, ha='left')
    math_y = 2.5
    ax.text(8, math_y + 1, '수학적 표현', fontsize=14, fontweight='bold', ha='center')
    ax.text(0.5, math_y, 'Dilated Convolution:', fontsize=11, fontweight='bold')
    ax.text(0.5, math_y - 0.3, 'y[t] = Σ x[t - d·k] · w[k]', fontsize=11, fontfamily='monospace')
    ax.text(0.5, math_y - 0.6, 'where d = dilation factor, k = kernel position', fontsize=10, style='italic')
    ax.text(8, math_y, 'Residual Connection:', fontsize=11, fontweight='bold')
    ax.text(8, math_y - 0.3, 'F(x) = H(x) + x', fontsize=11, fontfamily='monospace')
    ax.text(8, math_y - 0.6, 'where H(x) = main branch, x = identity mapping', fontsize=10, style='italic')
    performance_y = 0.5
    ax.text(8, performance_y + 0.5, '성능 결과', fontsize=14, fontweight='bold', ha='center')
    perf_box = FancyBboxPatch((2, performance_y), 12, 0.8, 
                              boxstyle="round,pad=0.1", 
                              facecolor='#F0F8FF', 
                              edgecolor='blue', linewidth=2)
    ax.add_patch(perf_box)
    ax.text(8, performance_y + 0.4, 'DRC-TCN: 99% Accuracy | CNN-LSTM: 94% Accuracy | 향상도: +5%', 
            ha='center', va='center', fontsize=12, fontweight='bold', color='blue')
    plt.tight_layout()
    return fig
def create_dilated_conv_visualization():
    """Dilated Convolution의 시각적 이해를 위한 다이어그램"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.text(7, 7.5, 'Dilated Convolution 이해', fontsize=16, fontweight='bold', ha='center')
    ax.text(2, 6.5, 'Dilation=1 (일반 컨볼루션)', fontsize=12, fontweight='bold', ha='center')
    ax.text(2, 6.2, 'Kernel Size=3', fontsize=10, ha='center')
    for i in range(7):
        ax.add_patch(patches.Rectangle((0.5 + i*0.8, 5.5), 0.6, 0.6, facecolor='lightblue', edgecolor='black'))
        ax.text(0.8 + i*0.8, 5.8, f'x{i}', ha='center', va='center', fontsize=9)
    kernel1 = patches.Rectangle((1.3, 4.5), 2.4, 0.6, facecolor='red', edgecolor='black', alpha=0.7)
    ax.add_patch(kernel1)
    ax.text(2.5, 4.8, 'Kernel', ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    ax.text(7, 6.5, 'Dilation=2', fontsize=12, fontweight='bold', ha='center')
    ax.text(7, 6.2, 'Kernel Size=3', fontsize=10, ha='center')
    for i in range(7):
        ax.add_patch(patches.Rectangle((5.5 + i*0.8, 5.5), 0.6, 0.6, facecolor='lightblue', edgecolor='black'))
        ax.text(5.8 + i*0.8, 5.8, f'x{i}', ha='center', va='center', fontsize=9)
    kernel2 = patches.Rectangle((6.1, 4.5), 1.6, 0.6, facecolor='red', edgecolor='black', alpha=0.7)
    ax.add_patch(kernel2)
    ax.text(6.9, 4.8, 'K', ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    ax.text(12, 6.5, 'Dilation=4', fontsize=12, fontweight='bold', ha='center')
    ax.text(12, 6.2, 'Kernel Size=3', fontsize=10, ha='center')
    for i in range(7):
        ax.add_patch(patches.Rectangle((10.5 + i*0.8, 5.5), 0.6, 0.6, facecolor='lightblue', edgecolor='black'))
        ax.text(10.8 + i*0.8, 5.8, f'x{i}', ha='center', va='center', fontsize=9)
    kernel3 = patches.Rectangle((11.3, 4.5), 0.8, 0.6, facecolor='red', edgecolor='black', alpha=0.7)
    ax.add_patch(kernel3)
    ax.text(11.7, 4.8, 'K', ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    ax.text(7, 3.5, 'Dilated Convolution의 장점:', fontsize=12, fontweight='bold', ha='center')
    ax.text(0.5, 3, '• Dilation=1: 인접한 3개 요소만 보임', fontsize=10, ha='left')
    ax.text(0.5, 2.6, '• Dilation=2: 2칸 간격으로 3개 요소를 보며, 수용 영역 확장', fontsize=10, ha='left')
    ax.text(0.5, 2.2, '• Dilation=4: 4칸 간격으로 3개 요소를 보며, 더 넓은 수용 영역', fontsize=10, ha='left')
    ax.text(0.5, 1.8, '• 지수적 수용 영역 확장으로 긴 시퀀스의 의존성 학습 가능', fontsize=10, ha='left')
    ax.text(7, 1.2, '수용 영역 계산:', fontsize=12, fontweight='bold', ha='center')
    ax.text(7, 0.8, 'Level 1: 3, Level 2: 7, Level 3: 15', fontsize=11, ha='center')
    ax.text(7, 0.4, '총 수용 영역: 15 timesteps', fontsize=11, ha='center', fontweight='bold')
    plt.tight_layout()
    return fig
if __name__ == "__main__":
    fig1 = create_drc_tcn_architecture()
    fig1.savefig('drc_tcn_architecture.png', dpi=300, bbox_inches='tight')
    print("✅ DRC-TCN 아키텍처 다이어그램이 'drc_tcn_architecture.png'로 저장되었습니다.")
    fig2 = create_dilated_conv_visualization()
    fig2.savefig('dilated_conv_visualization.png', dpi=300, bbox_inches='tight')
    print("✅ Dilated Convolution 시각화가 'dilated_conv_visualization.png'로 저장되었습니다.")
    plt.show()
