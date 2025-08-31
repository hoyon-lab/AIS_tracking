# 🚢 AIS Track Analysis System

AIS(Automatic Identification System) 데이터를 활용한 선박 추적 및 분석 시스템입니다.  
CNN-LSTM, 오리지널 LSTM, VQ-LSTM, TCN 기반 선박 분류와 VAE 기반 이상 탐지 기능을 제공합니다.

---

# 🚢 AIS Track Analysis System (English Version)

AIS (Automatic Identification System) data-based vessel tracking and analysis system.  
Provides vessel classification based on CNN-LSTM, Original LSTM, VQ-LSTM, TCN, and anomaly detection using VAE.

## 📋 Project Overview

### 🎯 Key Features
- **Vessel Classification**: MMSI-based vessel identification using CNN-LSTM, Original LSTM, VQ-LSTM, TCN
- **Anomaly Detection**: Abnormal vessel behavior detection using VAE
- **Various Loss Functions**: Support for Cross Entropy, Triplet Loss, Contrastive Loss, Combined Loss
- **Visualization**: Confusion Matrix, t-SNE embeddings, reconstruction error distribution, etc.

### 🏗️ Architecture
- **CNN-LSTM**: Learning spatial and temporal patterns from time-series AIS data
- **Original LSTM**: Pure LSTM-based time-series pattern learning
- **VQ-LSTM**: LSTM with Vector Quantization for more efficient embedding learning
- **VQ Bottleneck LSTM**: CNN-VQ-LSTM structure for information compression through semantic abstraction
- **Dual Stream LSTM**: Complementary feature learning by combining CNN-LSTM and VQ-LSTM
- **TCN (Temporal Convolutional Network)**: Time-series modeling based on dilated convolution
- **Dilated Residual TCN**: Improved TCN with Layer Normalization
- **Attention TCN**: TCN combined with Multi-head Attention
- **TCN-Transformer**: TCN + Transformer Hybrid model
- **Graph TCN**: TCN combined with Graph Convolution
- **MS-TCN-RF**: Multi-Scale TCN with Receptive Field Search (capturing both long and short-term patterns)
- **VAE (Variational AutoEncoder)**: Anomaly detection after learning normal patterns
- **Data Processing**: MinMaxScaler normalization, sequence generation

## 🚀 Installation and Execution

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv ais_tracking_env

# Activate virtual environment (Windows)
.\ais_tracking_env\Scripts\Activate.ps1

# Activate virtual environment (Linux/Mac)
source ais_tracking_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Execution Methods

#### Basic Execution (CNN-LSTM Classification)
```bash
python main_unified.py --csv_path your_data.csv
```

#### Classification with Different Models
```bash
# Original LSTM
python main_unified.py --model_type original_lstm --csv_path your_data.csv

# VQ-LSTM
python main_unified.py --model_type vq_lstm --csv_path your_data.csv

# VQ Bottleneck LSTM (CNN-VQ-LSTM)
python main_unified.py --model_type vq_bottleneck_lstm --csv_path your_data.csv

# Dual Stream LSTM (CNN-LSTM + VQ-LSTM)
python main_unified.py --model_type dual_stream_lstm --csv_path your_data.csv

# TCN (Basic)
python main_unified.py --model_type tcn --csv_path your_data.csv

# Dilated Residual TCN (Improved version)
python main_unified.py --model_type dilated_residual_tcn --csv_path your_data.csv

# Attention TCN
python main_unified.py --model_type attention_tcn --csv_path your_data.csv

# TCN-Transformer Hybrid
python main_unified.py --model_type tcn_transformer --csv_path your_data.csv

# Graph TCN
python main_unified.py --model_type graph_tcn --csv_path your_data.csv

# MS-TCN-RF (Multi-Scale TCN with Receptive Field Search)
python main_unified.py --model_type ms_tcn_rf --csv_path your_data.csv

# MS-TCN-RF Customization
python main_unified.py --model_type ms_tcn_rf --ms_use_delta_features --ms_use_deformable --ms_rf_branches "0.25,1,2,4" --csv_path your_data.csv
```

#### Using Various Loss Functions
```bash
# Cross Entropy (Default)
python main_unified.py --loss_type cross_entropy --csv_path your_data.csv

# Triplet Loss
python main_unified.py --loss_type triplet --triplet_margin 1.5 --csv_path your_data.csv

# Contrastive Loss
python main_unified.py --loss_type contrastive --contrastive_temperature 0.05 --csv_path your_data.csv

# Combined Loss (Cross Entropy + Triplet/Contrastive)
python main_unified.py --loss_type combined --combined_weight 0.7 --csv_path your_data.csv
```

#### VAE Anomaly Detection
```bash
python main_unified.py --mode anomaly --csv_path your_data.csv
```

#### Test Only Execution with Existing Checkpoints
```bash
python main_unified.py --test_only
```

## 📊 Usage

### Command Line Options
```bash
python main_unified.py [OPTIONS]

Options:
  --mode {classification,anomaly}     Analysis mode (default: classification)
  --model_type {cnn_lstm,original_lstm,vq_lstm,vq_bottleneck_lstm,dual_stream_lstm,tcn,dilated_residual_tcn,attention_tcn,tcn_transformer,graph_tcn}  Model type (default: tcn)
  --csv_path PATH                     AIS data CSV file path
  --epochs INT                        Number of training epochs (default: 80)
  --beta FLOAT                        VAE KL loss weight (default: 0.1)
  --lr FLOAT                          Learning rate (default: 5e-3)
  --weight_decay FLOAT                Weight decay (default: 1e-4)
  --grad_clip FLOAT                   Gradient clipping value (default: 1.0)
  --label_smoothing FLOAT             Label smoothing for CE loss (default: 0.1)
  --scheduler_warmup_epochs INT       LR warmup epochs (default: 1)
  --min_lr_ratio FLOAT                Minimum learning rate ratio (default: 0.4)
  --restart_period INT                 cosine_restart period (default: 10)
  --loss_type {cross_entropy,triplet,contrastive,combined}  Loss function type (default: cross_entropy)
  --triplet_margin FLOAT              Triplet loss margin (default: 1.0)
  --contrastive_temperature FLOAT     Contrastive loss temperature (default: 0.07)
  --combined_weight FLOAT             Combined loss weight (default: 0.5)
  --device {auto,cuda,cpu}           Computing device selection (default: auto)
  --split_path PATH                   Split indices file path
  --test_only                         Run test only without training if checkpoints exist
```

### Examples
```bash
# CNN-LSTM Classification (Basic)
python main_unified.py --csv_path combined_output.csv --epochs 100

# Original LSTM Classification
python main_unified.py --model_type original_lstm --csv_path combined_output.csv

# TCN Classification (Basic)
python main_unified.py --model_type tcn --csv_path combined_output.csv

# Attention TCN + Triplet Loss
python main_unified.py --model_type attention_tcn --loss_type triplet --triplet_margin 1.5 --csv_path combined_output.csv

# TCN-Transformer + Contrastive Loss
python main_unified.py --model_type tcn_transformer --loss_type contrastive --contrastive_temperature 0.05 --csv_path combined_output.csv

# Dilated Residual TCN + Combined Loss
python main_unified.py --model_type dilated_residual_tcn --loss_type combined --combined_weight 0.7 --csv_path combined_output.csv

# VAE Anomaly Detection
python main_unified.py --mode anomaly --csv_path combined_output.csv --epochs 50

# Test only with saved model
python main_unified.py --test_only
```

## 📁 Data Format

### Input CSV File Structure
```
MMSI,BaseDateTime,LAT,LON,SOG,COG,Heading,WDIR,WSPD,GST,PRES,ATMP,WTMP,...
```

### Required Columns
- **Classification Mode**: `LAT`, `LON`, `SOG`, `COG`, `Heading`
- **Anomaly Detection Mode**: `LAT`, `LON`, `SOG`, `COG`, `Heading`, `WDIR`, `WSPD`, `GST`, `PRES`, `ATMP`, `WTMP`

## 🧠 Model Description

### CNN-LSTM Classification Model
- **Purpose**: MMSI-based vessel classification
- **Input**: 5 features (latitude, longitude, speed, course, heading)
- **Structure**: CNN → LSTM → Classifier
- **Features**: Learning both spatial patterns and temporal dependencies

### Original LSTM Model
- **Purpose**: MMSI-based vessel classification
- **Input**: 5 features (latitude, longitude, speed, course, heading)
- **Structure**: LSTM → Classifier
- **Features**: Pure LSTM for time-series pattern learning

### Dual Stream LSTM Model
- **Purpose**: MMSI-based vessel classification
- **Input**: 5 features (latitude, longitude, speed, course, heading)
- **Structure**: CNN-LSTM (Stream A) + VQ-LSTM (Stream B) → Feature Fusion → Classifier
- **Features**: Complementary learning of continuous dynamics and discrete semantics

### VAE Anomaly Detection Model
- **Purpose**: Abnormal vessel behavior detection
- **Input**: 11 features (including weather data)
- **Structure**: Encoder → Latent Space → Decoder
- **Principle**: Anomaly detection through reconstruction error after learning normal patterns

### TCN (Temporal Convolutional Network) Model
- **Purpose**: MMSI-based vessel classification
- **Input**: 5 features (latitude, longitude, speed, course, heading)
- **Structure**: Dilated Convolution → Global Average Pooling → Classifier
- **Features**: Parallel processing possible, efficient for long sequences

### Dilated Residual TCN Model
- **Purpose**: MMSI-based vessel classification
- **Input**: 5 features (latitude, longitude, speed, course, heading)
- **Structure**: Dilated Residual Blocks → Global Average Pooling → Classifier
- **Features**: Enhanced stability with Layer Normalization, deep network learning with residual connections

### Attention TCN Model
- **Purpose**: MMSI-based vessel classification
- **Input**: 5 features (latitude, longitude, speed, course, heading)
- **Structure**: TCN → Multi-head Attention → Global Average Pooling → Classifier
- **Features**: Focusing on important time points for more accurate classification

### TCN-Transformer Hybrid Model
- **Purpose**: MMSI-based vessel classification
- **Input**: 5 features (latitude, longitude, speed, course, heading)
- **Structure**: TCN Encoder → Transformer Encoder → Global Average Pooling → Classifier
- **Features**: Powerful model combining long-range dependencies and attention

### Graph TCN Model
- **Purpose**: MMSI-based vessel classification
- **Input**: 5 features (latitude, longitude, speed, course, heading)
- **Structure**: TCN → Graph Convolution → Classifier
- **Features**: More accurate classification by modeling relationships between vessels

### MS-TCN-RF (Multi-Scale TCN with Receptive Field Search) Model
- **Purpose**: MMSI-based vessel classification (capturing both long and short-term patterns)
- **Input**: 5 features (latitude, longitude, speed, course, heading) + Δfeatures + time intervals
- **Structure**: 
  - **Multi-Scale RF Branches**: Parallel processing of different RFs (RF≈0.25×·1×·4×)
  - **Gating/Attention**: Learning optimal combinations through branch-wise weighted sums
  - **Δ-Feature Encoding**: COG/SOG absolute values + ΔCOG, ΔSOG, ΔLat, ΔLon, Δt
  - **Deformable Temporal Convolution**: Learning sample position offsets on the time axis
  - **Segment-Aware Pooling**: Statistics per segment (anchoring/acceleration/turning) + attention pooling
- **Features**: 
  - Robust dual-stream processing for irregular sampling
  - Alignment of vessel's irregular acceleration/deceleration/turning intervals
  - Behavior state-aware embedding

### Loss Functions
- **Cross Entropy**: Basic classification loss function
- **Triplet Loss**: Learning to bring same vessels closer and different vessels farther apart
- **Contrastive Loss**: Similarity-based learning for better embedding space generation
- **Combined Loss**: Combination of Cross Entropy and Triplet/Contrastive Loss

## 📈 Output Results

### Classification Mode
- **Performance Metrics**: Accuracy, Precision, Recall, F1-score
- **Visualization**: 
  - Confusion Matrix (original/normalized)
  - t-SNE embedding space
- **Saved Files**: `checkpoints/{model_type}_classifier.pt`

### Anomaly Detection Mode
- **Performance Metrics**: Anomaly detection accuracy, threshold
- **Visualization**:
  - Reconstruction error histogram
  - Latent space t-SNE
- **Saved Files**: `checkpoints/vae_final.pt`

## 🔧 Key Files

### Core Files
- `main_unified.py`: Main execution file
- `cnn_lstm_model.py`: CNN-LSTM classification model
- `lstm_models.py`: Original LSTM, VQ-LSTM models
- `va_vae_model.py`: VAE anomaly detection model
- `requirements.txt`: Dependency package list

### Data Files
- `combined_output.csv`: AIS data
- `split_indices.pkl`: Data split indices

### Output Files
- `checkpoints/`: Trained model checkpoints
- `logs/`: Training logs
- `*.png`: Visualization results

## 🎯 Model Operation Principles

### Core Ideas of MS-TCN-RF Model
1. **Multi-Scale RF Search**: Parallel processing of different receptive fields to capture both long and short-term patterns
2. **Δ-Feature Encoding**: Processing absolute values, changes, and time intervals as separate streams
3. **Deformable Temporal Convolution**: Learning sample position offsets on the time axis to align irregular patterns
4. **Segment-Aware Pooling**: Behavior state recognition for segment-wise statistics + attention pooling

### Classification Models
1. **Data Preprocessing**: Normalization with MinMaxScaler
2. **Sequence Generation**: Fixed-length sequences of 50 points
3. **Model-specific Processing**:
   - **CNN-LSTM**: CNN → LSTM → Classification
   - **Original LSTM**: LSTM → Classification
   - **VQ-LSTM**: LSTM → Vector Quantizer → Classification
4. **Classification**: MMSI-based vessel prediction

### Anomaly Detection (VAE)
1. **Normal Pattern Learning**: Learning normal behavior with majority of data
2. **Reconstruction**: Input → Latent Space → Restoration
3. **Error Calculation**: Difference between input and restoration results
4. **Anomaly Detection**: Detected as anomaly when exceeding threshold (95%)

## 🔍 Model Comparison

| Model | Features | Advantages | Disadvantages |
|-------|----------|------------|---------------|
| **CNN-LSTM** | Spatial + Temporal patterns | Complex pattern learning | High parameter count |
| **TCN** | Dilated convolution | Parallel processing, long sequences | Limited local patterns |
| **Dilated Residual TCN** | LayerNorm + Residual | Stable learning, deep networks | Increased parameter count |
| **Attention TCN** | TCN + Attention | Focus on important time points | Increased computational complexity |
| **TCN-Transformer** | TCN + Transformer | Long-range dependencies | Very complex structure |
| **Graph TCN** | TCN + Graph Conv | Vessel relationship modeling | Graph structure required |
| **MS-TCN-RF** | Multi-Scale RF + Δ-Features | Capturing long/short-term patterns, robust to irregular sampling | Complex structure, high computational cost |

## 🚨 Troubleshooting

### When Confusion Matrix Shows Only One Color
- Possibility of predictions being biased toward one class
- Check prediction/actual distribution in code
- Check data imbalance

### Memory Insufficient Error
- Reduce `batch_size`
- Reduce `fixed_len`
- Check GPU memory

### Checkpoint Load Failure
- Check file path
- Check model structure compatibility

### TCN Model Training Considerations
- **TCN**: Adjust kernel_size and num_channels for performance improvement
- **Attention TCN**: Optimize attention effects by adjusting num_heads
- **TCN-Transformer**: Optimize transformer performance by adjusting num_layers and num_heads
- **Graph TCN**: Graph structure definition may be required
- **MS-TCN-RF**: 
  - `--ms_use_delta_features`: Delta features usage (default: True)
  - `--ms_use_deformable`: Deformable convolution usage (default: True)
  - `--ms_rf_branches`: RF branch ratio setting (default: "1/4,1,4")
  - `--tcn_channels`: Adjust channel count to control model capacity

### Loss Function Selection Guide
- **Cross Entropy**: Suitable for basic classification
- **Triplet Loss**: Effective for learning vessel similarities
- **Contrastive Loss**: Useful for improving embedding space quality
- **Combined Loss**: Effective for complex pattern learning

## 📝 License

This project is created for educational and research purposes.

## 🤝 Contributing

Bug reports and feature suggestions are always welcome!

---

**Note**: This system is developed for research purposes related to maritime traffic safety and vessel tracking.

---

# 🚢 AIS Track Analysis System (한국어 버전)

AIS(Automatic Identification System) 데이터를 활용한 선박 추적 및 분석 시스템입니다.  
CNN-LSTM, 오리지널 LSTM, VQ-LSTM, TCN 기반 선박 분류와 VAE 기반 이상 탐지 기능을 제공합니다.

## 📋 프로젝트 개요

### 🎯 주요 기능
- **선박 분류 (Classification)**: CNN-LSTM, TCN을 사용한 MMSI별 선박 식별
- **이상 탐지 (Anomaly Detection)**: VAE를 사용한 비정상 선박 행동 탐지
- **다양한 손실 함수**: Cross Entropy, Triplet Loss, Contrastive Loss, Combined Loss 지원
- **시각화**: Confusion Matrix, t-SNE 임베딩, 재구성 오차 분포 등

### 🏗️ 아키텍처
- **CNN-LSTM**: 시계열 AIS 데이터의 공간적/시간적 패턴 학습
- **TCN (Temporal Convolutional Network)**: Dilated convolution 기반 시계열 모델링
- **Dilated Residual TCN**: Layer Normalization이 추가된 개선된 TCN
- **Attention TCN**: Multi-head Attention이 결합된 TCN
- **TCN-Transformer**: TCN + Transformer Hybrid 모델
- **Graph TCN**: Graph Convolution이 결합된 TCN
- **MS-TCN-RF**: Multi-Scale TCN with Receptive Field Search (장·단기 패턴 동시 포착)
- **VAE (Variational AutoEncoder)**: 정상 패턴 학습 후 이상 탐지
- **데이터 처리**: MinMaxScaler 정규화, 시퀀스 생성

## 🚀 설치 및 실행

### 1. 환경 설정
```bash
# 가상환경 생성
python -m venv ais_tracking_env

# 가상환경 활성화 (Windows)
.\ais_tracking_env\Scripts\Activate.ps1

# 가상환경 활성화 (Linux/Mac)
source ais_tracking_env/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 실행 방법

#### 기본 실행 (CNN-LSTM 분류)
```bash
python main_unified.py --csv_path your_data.csv
```

#### 다른 모델로 분류
```bash
# TCN (기본)
python main_unified.py --model_type tcn --csv_path your_data.csv

# Dilated Residual TCN (개선된 버전)
python main_unified.py --model_type dilated_residual_tcn --csv_path your_data.csv

# Attention TCN
python main_unified.py --model_type attention_tcn --csv_path your_data.csv

# TCN-Transformer Hybrid
python main_unified.py --model_type tcn_transformer --csv_path your_data.csv

# Graph TCN
python main_unified.py --model_type graph_tcn --csv_path your_data.csv

# MS-TCN-RF (Multi-Scale TCN with Receptive Field Search)
python main_unified.py --model_type ms_tcn_rf --csv_path your_data.csv

# MS-TCN-RF 커스터마이징
python main_unified.py --model_type ms_tcn_rf --ms_use_delta_features --ms_use_deformable --ms_rf_branches "0.25,1,2,4" --csv_path your_data.csv
```

#### 다양한 손실 함수 사용
```bash
# Cross Entropy (기본)
python main_unified.py --loss_type cross_entropy --csv_path your_data.csv

# Triplet Loss
python main_unified.py --loss_type triplet --triplet_margin 1.5 --csv_path your_data.csv

# Contrastive Loss
python main_unified.py --loss_type contrastive --contrastive_temperature 0.05 --csv_path your_data.csv

# Combined Loss (Cross Entropy + Triplet/Contrastive)
python main_unified.py --loss_type combined --combined_weight 0.7 --csv_path your_data.csv
```

#### VAE 이상 탐지
```bash
python main_unified.py --mode anomaly --csv_path your_data.csv
```

#### 체크포인트가 있을 때 테스트만 실행
```bash
python main_unified.py --test_only
```

## 📊 사용법

### 명령행 옵션
```bash
python main_unified.py [OPTIONS]

Options:
  --mode {classification,anomaly}     분석 모드 (기본값: classification)
  --model_type {cnn_lstm,original_lstm,tcn,dilated_residual_tcn,attention_tcn,tcn_transformer,graph_tcn}  모델 타입 (기본값: tcn)
  --csv_path PATH                     AIS 데이터 CSV 파일 경로
  --epochs INT                        학습 에포크 수 (기본값: 80)
  --beta FLOAT                        VAE KL 손실 가중치 (기본값: 0.1)
  --lr FLOAT                          학습률 (기본값: 5e-3)
  --weight_decay FLOAT                Weight decay (기본값: 1e-4)
  --grad_clip FLOAT                   Gradient clipping value (기본값: 1.0)
  --label_smoothing FLOAT             Label smoothing for CE loss (기본값: 0.1)
  --scheduler_warmup_epochs INT       LR warmup epochs (기본값: 1)
  --min_lr_ratio FLOAT                최소 학습률 비율 (기본값: 0.4)
  --restart_period INT                cosine_restart 주기 (기본값: 10)
  --loss_type {cross_entropy,triplet,contrastive,combined}  손실 함수 타입 (기본값: cross_entropy)
  --triplet_margin FLOAT              Triplet loss margin (기본값: 1.0)
  --contrastive_temperature FLOAT     Contrastive loss temperature (기본값: 0.07)
  --combined_weight FLOAT             Combined loss weight (기본값: 0.5)
  --device {auto,cuda,cpu}           연산 디바이스 선택 (기본값: auto)
  --split_path PATH                   분할 인덱스 파일 경로
  --test_only                         체크포인트가 있으면 학습 없이 테스트만 실행
```

### 예시
```bash
# CNN-LSTM 분류 (기본)
python main_unified.py --csv_path combined_output.csv --epochs 100

# TCN 분류 (기본)
python main_unified.py --model_type tcn --csv_path combined_output.csv

# Attention TCN + Triplet Loss
python main_unified.py --model_type attention_tcn --loss_type triplet --triplet_margin 1.5 --csv_path combined_output.csv

# TCN-Transformer + Contrastive Loss
python main_unified.py --model_type tcn_transformer --loss_type contrastive --contrastive_temperature 0.05 --csv_path combined_output.csv

# Dilated Residual TCN + Combined Loss
python main_unified.py --model_type dilated_residual_tcn --loss_type combined --combined_weight 0.7 --csv_path combined_output.csv

# VAE 이상 탐지
python main_unified.py --mode anomaly --csv_path combined_output.csv --epochs 50

# 저장된 모델로 테스트만 실행
python main_unified.py --test_only
```

## 📁 데이터 형식

### 입력 CSV 파일 구조
```
MMSI,BaseDateTime,LAT,LON,SOG,COG,Heading,WDIR,WSPD,GST,PRES,ATMP,WTMP,...
```

### 필수 컬럼
- **분류 모드**: `LAT`, `LON`, `SOG`, `COG`, `Heading`
- **이상탐지 모드**: `LAT`, `LON`, `SOG`, `COG`, `Heading`, `WDIR`, `WSPD`, `GST`, `PRES`, `ATMP`, `WTMP`

## 🧠 모델 설명

### CNN-LSTM 분류 모델
- **목적**: MMSI별 선박 분류
- **입력**: 5개 특성 (위도, 경도, 속도, 방향, 헤딩)
- **구조**: CNN → LSTM → 분류기
- **특징**: 공간적 패턴과 시간적 의존성을 모두 학습

### 오리지널 LSTM 모델
- **목적**: MMSI별 선박 분류
- **입력**: 5개 특성 (위도, 경도, 속도, 방향, 헤딩)
- **구조**: LSTM → 분류기
- **특징**: 순수 LSTM으로 시계열 패턴 학습

### VQ-LSTM 모델
- **목적**: MMSI별 선박 분류
- **입력**: 5개 특성 (위도, 경도, 속도, 방향, 헤딩)
- **구조**: LSTM → Vector Quantizer → 분류기
- **특징**: Vector Quantization으로 더 효율적인 임베딩 학습

### VQ Bottleneck LSTM 모델
- **목적**: MMSI별 선박 분류
- **입력**: 5개 특성 (위도, 경도, 속도, 방향, 헤딩)
- **구조**: CNN → VQ Bottleneck → LSTM → 분류기
- **특징**: Semantic abstraction을 통한 정보 압축 및 잡음 제거

### Dual Stream LSTM 모델
- **목적**: MMSI별 선박 분류
- **입력**: 5개 특성 (위도, 경도, 속도, 방향, 헤딩)
- **구조**: CNN-LSTM (Stream A) + VQ-LSTM (Stream B) → Feature Fusion → 분류기
- **특징**: 연속적 동역학과 이산적 의미론의 보완적 학습

### VAE 이상 탐지 모델
- **목적**: 비정상 선박 행동 탐지
- **입력**: 11개 특성 (기상 데이터 포함)
- **구조**: 인코더 → 잠재공간 → 디코더
- **원리**: 정상 패턴 학습 후 재구성 오차로 이상 탐지

### TCN (Temporal Convolutional Network) 모델
- **목적**: MMSI별 선박 분류
- **입력**: 5개 특성 (위도, 경도, 속도, 방향, 헤딩)
- **구조**: Dilated Convolution → Global Average Pooling → 분류기
- **특징**: 병렬 처리 가능, 긴 시퀀스 처리에 효율적

### Dilated Residual TCN 모델
- **목적**: MMSI별 선박 분류
- **입력**: 5개 특성 (위도, 경도, 속도, 방향, 헤딩)
- **구조**: Dilated Residual Blocks → Global Average Pooling → 분류기
- **특징**: Layer Normalization으로 안정성 향상, Residual connection으로 깊은 네트워크 학습

### Attention TCN 모델
- **목적**: MMSI별 선박 분류
- **입력**: 5개 특성 (위도, 경도, 속도, 방향, 헤딩)
- **구조**: TCN → Multi-head Attention → Global Average Pooling → 분류기
- **특징**: 중요한 시점에 집중하여 더 정확한 분류

### TCN-Transformer Hybrid 모델
- **목적**: MMSI별 선박 분류
- **입력**: 5개 특성 (위도, 경도, 속도, 방향, 헤딩)
- **구조**: TCN Encoder → Transformer Encoder → Global Average Pooling → 분류기
- **특징**: 장거리 의존성과 Attention을 결합한 강력한 모델

### Graph TCN 모델
- **목적**: MMSI별 선박 분류
- **입력**: 5개 특성 (위도, 경도, 속도, 방향, 헤딩)
- **구조**: TCN → Graph Convolution → 분류기
- **특징**: 선박간 관계를 모델링하여 더 정확한 분류

### MS-TCN-RF (Multi-Scale TCN with Receptive Field Search) 모델
- **목적**: MMSI별 선박 분류 (장·단기 패턴 동시 포착)
- **입력**: 5개 특성 (위도, 경도, 속도, 방향, 헤딩) + Δ특성 + 시간간격
- **구조**: 
  - **Multi-Scale RF 브랜치**: 서로 다른 RF를 병렬로 처리 (RF≈0.25×·1×·4×)
  - **게이팅/어텐션**: 브랜치별 가중합으로 최적 조합 학습
  - **Δ-Feature 인코딩**: COG/SOG 절대값 + ΔCOG, ΔSOG, ΔLat, ΔLon, Δt
  - **Deformable Temporal Convolution**: 시간축에서 샘플 위치 오프셋 학습
  - **Segment-Aware Pooling**: 정박/가속/선회 세그먼트별 통계 + 어텐션 풀링
- **특징**: 
  - 불규칙 샘플링에 강건한 Dual-Stream 처리
  - 선박의 불규칙한 가감속/회전 구간 정렬
  - 행동 상태 인식 기반 임베딩

### 손실 함수들
- **Cross Entropy**: 기본 분류 손실 함수
- **Triplet Loss**: 같은 선박은 가깝게, 다른 선박은 멀게 학습
- **Contrastive Loss**: 유사도 기반 학습으로 더 나은 임베딩 공간 생성
- **Combined Loss**: Cross Entropy와 Triplet/Contrastive Loss를 조합

## 📈 출력 결과

### 분류 모드
- **성능 지표**: Accuracy, Precision, Recall, F1-score
- **시각화**: 
  - Confusion Matrix (원본/정규화)
  - t-SNE 임베딩 공간
- **저장 파일**: `checkpoints/{model_type}_classifier.pt`

### 이상탐지 모드
- **성능 지표**: 이상 탐지 정확도, 임계값
- **시각화**:
  - 재구성 오차 히스토그램
  - 잠재 공간 t-SNE
- **저장 파일**: `checkpoints/vae_final.pt`

## 🔧 주요 파일

### 핵심 파일
- `main_unified.py`: 메인 실행 파일
- `cnn_lstm_model.py`: CNN-LSTM 분류 모델
- `lstm_models.py`: 오리지널 LSTM, VQ-LSTM 모델
- `va_vae_model.py`: VAE 이상 탐지 모델
- `requirements.txt`: 의존성 패키지 목록

### 데이터 파일
- `combined_output.csv`: AIS 데이터
- `split_indices.pkl`: 데이터 분할 인덱스

### 출력 파일
- `checkpoints/`: 학습된 모델 체크포인트
- `logs/`: 학습 로그
- `*.png`: 시각화 결과

## 🎯 모델 동작 원리

### MS-TCN-RF 모델의 핵심 아이디어
1. **Multi-Scale RF Search**: 서로 다른 수용역을 병렬로 처리하여 장·단기 패턴 동시 포착
2. **Δ-Feature 인코딩**: 절대값뿐만 아니라 변화량과 시간간격을 별도 스트림으로 처리
3. **Deformable Temporal Convolution**: 시간축에서 샘플 위치 오프셋을 학습하여 불규칙한 패턴 정렬
4. **Segment-Aware Pooling**: 정박/가속/선회 등 행동 상태를 인식하여 세그먼트별 통계 + 어텐션 풀링

### 분류 모델들
1. **데이터 전처리**: MinMaxScaler로 정규화
2. **시퀀스 생성**: 50개 포인트 고정 길이 시퀀스
3. **모델별 처리**:
   - **CNN-LSTM**: CNN → LSTM → 분류
   - **오리지널 LSTM**: LSTM → 분류
   - **VQ-LSTM**: LSTM → Vector Quantizer → 분류
4. **분류**: MMSI별 선박 예측

### 이상탐지 (VAE)
1. **정상 패턴 학습**: 대부분의 데이터로 정상 행동 학습
2. **재구성**: 입력 → 잠재공간 → 복원
3. **오차 계산**: 입력과 복원 결과의 차이
4. **이상 판정**: 임계값(95%) 초과 시 이상으로 탐지

## 🔍 모델 비교

| 모델 | 특징 | 장점 | 단점 |
|------|------|------|------|
| **CNN-LSTM** | 공간적+시간적 패턴 | 복잡한 패턴 학습 | 파라미터 수 많음 |
| **오리지널 LSTM** | 순수 시계열 | 단순하고 빠름 | 공간적 패턴 제한적 |
| **VQ-LSTM** | 양자화된 임베딩 | 효율적인 표현 | 학습 복잡도 높음 |
| **VQ Bottleneck LSTM** | Semantic abstraction | 잡음 제거, 해석 가능 | 정보 손실 가능성 |
| **Dual Stream LSTM** | 보완적 특성 학습 | 강건한 학습 | 복잡한 구조 |
| **TCN** | Dilated convolution | 병렬 처리, 긴 시퀀스 | 로컬 패턴 제한적 |
| **Dilated Residual TCN** | LayerNorm + Residual | 안정적 학습, 깊은 네트워크 | 파라미터 수 증가 |
| **Attention TCN** | TCN + Attention | 중요한 시점 집중 | 계산 복잡도 증가 |
| **TCN-Transformer** | TCN + Transformer | 장거리 의존성 | 매우 복잡한 구조 |
| **Graph TCN** | TCN + Graph Conv | 선박간 관계 모델링 | 그래프 구조 필요 |
| **MS-TCN-RF** | Multi-Scale RF + Δ-Features | 장·단기 패턴 동시 포착, 불규칙 샘플링 강건 | 복잡한 구조, 계산 비용 높음 |

## 🚨 문제 해결

### Confusion Matrix가 한 색상만 보일 때
- 예측이 한 클래스로 쏠렸을 가능성
- 코드에서 예측/실제 분포 확인
- 데이터 불균형 점검

### 메모리 부족 오류
- `batch_size` 줄이기
- `fixed_len` 줄이기
- GPU 메모리 확인

### 체크포인트 로드 실패
- 파일 경로 확인
- 모델 구조 일치 여부 확인

### VQ-LSTM 학습 시 주의사항
- commitment_cost 조정 필요할 수 있음
- perplexity 값 모니터링
- VQ 손실과 분류 손실 균형 조정

### TCN 모델 학습 시 주의사항
- **TCN**: kernel_size와 num_channels 조정으로 성능 향상
- **Attention TCN**: num_heads 조정으로 attention 효과 최적화
- **TCN-Transformer**: num_layers와 num_heads 조정으로 transformer 성능 최적화
- **Graph TCN**: 그래프 구조 정의가 필요할 수 있음
- **MS-TCN-RF**: 
  - `--ms_use_delta_features`: Delta features 사용 여부 (기본값: True)
  - `--ms_use_deformable`: Deformable convolution 사용 여부 (기본값: True)
  - `--ms_rf_branches`: RF 브랜치 비율 설정 (기본값: "1/4,1,4")
  - `--tcn_channels`: 채널 수 조정으로 모델 용량 조절

### 손실 함수 선택 가이드
- **Cross Entropy**: 기본 분류에 적합
- **Triplet Loss**: 선박 간 유사도 학습에 효과적
- **Contrastive Loss**: 임베딩 공간 품질 향상에 유용
- **Combined Loss**: 복잡한 패턴 학습에 효과적

## 📝 라이센스

이 프로젝트는 교육 및 연구 목적으로 제작되었습니다.

## 🤝 기여

버그 리포트나 기능 제안은 언제든 환영합니다!

---

**참고**: 이 시스템은 해양 교통 안전과 선박 추적을 위한 연구 목적으로 개발되었습니다. 
