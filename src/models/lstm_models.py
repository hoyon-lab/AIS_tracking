import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.xavier_uniform_(self.embedding.weight)
    def forward(self, inputs):
        batch_size = inputs.size(0)
        distances = torch.sum(inputs**2, dim=1, keepdim=True) + \
                   torch.sum(self.embedding.weight**2, dim=1) - \
                   2 * torch.matmul(inputs, self.embedding.weight.t())
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = torch.zeros(batch_size, self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)
        quantized = torch.matmul(encodings, self.embedding.weight)
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return quantized, loss, perplexity
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()
    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)
class DilatedResidualBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(n_outputs)  
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(n_outputs)  
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.bn1, self.relu1, self.dropout1,
                                self.conv2, self.chomp2, self.bn2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()
    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model)
        output = self.w_o(attention_output)
        return output, attention_weights
class TCN(nn.Module):
    def __init__(self, input_dim=5, num_channels=[48, 96, 192], kernel_size=5, dropout=0.15, 
                 num_classes=None, mode='classification'):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.mode = mode
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                   dilation=dilation_size, padding=(kernel_size-1) * dilation_size,
                                   dropout=dropout)]
        self.tcn = nn.Sequential(*layers)
        if mode == 'classification' and num_classes:
            self.classifier = nn.Sequential(
                nn.Linear(num_channels[-1], num_channels[-1] // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(num_channels[-1] // 2, num_classes)
            )
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x = x.transpose(1, 2)  
        x = self.tcn(x)
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)  
        if self.mode == 'classification' and self.num_classes:
            output = self.classifier(x)
            return output, x
        else:
            return x, x
class DilatedResidualTCN(nn.Module):
    def __init__(self, input_dim=5, num_channels=[48, 96, 192], kernel_size=5, dropout=0.15, 
                 num_classes=None, mode='classification'):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.mode = mode
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [DilatedResidualBlock(in_channels, out_channels, kernel_size, stride=1,
                                          dilation=dilation_size, padding=(kernel_size-1) * dilation_size,
                                          dropout=dropout)]
        self.tcn = nn.Sequential(*layers)
        if mode == 'classification' and num_classes:
            self.classifier = nn.Sequential(
                nn.Linear(num_channels[-1], num_channels[-1] // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(num_channels[-1] // 2, num_classes)
            )
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x = x.transpose(1, 2)
        x = self.tcn(x)
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        if self.mode == 'classification' and self.num_classes:
            output = self.classifier(x)
            return output, x
        else:
            return x, x
class AttentionTCN(nn.Module):
    def __init__(self, input_dim=5, num_channels=[48, 96, 192], kernel_size=5, dropout=0.15, 
                 num_heads=12, num_classes=None, mode='classification'):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.mode = mode
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                   dilation=dilation_size, padding=(kernel_size-1) * dilation_size,
                                   dropout=dropout)]
        self.tcn = nn.Sequential(*layers)
        self.attention = MultiHeadAttention(num_channels[-1], num_heads, dropout)
        if mode == 'classification' and num_classes:
            self.classifier = nn.Sequential(
                nn.Linear(num_channels[-1], num_channels[-1] // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(num_channels[-1] // 2, num_classes)
            )
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x = x.transpose(1, 2)
        x = self.tcn(x)
        x = x.transpose(1, 2)  
        x, attention_weights = self.attention(x)
        x = x.mean(dim=1)  
        if self.mode == 'classification' and self.num_classes:
            output = self.classifier(x)
            return output, x
        else:
            return x, x
class TCNTransformer(nn.Module):
    def __init__(self, input_dim=5, num_channels=[48, 96, 192], kernel_size=5, dropout=0.15, 
                 num_heads=12, num_layers=3, num_classes=None, mode='classification'):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.mode = mode
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                   dilation=dilation_size, padding=(kernel_size-1) * dilation_size,
                                   dropout=dropout)]
        self.tcn = nn.Sequential(*layers)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=num_channels[-1], 
            nhead=num_heads, 
            dim_feedforward=num_channels[-1] * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        if mode == 'classification' and num_classes:
            self.classifier = nn.Sequential(
                nn.Linear(num_channels[-1], num_channels[-1] // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(num_channels[-1] // 2, num_classes)
            )
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x = x.transpose(1, 2)
        x = self.tcn(x)
        x = x.transpose(1, 2)  
        x = self.transformer(x)
        x = x.mean(dim=1)  
        if self.mode == 'classification' and self.num_classes:
            output = self.classifier(x)
            return output, x
        else:
            return x, x
class GraphTCN(nn.Module):
    def __init__(self, input_dim=5, num_channels=[48, 96, 192], kernel_size=5, dropout=0.15, 
                 num_classes=None, mode='classification'):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.mode = mode
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                   dilation=dilation_size, padding=(kernel_size-1) * dilation_size,
                                   dropout=dropout)]
        self.tcn = nn.Sequential(*layers)
        self.graph_conv = nn.Linear(num_channels[-1], num_channels[-1])
        if mode == 'classification' and num_classes:
            self.classifier = nn.Sequential(
                nn.Linear(num_channels[-1], num_channels[-1] // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(num_channels[-1] // 2, num_classes)
            )
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x = x.transpose(1, 2)
        x = self.tcn(x)
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        x = self.graph_conv(x)
        if self.mode == 'classification' and self.num_classes:
            output = self.classifier(x)
            return output, x
        else:
            return x, x
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        loss = F.relu(pos_dist - neg_dist + self.margin)
        if torch.isnan(loss.mean()) or torch.isinf(loss.mean()):
            print(f"⚠️ TripletLoss에서 nan/inf 발생: {loss.mean()}")
            print(f"   pos_dist 범위: {pos_dist.min():.4f} ~ {pos_dist.max():.4f}")
            print(f"   neg_dist 범위: {neg_dist.min():.4f} ~ {neg_dist.max():.4f}")
            return torch.tensor(0.0, device=anchor.device, requires_grad=True)
        return loss.mean()
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    def forward(self, features, labels):
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).float()
        logits_mask = torch.ones_like(mask)
        logits_mask.fill_diagonal_(0)
        mask = mask * logits_mask
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        log_sum_exp = exp_logits.sum(1, keepdim=True)
        log_sum_exp = torch.clamp(log_sum_exp, min=1e-8)
        log_prob = similarity_matrix - torch.log(log_sum_exp)
        mask_sum = mask.sum(1)
        mask_sum = torch.clamp(mask_sum, min=1e-8)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        loss = -mean_log_prob_pos.mean()
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"⚠️ ContrastiveLoss에서 nan/inf 발생: {loss}")
            print(f"   similarity_matrix 범위: {similarity_matrix.min():.4f} ~ {similarity_matrix.max():.4f}")
            print(f"   exp_logits 범위: {exp_logits.min():.4f} ~ {exp_logits.max():.4f}")
            print(f"   mask_sum 범위: {mask_sum.min():.4f} ~ {mask_sum.max():.4f}")
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        return loss
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        gather_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        gather_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1.0 - gather_probs).pow(self.gamma)
        loss = -focal_weight * gather_log_probs
        if isinstance(self.alpha, torch.Tensor):
            alpha_weight = self.alpha.to(logits.device)[targets]
            loss = alpha_weight * loss
        elif isinstance(self.alpha, (float, int)):
            loss = float(self.alpha) * loss
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss
class OriginalLSTM(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128, num_layers=2, num_classes=None, mode='classification'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.mode = mode
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        if mode == 'classification' and num_classes:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim // 2, num_classes)
            )
    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]  
        if self.mode == 'classification' and self.num_classes:
            output = self.classifier(last_hidden)
            return output, last_hidden
        else:
            return last_hidden, last_hidden
class VQLSTM(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128, num_layers=2, num_embeddings=128, 
                 embedding_dim=64, commitment_cost=0.25, num_classes=None, mode='classification'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.mode = mode
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.projection = nn.Linear(hidden_dim, embedding_dim)
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        if mode == 'classification' and num_classes:
            self.classifier = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(embedding_dim // 2, num_classes)
            )
    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]  
        projected = self.projection(last_hidden)
        vq_output, vq_loss, perplexity = self.vq_layer(projected)
        if self.mode == 'classification' and self.num_classes:
            output = self.classifier(vq_output)
            return output, vq_output, vq_loss, perplexity
        else:
            return vq_output, vq_output, vq_loss, perplexity
class VQBottleneckLSTM(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=16, num_layers=1,  
                 num_embeddings=16, embedding_dim=8, commitment_cost=0.01,  
                 num_classes=None, mode='classification'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.mode = mode
        self.conv1 = nn.Conv1d(input_dim, 4, kernel_size=3, padding=1)  
        self.bn1 = nn.BatchNorm1d(4)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.projection = nn.Linear(4, embedding_dim)  
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=0.2)
        if mode == 'classification' and num_classes:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim // 2, num_classes)
            )
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        batch_size, seq_len, cnn_dim = x.shape
        x_flat = x.reshape(-1, cnn_dim)
        x_projected = self.projection(x_flat)
        try:
            vq_output, vq_loss, perplexity = self.vq_layer(x_projected)
            vq_loss = torch.clamp(vq_loss, max=5.0)  
        except Exception as e:
            print(f"VQ Layer 오류: {e}")
            vq_output = x_projected
            vq_loss = torch.tensor(0.0, device=x.device, requires_grad=True)
            perplexity = torch.tensor(1.0, device=x.device)
        vq_output = vq_output.reshape(batch_size, seq_len, -1)
        lstm_out, (h_n, c_n) = self.lstm(vq_output)
        last_hidden = h_n[-1]
        if self.mode == 'classification' and self.num_classes:
            output = self.classifier(last_hidden)
            return output, last_hidden, vq_loss, perplexity
        else:
            return last_hidden, last_hidden, vq_loss, perplexity
class DualStreamLSTM(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=2, num_embeddings=64, 
                 embedding_dim=32, commitment_cost=0.01, num_classes=None, mode='classification'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.mode = mode
        self.conv1 = nn.Conv1d(input_dim, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.cnn_lstm = nn.LSTM(32, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.vq_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.projection = nn.Linear(hidden_dim, embedding_dim)
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        fusion_dim = hidden_dim * 2
        if mode == 'classification' and num_classes:
            self.classifier = nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(fusion_dim // 2, num_classes)
            )
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x_cnn = x.transpose(1, 2)
        x_cnn = self.conv1(x_cnn)
        x_cnn = self.bn1(x_cnn)
        x_cnn = self.relu(x_cnn)
        x_cnn = self.dropout(x_cnn)
        x_cnn = x_cnn.transpose(1, 2)
        cnn_lstm_out, (cnn_h_n, _) = self.cnn_lstm(x_cnn)
        cnn_features = cnn_h_n[-1]
        vq_lstm_out, (vq_h_n, _) = self.vq_lstm(x)
        vq_hidden = vq_h_n[-1]
        vq_projected = self.projection(vq_hidden)
        vq_output, vq_loss, perplexity = self.vq_layer(vq_projected)
        combined_features = torch.cat([cnn_features, vq_output], dim=1)
        if self.mode == 'classification' and self.num_classes:
            output = self.classifier(combined_features)
            return output, combined_features, vq_loss, perplexity
        else:
            return combined_features, combined_features, vq_loss, perplexity
class CNNLSTMTrackNet(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128, num_layers=2, num_classes=None, mode='classification'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.mode = mode
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.lstm = nn.LSTM(64, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        if mode == 'classification' and num_classes:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim // 2, num_classes)
            )
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x = x.transpose(1, 2)  
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)  
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]  
        if self.mode == 'classification' and self.num_classes:
            output = self.classifier(last_hidden)
            return output, last_hidden
        else:
            return last_hidden, last_hidden
class MSTCNRF(nn.Module):
    def __init__(self, input_dim=5, num_channels=[48, 96, 192], kernel_size=5, dropout=0.15, 
                 num_classes=None, mode='classification', use_delta_features=True, use_deformable=True,
                 rf_branches='1/4,1,4'):
        super().__init__()
        self.input_dim = input_dim
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.num_classes = num_classes
        self.mode = mode
        self.use_delta_features = use_delta_features
        self.use_deformable = use_deformable
        self.rf_branches = rf_branches
        if use_delta_features:
            self.delta_encoder = nn.Sequential(
                nn.Linear(input_dim * 2, num_channels[0]),  
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.feature_fusion = nn.Linear(num_channels[0] * 2, num_channels[0])
        else:
            self.input_proj = nn.Linear(input_dim, num_channels[0])
        self.rf_branches = nn.ModuleList()
        def parse_ratio(ratio_str):
            ratio_str = ratio_str.strip()
            if '/' in ratio_str:
                num, denom = ratio_str.split('/')
                return float(num) / float(denom)
            else:
                return float(ratio_str)
        rf_ratios = [parse_ratio(x) for x in rf_branches.split(',')]
        for ratio in rf_ratios:
            if ratio < 1:  
                dilation = 1
                k_size = 3
                padding = 1
            elif ratio == 1:  
                dilation = 1
                k_size = kernel_size
                padding = (kernel_size - 1) // 2
            else:  
                dilation = int(ratio)
                k_size = kernel_size
                padding = (kernel_size - 1) * dilation // 2
            self.rf_branches.append(
                nn.Sequential(
                    nn.Conv1d(num_channels[0], num_channels[0], k_size, padding=padding, dilation=dilation),
                    nn.BatchNorm1d(num_channels[0]),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )
        self.branch_attention = nn.MultiheadAttention(
            embed_dim=num_channels[0], 
            num_heads=8, 
            dropout=dropout,
            batch_first=True
        )
        if use_deformable:
            self.deformable_conv = nn.Sequential(
                nn.Conv1d(num_channels[0], num_channels[0], kernel_size=3, padding=1),
                nn.BatchNorm1d(num_channels[0]),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.offset_conv = nn.Conv1d(num_channels[0], 1, kernel_size=3, padding=1)
        self.tcn_layers = nn.ModuleList()
        in_channels = num_channels[0]
        for i, out_channels in enumerate(num_channels):
            self.tcn_layers.append(
                TemporalBlock(in_channels, out_channels, kernel_size, stride=1, 
                            dilation=2**i, padding=(kernel_size-1)*2**i, dropout=dropout)
            )
            in_channels = out_channels
        self.segment_classifier = nn.Sequential(
            nn.Linear(num_channels[-1] * 3, num_channels[-1] // 2),  
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_channels[-1] // 2, 3)  
        )
        if mode == 'classification' and num_classes:
            self.classifier = nn.Sequential(
                nn.Linear(num_channels[-1] * 3, num_channels[-1] // 2),  
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(num_channels[-1] // 2, num_classes)
            )
    def compute_delta_features(self, x):
        """Delta features 계산"""
        batch_size, seq_len, features = x.shape
        delta_x = torch.zeros_like(x)
        delta_x[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
        combined = torch.cat([x, delta_x], dim=-1)
        return combined
    def segment_aware_pooling(self, x):
        """세그먼트별 통계 + 어텐션 풀링"""
        batch_size, seq_len, features = x.shape
        segment_length = seq_len // 3
        segments = []
        for i in range(3):
            start_idx = i * segment_length
            end_idx = start_idx + segment_length if i < 2 else seq_len
            segment = x[:, start_idx:end_idx, :]
            segments.append(segment)
        segment_stats = []
        for segment in segments:
            mean_feat = torch.mean(segment, dim=1)
            max_feat = torch.max(segment, dim=1)[0]
            std_feat = torch.std(segment, dim=1)
            combined = torch.cat([mean_feat, max_feat, std_feat], dim=-1)
            segment_stats.append(combined)
        segment_embeddings = torch.stack(segment_stats, dim=1)  
        segment_attention = F.softmax(
            torch.matmul(segment_embeddings, segment_embeddings.transpose(-2, -1)) / 
            (segment_embeddings.size(-1) ** 0.5), dim=-1
        )
        attended_segments = torch.matmul(segment_attention, segment_embeddings)
        segment_states = self.segment_classifier(attended_segments)  
        final_features = attended_segments.mean(dim=1)  
        return final_features, segment_states
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        if self.use_delta_features:
            delta_features = self.compute_delta_features(x)
            delta_encoded = self.delta_encoder(delta_features)
            if hasattr(self, 'input_proj'):
                original_encoded = self.input_proj(x)
            else:
                original_encoded = delta_encoded
            fused_features = self.feature_fusion(
                torch.cat([original_encoded, delta_encoded], dim=-1)
            )
        else:
            fused_features = self.input_proj(x)
        x_conv = fused_features.transpose(1, 2)  
        branch_outputs = []
        for branch in self.rf_branches:
            branch_out = branch(x_conv)
            branch_outputs.append(branch_out.transpose(1, 2))  
        branch_concat = torch.stack(branch_outputs, dim=1)  
        batch_size, num_branches, seq_len, channels = branch_concat.shape
        branch_concat_reshaped = branch_concat.view(batch_size * num_branches, seq_len, channels)
        attended_branches, _ = self.branch_attention(
            branch_concat_reshaped, branch_concat_reshaped, branch_concat_reshaped
        )
        attended_branches = attended_branches.view(batch_size, num_branches, seq_len, channels)
        branch_weights = F.softmax(torch.randn(batch_size, num_branches, 1, 1, device=x.device), dim=1)
        weighted_sum = torch.sum(attended_branches * branch_weights, dim=1)  
        if self.use_deformable:
            x_deform = weighted_sum.transpose(1, 2)  
            offset = self.offset_conv(x_deform)  
            x_deform = self.deformable_conv(x_deform)
            weighted_sum = x_deform.transpose(1, 2)  
        x_tcn = weighted_sum
        for tcn_layer in self.tcn_layers:
            x_tcn = tcn_layer(x_tcn.transpose(1, 2)).transpose(1, 2)
        final_features, segment_states = self.segment_aware_pooling(x_tcn)
        if self.mode == 'classification' and self.num_classes:
            output = self.classifier(final_features)
            return output, final_features
        else:
            return final_features, final_features
def create_model(
    model_type,
    input_dim=5,
    num_classes=None,
    mode='classification',
    commitment_cost=0.01,
    num_embeddings=128,
    embedding_dim=64,
    tcn_channels=None,
    kernel_size=5,
    dropout=0.15,
    num_heads=12,
    transformer_layers=3,
    ms_use_delta_features=True,
    ms_use_deformable=True,
    ms_rf_branches='1/4,1,4',
):
    if model_type == 'cnn_lstm':
        return CNNLSTMTrackNet(input_dim=input_dim, hidden_dim=128, num_layers=2, 
                               num_classes=num_classes, mode=mode)
    elif model_type == 'original_lstm':
        return OriginalLSTM(input_dim=input_dim, hidden_dim=128, num_layers=2, 
                           num_classes=num_classes, mode=mode)
    elif model_type == 'vq_lstm':
        return VQLSTM(input_dim=input_dim, hidden_dim=128, num_layers=2, 
                     num_embeddings=num_embeddings, embedding_dim=embedding_dim, 
                     commitment_cost=commitment_cost, num_classes=num_classes, mode=mode)
    elif model_type == 'vq_bottleneck_lstm':
        return VQBottleneckLSTM(input_dim=input_dim, hidden_dim=16, num_layers=1,
                               num_embeddings=num_embeddings, embedding_dim=embedding_dim,
                               commitment_cost=commitment_cost, num_classes=num_classes, mode=mode)
    elif model_type == 'dual_stream_lstm':
        return DualStreamLSTM(input_dim=input_dim, hidden_dim=64, num_layers=2,
                             num_embeddings=num_embeddings, embedding_dim=embedding_dim,
                             commitment_cost=commitment_cost, num_classes=num_classes, mode=mode)
    elif model_type == 'tcn':
        channels = tcn_channels if tcn_channels is not None else [32, 64, 128]
        return TCN(input_dim=input_dim, num_channels=channels, kernel_size=kernel_size, dropout=dropout,
                   num_classes=num_classes, mode=mode)
    elif model_type == 'dilated_residual_tcn':
        channels = tcn_channels if tcn_channels is not None else [32, 64, 128]
        return DilatedResidualTCN(input_dim=input_dim, num_channels=channels, kernel_size=kernel_size, dropout=dropout,
                                 num_classes=num_classes, mode=mode)
    elif model_type == 'attention_tcn':
        channels = tcn_channels if tcn_channels is not None else [32, 64, 128]
        return AttentionTCN(input_dim=input_dim, num_channels=channels, kernel_size=kernel_size, dropout=dropout,
                           num_heads=num_heads, num_classes=num_classes, mode=mode)
    elif model_type == 'tcn_transformer':
        channels = tcn_channels if tcn_channels is not None else [32, 64, 128]
        return TCNTransformer(input_dim=input_dim, num_channels=channels, kernel_size=kernel_size, dropout=dropout,
                             num_heads=num_heads, num_layers=transformer_layers, num_classes=num_classes, mode=mode)
    elif model_type == 'graph_tcn':
        channels = tcn_channels if tcn_channels is not None else [32, 64, 128]
        return GraphTCN(input_dim=input_dim, num_channels=channels, kernel_size=kernel_size, dropout=dropout,
                       num_classes=num_classes, mode=mode)
    elif model_type == 'ms_tcn_rf':
        channels = tcn_channels if tcn_channels is not None else [48, 96, 192]
        return MSTCNRF(input_dim=input_dim, num_channels=channels, kernel_size=kernel_size, dropout=dropout,
                      num_classes=num_classes, mode=mode, 
                      use_delta_features=ms_use_delta_features, use_deformable=ms_use_deformable,
                      rf_branches=ms_rf_branches)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
