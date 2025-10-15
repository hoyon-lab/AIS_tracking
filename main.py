import os
import csv
import argparse
import pickle
from re import T
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from src.models.va_vae_model import VAVAE
from src.models.lstm_models import create_model, TripletLoss, ContrastiveLoss, FocalLoss
class UnifiedAISTrackDataset(Dataset):
    def __init__(self, csv_path, min_len=50, fixed_len=50, mode='classification'):
        df = pd.read_csv(csv_path)
        if mode == 'anomaly':
            feature_cols = ["LAT", "LON", "SOG", "COG", "Heading", "WDIR", "WSPD", "GST", "PRES", "ATMP", "WTMP"]
        else:
            feature_cols = ["LAT", "LON", "SOG", "COG", "Heading"]
        scaler = MinMaxScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        self.sequences = []
        self.labels = []
        self.mode = mode
        if mode == 'classification':
            mmsi_to_label = {}
            current_label = 0
            for mmsi, group in df.groupby("MMSI"):
                group = group.sort_values("BaseDateTime")
                if len(group) < min_len:
                    continue
                if mmsi not in mmsi_to_label:
                    mmsi_to_label[mmsi] = current_label
                    current_label += 1
                data = group[feature_cols].values
                for i in range(0, len(data) - fixed_len + 1):
                    self.sequences.append(data[i:i + fixed_len])
                    self.labels.append(mmsi_to_label[mmsi])
            self.sequences = torch.tensor(self.sequences, dtype=torch.float32)
            self.labels = torch.tensor(self.labels, dtype=torch.long)
        else:
            mmsi_groups = df.groupby("MMSI")
            for _, group in mmsi_groups:
                group = group.sort_values(by="BaseDateTime")
                if len(group) < min_len:
                    continue
                data = group[feature_cols].values
                for i in range(0, len(data) - fixed_len + 1):
                    self.sequences.append(data[i:i + fixed_len])
            self.sequences = torch.tensor(self.sequences, dtype=torch.float32)
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        if self.mode == 'classification':
            return self.sequences[idx], self.labels[idx]
        else:
            return self.sequences[idx]
def classification_train(model, train_loader, optimizer, criterion, device, model_type, vq_loss_weight, epoch=0, args=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    total_vq_loss = 0
    total_perplexity = 0
    if epoch == 1:
        print(f"🎯 사용 중인 Loss Function: {args.loss_type.upper()}")
        if args.loss_type == 'combined':
            print(f"   📊 Combined Loss Weight: {args.combined_weight}")
            print(f"   🔗 Triplet Loss Margin: {args.triplet_margin}")
            print(f"   🌡️ Contrastive Loss Temperature: {args.contrastive_temperature}")
        elif args.loss_type == 'triplet':
            print(f"   🔗 Triplet Loss Margin: {args.triplet_margin}")
        elif args.loss_type == 'contrastive':
            print(f"   🌡️ Contrastive Loss Temperature: {args.contrastive_temperature}")
        elif args.loss_type == 'focal':
            print(f"   🎯 Focal Loss Gamma: {args.focal_gamma}")
        if args.loss_type == 'combined':
            print(f"   📈 Cross Entropy Weight: {1 - args.combined_weight}")
        elif args.loss_type == 'cross_entropy':
            print(f"   📈 Cross Entropy Weight: 1.0")
        else:
            print(f"   📈 Cross Entropy Weight: 0.0 (사용되지 않음)")
        print()
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
    if args.loss_type == 'triplet':
        triplet_loss = TripletLoss(margin=args.triplet_margin)
    elif args.loss_type == 'contrastive':
        contrastive_loss = ContrastiveLoss(temperature=args.contrastive_temperature)
    elif args.loss_type == 'combined':
        triplet_loss = TripletLoss(margin=args.triplet_margin)
        contrastive_loss = ContrastiveLoss(temperature=args.contrastive_temperature)
    optimizer.zero_grad()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        if args.use_amp:
            with torch.cuda.amp.autocast():
                if model_type in ['vq_lstm', 'vq_bottleneck_lstm', 'dual_stream_lstm']:
                    out, features, vq_loss, perplexity = model(data)
                    cls_loss = criterion(out, target)
                    current_vq_loss_weight = vq_loss_weight * min(1.0, (epoch / args.vq_warmup_epochs)) if args.vq_warmup_epochs > 0 else vq_loss_weight
                    if torch.isnan(vq_loss) or torch.isinf(vq_loss) or vq_loss > 50:
                        print(f"⚠️ VQ Loss 이상 감지: {vq_loss.item()}, VQ loss를 0으로 설정")
                        vq_loss = torch.tensor(0.0, device=device, requires_grad=True)
                    vq_loss = torch.clamp(vq_loss, max=20.0)
                    if args.loss_type == 'cross_entropy':
                        loss = cls_loss + current_vq_loss_weight * vq_loss
                    elif args.loss_type == 'triplet':
                        batch_size = features.size(0)
                        if batch_size >= 3:
                            anchor_idx = torch.randint(0, batch_size, (batch_size,))
                            positive_idx = torch.randint(0, batch_size, (batch_size,))
                            negative_idx = torch.randint(0, batch_size, (batch_size,))
                            for i in range(batch_size):
                                same_class = (target == target[i]).nonzero(as_tuple=True)[0]
                                diff_class = (target != target[i]).nonzero(as_tuple=True)[0]
                                if len(same_class) > 1:
                                    positive_idx[i] = same_class[torch.randint(0, len(same_class), (1,))]
                                if len(diff_class) > 0:
                                    negative_idx[i] = diff_class[torch.randint(0, len(diff_class), (1,))]
                            triplet_loss_val = triplet_loss(features[anchor_idx], features[positive_idx], features[negative_idx])
                            loss = triplet_loss_val + current_vq_loss_weight * vq_loss
                        else:
                            loss = current_vq_loss_weight * vq_loss
                    elif args.loss_type == 'contrastive':
                        contrastive_loss_val = contrastive_loss(features, target)
                        loss = contrastive_loss_val + current_vq_loss_weight * vq_loss
                    elif args.loss_type == 'combined':
                        triplet_loss_val = triplet_loss(features, features, features)  
                        contrastive_loss_val = contrastive_loss(features, target)
                        combined_loss_val = triplet_loss_val + contrastive_loss_val
                        loss = (1 - args.combined_weight) * cls_loss + args.combined_weight * combined_loss_val + current_vq_loss_weight * vq_loss
                    else:
                        loss = cls_loss + current_vq_loss_weight * vq_loss
                    total_vq_loss += vq_loss.item()
                    total_perplexity += perplexity.item()
                else:
                    out, features = model(data)
                    if args.loss_type == 'cross_entropy':
                        loss = criterion(out, target)
                    elif args.loss_type == 'triplet':
                        batch_size = features.size(0)
                        if batch_size >= 3:
                            anchor_idx = torch.randint(0, batch_size, (batch_size,))
                            positive_idx = torch.randint(0, batch_size, (batch_size,))
                            negative_idx = torch.randint(0, batch_size, (batch_size,))
                            for i in range(batch_size):
                                same_class = (target == target[i]).nonzero(as_tuple=True)[0]
                                diff_class = (target != target[i]).nonzero(as_tuple=True)[0]
                                if len(same_class) > 1:
                                    positive_idx[i] = same_class[torch.randint(0, len(same_class), (1,))]
                                if len(diff_class) > 0:
                                    negative_idx[i] = diff_class[torch.randint(0, len(diff_class), (1,))]
                            triplet_loss_val = triplet_loss(features[anchor_idx], features[positive_idx], features[negative_idx])
                            loss = triplet_loss_val
                        else:
                            loss = torch.tensor(0.0, device=device, requires_grad=True)
                    elif args.loss_type == 'contrastive':
                        contrastive_loss_val = contrastive_loss(features, target)
                        loss = contrastive_loss_val
                    elif args.loss_type == 'combined':
                        cls_loss = criterion(out, target)
                        triplet_loss_val = triplet_loss(features, features, features)
                        contrastive_loss_val = contrastive_loss(features, target)
                        combined_loss_val = triplet_loss_val + contrastive_loss_val
                        loss = (1 - args.combined_weight) * cls_loss + args.combined_weight * combined_loss_val
                    else:
                        loss = criterion(out, target)
                loss = loss / args.accumulation_steps
                scaler.scale(loss).backward()
                if (batch_idx + 1) % args.accumulation_steps == 0:
                    if args.grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
        else:
            if model_type in ['vq_lstm', 'vq_bottleneck_lstm', 'dual_stream_lstm']:
                out, features, vq_loss, perplexity = model(data)
                cls_loss = criterion(out, target)
                current_vq_loss_weight = vq_loss_weight * min(1.0, (epoch / args.vq_warmup_epochs)) if args.vq_warmup_epochs > 0 else vq_loss_weight
                if torch.isnan(vq_loss) or torch.isinf(vq_loss) or vq_loss > 50:
                    print(f"⚠️ VQ Loss 이상 감지: {vq_loss.item()}, VQ loss를 0으로 설정")
                    vq_loss = torch.tensor(0.0, device=device, requires_grad=True)
                vq_loss = torch.clamp(vq_loss, max=20.0)
                if args.loss_type == 'cross_entropy':
                    loss = cls_loss + current_vq_loss_weight * vq_loss
                elif args.loss_type == 'triplet':
                    batch_size = features.size(0)
                    if batch_size >= 3:
                        anchor_idx = torch.randint(0, batch_size, (batch_size,))
                        positive_idx = torch.randint(0, batch_size, (batch_size,))
                        negative_idx = torch.randint(0, batch_size, (batch_size,))
                        for i in range(batch_size):
                            same_class = (target == target[i]).nonzero(as_tuple=True)[0]
                            diff_class = (target != target[i]).nonzero(as_tuple=True)[0]
                            if len(same_class) > 1:
                                positive_idx[i] = same_class[torch.randint(0, len(same_class), (1,))]
                            if len(diff_class) > 0:
                                negative_idx[i] = diff_class[torch.randint(0, len(diff_class), (1,))]
                        triplet_loss_val = triplet_loss(features[anchor_idx], features[positive_idx], features[negative_idx])
                        loss = cls_loss + current_vq_loss_weight * vq_loss + triplet_loss_val
                    else:
                        loss = cls_loss + current_vq_loss_weight * vq_loss
                elif args.loss_type == 'contrastive':
                    contrastive_loss_val = contrastive_loss(features, target)
                    loss = cls_loss + current_vq_loss_weight * vq_loss + contrastive_loss_val
                elif args.loss_type == 'combined':
                    triplet_loss_val = triplet_loss(features, features, features)  
                    contrastive_loss_val = contrastive_loss(features, target)
                    combined_loss_val = triplet_loss_val + contrastive_loss_val
                    loss = (1 - args.combined_weight) * cls_loss + args.combined_weight * combined_loss_val + current_vq_loss_weight * vq_loss
                else:
                    loss = cls_loss + current_vq_loss_weight * vq_loss
                total_vq_loss += vq_loss.item()
                total_perplexity += perplexity.item()
            else:
                out, features = model(data)
                if args.loss_type == 'cross_entropy':
                    loss = criterion(out, target)
                elif args.loss_type == 'triplet':
                    cls_loss = criterion(out, target)
                    batch_size = features.size(0)
                    if batch_size >= 3:
                        anchor_idx = torch.randint(0, batch_size, (batch_size,))
                        positive_idx = torch.randint(0, batch_size, (batch_size,))
                        negative_idx = torch.randint(0, batch_size, (batch_size,))
                        for i in range(batch_size):
                            same_class = (target == target[i]).nonzero(as_tuple=True)[0]
                            diff_class = (target != target[i]).nonzero(as_tuple=True)[0]
                            if len(same_class) > 1:
                                positive_idx[i] = same_class[torch.randint(0, len(same_class), (1,))]
                            if len(diff_class) > 0:
                                negative_idx[i] = diff_class[torch.randint(0, len(diff_class), (1,))]
                        triplet_loss_val = triplet_loss(features[anchor_idx], features[positive_idx], features[negative_idx])
                        loss = cls_loss + triplet_loss_val
                    else:
                        loss = criterion(out, target)
                elif args.loss_type == 'contrastive':
                    cls_loss = criterion(out, target)
                    contrastive_loss_val = contrastive_loss(features, target)
                    if torch.isnan(contrastive_loss_val) or torch.isinf(contrastive_loss_val):
                        print(f"⚠️ ContrastiveLoss가 nan/inf: {contrastive_loss_val}, cls_loss만 사용")
                        loss = cls_loss
                    else:
                        loss = cls_loss + contrastive_loss_val
                elif args.loss_type == 'combined':
                    cls_loss = criterion(out, target)
                    triplet_loss_val = triplet_loss(features, features, features)
                    contrastive_loss_val = contrastive_loss(features, target)
                    if torch.isnan(triplet_loss_val) or torch.isinf(triplet_loss_val):
                        print(f"⚠️ TripletLoss가 nan/inf: {triplet_loss_val}, 0으로 대체")
                        triplet_loss_val = torch.tensor(0.0, device=features.device, requires_grad=True)
                    if torch.isnan(contrastive_loss_val) or torch.isinf(contrastive_loss_val):
                        print(f"⚠️ ContrastiveLoss가 nan/inf: {contrastive_loss_val}, 0으로 대체")
                        contrastive_loss_val = torch.tensor(0.0, device=features.device, requires_grad=True)
                    combined_loss_val = triplet_loss_val + contrastive_loss_val
                    loss = (1 - args.combined_weight) * cls_loss + args.combined_weight * combined_loss_val
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"⚠️ 최종 loss가 nan/inf: {loss}, cls_loss만 사용")
                        loss = cls_loss
                else:
                    loss = criterion(out, target)
            loss = loss / args.accumulation_steps
            loss.backward()
            if (batch_idx + 1) % args.accumulation_steps == 0:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
        total_loss += loss.item() * args.accumulation_steps
        pred = out.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(train_loader)
    if model_type in ['vq_lstm', 'vq_bottleneck_lstm', 'dual_stream_lstm']:
        avg_vq_loss = total_vq_loss / len(train_loader)
        avg_perplexity = total_perplexity / len(train_loader)
        print(f"[Classification] Loss: {avg_loss:.4f} | Accuracy: {accuracy/100:.4f} | VQ Loss: {avg_vq_loss:.4f} | Perplexity: {avg_perplexity:.2f}")
    else:
        print(f"[Classification] Loss: {avg_loss:.4f} | Accuracy: {accuracy/100:.4f}")
    return avg_loss, accuracy/100
def classification_eval(model, loader, device, num_classes, model_type='cnn_lstm'):
    model.eval()
    all_preds, all_labels, all_features = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            if model_type == 'vq_lstm':
                out, features, _, _ = model(x)
            elif model_type in ['vq_bottleneck_lstm', 'dual_stream_lstm']:
                out, features, _, _ = model(x)
            elif model_type == 'ms_tcn_rf':
                out, features = model(x)
            else:
                out, features = model(x)
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
            all_features.extend(features.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro')
    rec = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"\n✅ 분류 성능 결과")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    import numpy as np
    import seaborn as sns
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix 원본:")
    print(cm)
    print("예측 분포:", np.bincount(np.array(all_preds)))
    print("실제 분포:", np.bincount(np.array(all_labels)))
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap='Reds', 
                cbar_kws={'label': 'Count'})
    plt.title("Confusion Matrix - Vessel Classification", fontsize=16, fontweight='bold')
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.show()
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
    print("정규화된 Confusion Matrix:")
    print(cm_normalized)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=False, cmap='Reds', 
                cbar_kws={'label': 'Normalized Count'})
    plt.title("Confusion Matrix (Normalized) - Vessel Tracking Association", fontsize=16, fontweight='bold')
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("confusion_matrix_normalized.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("t-SNE 임베딩 시각화 중...")
    all_features = np.array(all_features)
    z_2d = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(all_features)
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=all_labels, cmap='tab20', s=5)
    plt.colorbar(scatter, label="MMSI Label")
    plt.title("t-SNE of Latent Embeddings - 선박 분류")
    plt.tight_layout()
    plt.savefig("tsne_classification.png")
    plt.show()
def vae_loss(x_recon, x, mu, logvar, beta=1.0):
    recon = F.mse_loss(x_recon, x, reduction='mean')
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl, recon, kl
def vae_train(model, train_loader, val_loader, optimizer, device, total_epochs=100, beta=0.1, log_path="logs/vae_train_log.csv"):
    model.train()
    os.makedirs("logs", exist_ok=True)
    with open(log_path, 'w', newline='') as logfile:
        writer = csv.writer(logfile)
        writer.writerow(["Epoch", "TrainLoss", "ValLoss", "ReconLoss", "KLLoss"])
        for epoch in range(1, total_epochs + 1):
            total_loss, total_recon, total_kl = 0, 0, 0
            model.train()
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                x_recon, mu, logvar = model(batch)
                loss, recon, kl = vae_loss(x_recon, batch, mu, logvar, beta)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_recon += recon.item()
                total_kl += kl.item()
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for val_batch in val_loader:
                    val_batch = val_batch.to(device)
                    x_recon, mu, logvar = model(val_batch)
                    v_loss, _, _ = vae_loss(x_recon, val_batch, mu, logvar, beta)
                    val_loss += v_loss.item()
            avg_train_loss = total_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            print(f"[VAE Epoch {epoch}] TrainLoss: {avg_train_loss:.4f} | ValLoss: {avg_val_loss:.4f} | Recon: {total_recon:.4f} | KL: {total_kl:.4f}")
            writer.writerow([epoch, avg_train_loss, avg_val_loss, total_recon, total_kl])
            if epoch % 10 == 0 or epoch == total_epochs:
                torch.save(model.state_dict(), f"checkpoints/vae_epoch_{epoch}.pt")
def vae_test(model, test_loader, device):
    model.eval()
    all_errors = []
    all_mu = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            x_recon, mu, logvar = model(batch)
            recon_error = ((x_recon - batch) ** 2).mean(dim=(1, 2))
            all_errors.extend(recon_error.cpu().numpy())
            all_mu.extend(mu.cpu().numpy())
    all_errors = np.array(all_errors)
    all_mu = np.array(all_mu)
    threshold = np.percentile(all_errors, 95)
    predicted = (all_errors > threshold).astype(int)
    true_labels = np.zeros_like(predicted)
    true_labels[all_errors > threshold] = 1
    acc = accuracy_score(true_labels, predicted)
    prec = precision_score(true_labels, predicted)
    rec = recall_score(true_labels, predicted)
    f1 = f1_score(true_labels, predicted)
    print(f"\n✅ 이상 탐지 결과")
    print(f"탐지된 이상: {predicted.sum()} / {len(predicted)} 샘플")
    print(f"임계값 (95%): {threshold:.4f}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    plt.figure(figsize=(8, 5))
    plt.hist(all_errors, bins=100)
    plt.axvline(threshold, color='r', linestyle='--', label='임계값')
    plt.title("재구성 오차 분포 - 이상 탐지")
    plt.xlabel("MSE")
    plt.ylabel("빈도")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("recon_error_hist.png")
    plt.show()
    print("t-SNE 잠재 공간 시각화 중...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    z_2d = tsne.fit_transform(all_mu)
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=all_errors, cmap='viridis', s=5)
    plt.colorbar(scatter, label="재구성 오차")
    plt.title("잠재 공간 (t-SNE) - 이상 탐지")
    plt.xlabel("Z1")
    plt.ylabel("Z2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("tsne_anomaly.png")
    plt.show()
def get_split_indices(dataset, path="split_indices.pkl"):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            train_idx, val_idx, test_idx = pickle.load(f)
        print("✅ 기존 분할 인덱스 로드 완료")
    else:
        total_len = len(dataset)
        indices = np.arange(total_len)
        np.random.seed(42)
        np.random.shuffle(indices)
        train_len = int(total_len * 0.7)
        val_len = int(total_len * 0.1)
        test_len = total_len - train_len - val_len
        train_idx = indices[:train_len]
        val_idx = indices[train_len:train_len+val_len]
        test_idx = indices[train_len+val_len:]
        with open(path, 'wb') as f:
            pickle.dump((train_idx, val_idx, test_idx), f)
        print("💾 새로운 분할 인덱스 저장 완료")
    return train_idx, val_idx, test_idx
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AIS Track Analysis 시스템')
    parser.add_argument('--mode', choices=['classification', 'anomaly'], default='classification',
                       help='분석 모드: classification(분류, 기본값), anomaly(이상탐지)')
    parser.add_argument('--model_type', choices=['cnn_lstm', 'original_lstm', 'vq_lstm', 'vq_bottleneck_lstm', 'dual_stream_lstm', 'tcn', 'dilated_residual_tcn', 'attention_tcn', 'tcn_transformer', 'graph_tcn', 'ms_tcn_rf'], 
                        default='dilated_residual_tcn',
                        help='모델 타입: cnn_lstm, original_lstm, vq_lstm, vq_bottleneck_lstm, dual_stream_lstm, tcn, dilated_residual_tcn, attention_tcn, tcn_transformer, graph_tcn, ms_tcn_rf')
    parser.add_argument('--csv_path', type=str, default="combined_output.csv",
                       help='AIS 데이터 CSV 파일 경로')
    parser.add_argument('--epochs', type=int, default=80, help='학습 에포크 수')
    parser.add_argument('--beta', type=float, default=0.1, help='VAE KL 손실 가중치')
    parser.add_argument('--lr', type=float, default=3e-3, help='학습률')  
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')  
    parser.add_argument('--grad_clip', type=float, default=5.0,  
                        help='Gradient clipping value (default: 5.0)')
    parser.add_argument('--label_smoothing', type=float, default=0.05, help='Label smoothing for CE loss')  
    parser.add_argument('--scheduler_warmup_epochs', type=int, default=5, help='LR warmup epochs')  
    parser.add_argument('--scheduler_type', choices=['cosine', 'cosine_restart', 'step', 'plateau'], default='cosine',
                        help='학습률 스케줄러 타입: cosine, cosine_restart, step, plateau')
    parser.add_argument('--min_lr_ratio', type=float, default=0.6, help='최소 학습률 비율 (기본값: 0.6)')  
    parser.add_argument('--restart_period', type=int, default=20, help='cosine_restart 주기(epoch)')  
    parser.add_argument('--early_stopping_patience', type=int, default=20, help='조기 종료 patience (기본값: 20)')  
    parser.add_argument('--loss_type', choices=['cross_entropy', 'focal', 'triplet', 'contrastive', 'combined'], default='combined',
                        help='손실 함수 타입: cross_entropy, triplet, contrastive, combined')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal loss gamma')
    parser.add_argument('--class_weight', type=str, default=None, help='클래스 가중치 리스트 예: 1.0,0.8,1.2')
    parser.add_argument('--triplet_margin', type=float, default=1.0, help='Triplet loss margin')
    parser.add_argument('--contrastive_temperature', type=float, default=0.07, help='Contrastive loss temperature')
    parser.add_argument('--combined_weight', type=float, default=0.35, help='Combined loss weight (CE vs triplet/contrastive)')
    parser.add_argument('--tcn_channels', type=str, default='48,96,192', help='예: 48,96,192 (기본값: 48,96,192)')
    parser.add_argument('--tcn_kernel_size', type=int, default=5)
    parser.add_argument('--tcn_dropout', type=float, default=0.15) 
    parser.add_argument('--tcn_heads', type=int, default=12)
    parser.add_argument('--tcn_transformer_layers', type=int, default=3)
    parser.add_argument('--ms_use_delta_features', action='store_true', default=True,
                        help='MS-TCN-RF에서 Delta features + 시간간격 임베딩 사용 (기본값: True)')
    parser.add_argument('--ms_use_deformable', action='store_true', default=True,
                        help='MS-TCN-RF에서 Deformable Temporal Convolution 사용 (기본값: True)')
    parser.add_argument('--ms_rf_branches', type=str, default='0.25,1,4',
                        help='MS-TCN-RF의 Receptive Field 브랜치 비율 (기본값: 0.25,1,4)')
    parser.add_argument('--vq_commitment_cost', type=float, default=0.01,  
                        help='VQ commitment cost (default: 0.01)')
    parser.add_argument('--vq_loss_weight', type=float, default=0.001,  
                        help='VQ loss weight (default: 0.001)')
    parser.add_argument('--vq_warmup_epochs', type=int, default=10,  
                        help='VQ loss weight warmup epochs (default: 10)')
    parser.add_argument('--num_embeddings', type=int, default=16, help='VQ codebook size')  
    parser.add_argument('--embedding_dim', type=int, default=8, help='VQ embedding dimension')  
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto', help='연산 디바이스 선택 (auto/cuda/cpu)')
    parser.add_argument('--split_path', type=str, default='split_indices.pkl',
                       help='분할 인덱스 파일 경로')
    parser.add_argument('--test_only', action='store_true', default=False,
                       help='체크포인트가 있으면 학습 없이 바로 테스트/시각화만 실행 (체크포인트 우선순위: best > final)')
    parser.add_argument('--batch_size', type=int, default=32, help='배치 크기')
    parser.add_argument('--use_amp', action='store_true', help='Automatic Mixed Precision 사용')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    args = parser.parse_args()
    import platform
    import torch
    want_cuda = (args.device == 'cuda') or (args.device == 'auto' and torch.cuda.is_available())
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA를 강제했지만 torch.cuda.is_available() == False 입니다. CUDA 드라이버/빌드 확인 필요.')
    device = torch.device('cuda' if want_cuda else 'cpu')
    print(f"🚀 디바이스: {device}")
    print(f"🔢 torch: {torch.__version__}")
    print(f"🧩 torch.cuda.build: {getattr(torch.version, 'cuda', None)} | cuda_available: {torch.cuda.is_available()} | devices: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        try:
            print(f"🖥️ cuda[0]: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass
    print(f"🐍 python: {platform.python_version()}")
    print(f"📊 모드: {args.mode}")
    print(f"🧠 모델: {args.model_type}")
    if args.test_only:
        print("🧪 TEST_ONLY 모드가 활성화되었습니다.")
        print("   - 학습을 건너뛰고 기존 체크포인트로 테스트만 실행합니다.")
        print("   - 체크포인트 우선순위: best > final")
        print("   - 체크포인트가 없으면 오류 메시지와 함께 종료됩니다.")
        print()
    else:
        print("🏋️ 학습 모드가 활성화되었습니다.")
        print("   - 체크포인트가 있으면 로드 후 학습을 계속합니다.")
        print("   - 체크포인트가 없으면 처음부터 학습을 시작합니다.")
        print()
    if args.mode == 'anomaly':
        dataset = UnifiedAISTrackDataset(args.csv_path, mode='anomaly')
        train_idx, val_idx, test_idx = get_split_indices(dataset, args.split_path)
        train_set = Subset(dataset, train_idx)
        val_set = Subset(dataset, val_idx)
        test_set = Subset(dataset, test_idx)
        train_loader = DataLoader(train_set, batch_size=32, shuffle=True, pin_memory=torch.cuda.is_available())
        val_loader = DataLoader(val_set, batch_size=32, shuffle=False, pin_memory=torch.cuda.is_available())
        test_loader = DataLoader(test_set, batch_size=32, shuffle=False, pin_memory=torch.cuda.is_available())
        model = VAVAE(input_dim=11, hidden_dim=64, latent_dim=32).to(device)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"🔍 test 시작...")
        print(f"모델 파라미터 수: {param_count:,}")
        checkpoint_path = "checkpoints/dilated_residual_tcn_best.pt"
        best_checkpoint_path = "checkpoints/dilated_residual_tcn_best.pt"
        if args.test_only:
            print("🧪 TEST_ONLY 모드: 체크포인트 로드 후 테스트/시각화만 실행합니다.")
            if os.path.exists(best_checkpoint_path):
                print(f"✅ 최고 성능 체크포인트 발견: {best_checkpoint_path}")
                checkpoint_to_load = best_checkpoint_path
            elif os.path.exists(checkpoint_path):
                print(f"✅ 일반 VAE 체크포인트 발견: {checkpoint_path}")
                checkpoint_to_load = checkpoint_path
            else:
                print(f"❌ 체크포인트 파일을 찾을 수 없습니다!")
                print(f"   찾는 경로:")
                print(f"   - 최고 성능: {best_checkpoint_path}")
                print(f"   - 일반: {checkpoint_path}")
                print(f"   현재 디렉토리: {os.getcwd()}")
                print(f"   checkpoints 폴더 내용:")
                if os.path.exists("checkpoints"):
                    for file in os.listdir("checkpoints"):
                        if file.endswith('.pt'):
                            print(f"     - {file}")
                else:
                    print("     checkpoints 폴더가 존재하지 않습니다.")
                sys.exit(1)
            try:
                checkpoint = torch.load(checkpoint_to_load, map_location=device)
                model.load_state_dict(checkpoint)
                print(f"✅  체크포인트 로드 완료: {checkpoint_to_load}")
                print(f"📊 모델 정보:")
                print(f"   - 입력 차원: 11 (LAT, LON, SOG, COG, Heading, WDIR, WSPD, GST, PRES, ATMP, WTMP)")
                print(f"   - Hidden 차원: 64")
                print(f"   - Latent 차원: 32")
                print(f"   - 파라미터 수: {param_count:,}")
                print(f"🧪 테스트 데이터셋 크기: {len(test_set)}")
                print(f"🔍 VAE 이상 탐지 평가 시작...")
                vae_test(model, test_loader, device)
                print("🎉 TEST_ONLY 모드 완료!")
                sys.exit(0)
            except Exception as e:
                print(f"❌ 체크포인트 로드 중 오류 발생: {e}")
                print("💡 체크포인트 파일이 손상되었을 수 있습니다.")
                sys.exit(1)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            os.makedirs("checkpoints", exist_ok=True)
            vae_train(model, train_loader, val_loader, optimizer, device, 
                      total_epochs=args.epochs, beta=args.beta)
            torch.save(model.state_dict(), checkpoint_path)
            print("🔍 VAE 이상 탐지 평가...")
            vae_test(model, test_loader, device)
    else:  
        dataset = UnifiedAISTrackDataset(args.csv_path, mode='classification')
        train_idx, val_idx, test_idx = get_split_indices(dataset, args.split_path)
        train_set = Subset(dataset, train_idx)
        val_set = Subset(dataset, val_idx)
        test_set = Subset(dataset, test_idx)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=torch.cuda.is_available())
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, pin_memory=torch.cuda.is_available())
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=torch.cuda.is_available())
        num_classes = len(set(dataset.labels.numpy().tolist()))
        model = create_model(
            model_type=args.model_type,
            input_dim=5,
            num_classes=num_classes,
            mode='classification',
            commitment_cost=args.vq_commitment_cost,
            num_embeddings=args.num_embeddings,
            embedding_dim=args.embedding_dim,
            tcn_channels=[int(x) for x in args.tcn_channels.split(',')] if args.tcn_channels else None,
            kernel_size=args.tcn_kernel_size,
            dropout=args.tcn_dropout,
            num_heads=args.tcn_heads,
            transformer_layers=args.tcn_transformer_layers,
            ms_use_delta_features=args.ms_use_delta_features,
            ms_use_deformable=args.ms_use_deformable,
            ms_rf_branches=args.ms_rf_branches
        ).to(device)
        param_count = sum(p.numel() for p in model.parameters())
        print("🏷️ 분류 모델 학습 시작...")
        print(f"모델 타입: {args.model_type}")
        print(f"모델 파라미터 수: {param_count:,}")
        if args.model_type.startswith('vq'):
            print(f"VQ Commitment Cost: {args.vq_commitment_cost}")
            print(f"VQ Loss Weight (target): {args.vq_loss_weight}")
            print(f"VQ Warmup Epochs: {args.vq_warmup_epochs}")
            print(f"VQ codebook: num_embeddings={args.num_embeddings}, embedding_dim={args.embedding_dim}")
        elif args.model_type == 'ms_tcn_rf':
            print(f"MS-TCN-RF Delta Features: {args.ms_use_delta_features}")
            print(f"MS-TCN-RF Deformable Conv: {args.ms_use_deformable}")
            print(f"MS-TCN-RF RF Branches: {args.ms_rf_branches}")
        checkpoint_path = f"checkpoints/{args.model_type}_classifier.pt"
        best_checkpoint_path = f"checkpoints/{args.model_type}_best.pt"
        if args.test_only:
            print("🧪 TEST_ONLY 모드: 체크포인트 로드 후 테스트/시각화만 실행합니다.")
            if os.path.exists(best_checkpoint_path):
                print(f"✅ 최고 성능 체크포인트 발견: {best_checkpoint_path}")
                checkpoint_to_load = best_checkpoint_path
            elif os.path.exists(checkpoint_path):
                print(f"✅ 일반 체크포인트 발견: {checkpoint_path}")
                checkpoint_to_load = checkpoint_path
            else:
                print(f"❌ 체크포인트 파일을 찾을 수 없습니다!")
                print(f"   찾는 경로:")
                print(f"   - 최고 성능: {best_checkpoint_path}")
                print(f"   - 일반: {checkpoint_path}")
                print(f"   현재 디렉토리: {os.getcwd()}")
                print(f"   checkpoints 폴더 내용:")
                if os.path.exists("checkpoints"):
                    for file in os.listdir("checkpoints"):
                        if file.endswith('.pt'):
                            print(f"     - {file}")
                else:
                    print("     checkpoints 폴더가 존재하지 않습니다.")
                print("💡 해결 방법:")
                print("   1. 먼저 학습을 실행하여 체크포인트를 생성하세요")
                print("   2. 또는 올바른 체크포인트 경로를 지정하세요")
                sys.exit(1)
            try:
                checkpoint = torch.load(checkpoint_to_load, map_location=device)
                model.load_state_dict(checkpoint)
                print(f"✅ 체크포인트 로드 완료: {checkpoint_to_load}")
                print(f"📊 모델 정보:")
                print(f"   - 타입: {args.model_type}")
                print(f"   - 입력 차원: 5 (LAT, LON, SOG, COG, Heading)")
                print(f"   - 클래스 수: {num_classes}")
                print(f"   - 파라미터 수: {param_count:,}")
                print(f"🧪 테스트 데이터셋 크기: {len(test_set)}")
                print(f"🔍 분류 모델 평가 시작...")
                classification_eval(model, test_loader, device, num_classes, args.model_type)
                print("🎉 TEST_ONLY 모드 완료!")
                sys.exit(0)
            except Exception as e:
                print(f"❌ 체크포인트 로드 중 오류 발생: {e}")
                print("💡 체크포인트 파일이 손상되었을 수 있습니다.")
                sys.exit(1)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            class_weight_tensor = None
            if args.class_weight is not None:
                try:
                    weights = [float(w) for w in args.class_weight.split(',')]
                    if len(weights) == num_classes:
                        class_weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
                except Exception:
                    pass
            if args.loss_type == 'focal':
                criterion = FocalLoss(gamma=args.focal_gamma, alpha=class_weight_tensor, reduction='mean')
            else:
                criterion = nn.CrossEntropyLoss(weight=class_weight_tensor, label_smoothing=args.label_smoothing)
            import math
            def get_scheduler_lambda(scheduler_type, warmup_epochs, total_epochs, min_lr_ratio):
                if scheduler_type == 'cosine':
                    def lr_lambda(current_epoch):
                        if current_epoch < warmup_epochs:
                            return (current_epoch + 1) / max(1, warmup_epochs)
                        progress = (current_epoch - warmup_epochs) / max(1, (total_epochs - warmup_epochs))
                        smooth_progress = progress ** 0.5
                        return min_lr_ratio + (1 - min_lr_ratio) * (1 + math.cos(math.pi * smooth_progress)) / 2.0
                    return lr_lambda
                elif scheduler_type == 'cosine_restart':
                    def lr_lambda(current_epoch):
                        if current_epoch < warmup_epochs:
                            return (current_epoch + 1) / max(1, warmup_epochs)
                        restart_epoch = max(1, args.restart_period)
                        cycle = (current_epoch - warmup_epochs) // restart_epoch
                        cycle_epoch = (current_epoch - warmup_epochs) % restart_epoch
                        progress = cycle_epoch / restart_epoch
                        flat = 0.4  
                        if progress < flat:
                            return 1.0
                        adj_progress = (progress - flat) / (1 - flat)
                        adj_progress = max(0.0, min(1.0, adj_progress))
                        smooth_adj_progress = adj_progress ** 0.5
                        return min_lr_ratio + (1 - min_lr_ratio) * (1 + math.cos(math.pi * smooth_adj_progress)) / 2.0
                    return lr_lambda
                elif scheduler_type == 'step':
                    def lr_lambda(current_epoch):
                        if current_epoch < warmup_epochs:
                            return (current_epoch + 1) / max(1, warmup_epochs)
                        step_size = 20
                        gamma = 0.8
                        steps = (current_epoch - warmup_epochs) // step_size
                        return max(min_lr_ratio, gamma ** steps)
                    return lr_lambda
                else:  
                    def lr_lambda(current_epoch):
                        if current_epoch < warmup_epochs:
                            return (current_epoch + 1) / max(1, warmup_epochs)
                        progress = (current_epoch - warmup_epochs) / max(1, (total_epochs - warmup_epochs))
                        smooth_progress = progress ** 0.5
                        return min_lr_ratio + (1 - min_lr_ratio) * (1 + math.cos(math.pi * smooth_progress)) / 2.0
                    return lr_lambda
            lr_lambda = get_scheduler_lambda(args.scheduler_type, args.scheduler_warmup_epochs, args.epochs, args.min_lr_ratio)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
            best_acc = 0.0
            patience = args.early_stopping_patience
            patience_counter = 0
            for epoch in range(1, args.epochs + 1):
                if args.model_type.startswith('vq'):
                    if epoch <= args.vq_warmup_epochs:
                        effective_vq_w = 0.0
                    else:
                        ramp = (epoch - args.vq_warmup_epochs) / max(1, (args.epochs - args.vq_warmup_epochs))
                        effective_vq_w = args.vq_loss_weight * min(1.0, max(0.0, ramp))
                else:
                    effective_vq_w = 0.0
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch}/{args.epochs} | LR: {current_lr:.6f} | VQ_w: {effective_vq_w:.4f}")
                train_loss, train_acc = classification_train(model, train_loader, optimizer, criterion, device, args.model_type, effective_vq_w, epoch, args)
                if current_lr < args.lr * 0.1:
                    print(f"⚠️ 학습률이 너무 작습니다: {current_lr:.6f} (초기값: {args.lr})")
                if epoch % 3 == 0:  
                    model.eval()
                    val_correct = 0
                    val_total = 0
                    with torch.no_grad():
                        for data, target in val_loader:
                            data, target = data.to(device), target.to(device)
                            if args.model_type in ['vq_lstm', 'vq_bottleneck_lstm', 'dual_stream_lstm']:
                                out, _, _, _ = model(data)
                            elif args.model_type == 'ms_tcn_rf':
                                out, _ = model(data)
                            else:
                                out, _ = model(data)
                            pred = out.argmax(dim=1, keepdim=True)
                            val_correct += pred.eq(target.view_as(pred)).sum().item()
                            val_total += target.size(0)
                    val_acc = val_correct / val_total
                    print(f"🔍 Validation Accuracy: {val_acc:.4f}")
                    if val_acc > best_acc:
                        best_acc = val_acc
                        patience_counter = 0
                        torch.save(model.state_dict(), f"checkpoints/{args.model_type}_best.pt")
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            print(f"🛑 {patience} 에포크 동안 개선이 없어 조기 종료합니다.")
                            break
                scheduler.step()
            if os.path.exists(f"checkpoints/{args.model_type}_best.pt"):
                model.load_state_dict(torch.load(f"checkpoints/{args.model_type}_best.pt", map_location=device))
                print(f"✅ 최고 성능 모델 로드 (Accuracy: {best_acc:.4f})")
            torch.save(model.state_dict(), checkpoint_path)
            print("🏷️ 분류 모델 평가...")
            classification_eval(model, test_loader, device, num_classes, args.model_type)
    print("🎉 모든 작업 완료!")
