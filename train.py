"""
train.py

模型架構：EfficientNet-B0（ImageNet 預訓練）當骨幹 + 座位（前/後）embedding，
兩者串接後接分類頭，二分類輸出（乾淨 / 需留意）。

處理正類樣本極少（~4%）的作法：
1. WeightedRandomSampler：訓練時對「需留意」樣本加權抽樣，讓每個 batch 都能見到足夠正類
2. 損失函數用 class weight 加權 CrossEntropy
3. 早停與模型選擇的依據是「需留意」類別的 recall（而非整體 accuracy），
   因為漏抓一台真的髒的車，代價遠高於誤判一台乾淨車

使用方式：
    python train.py --splits_dir ./splits --output_dir ./runs/exp1 --epochs 30
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, fbeta_score, roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models

from dataset import CarInteriorDataset


class SeatAwareClassifier(nn.Module):
    def __init__(self, num_seats: int = 2, seat_embed_dim: int = 8, freeze_backbone_until: int = 6):
        super().__init__()
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        backbone_out_dim = backbone.classifier[1].in_features  # 1280 for EfficientNet-B0

        # 凍結前段層，只微調後段（小資料集常見策略，避免過擬合）
        for i, block in enumerate(self.features):
            if i < freeze_backbone_until:
                for p in block.parameters():
                    p.requires_grad = False

        self.seat_embed = nn.Embedding(num_seats, seat_embed_dim)

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(backbone_out_dim + seat_embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(self, img, seat_idx):
        x = self.features(img)
        x = self.pool(x).flatten(1)
        s = self.seat_embed(seat_idx)
        x = torch.cat([x, s], dim=1)
        return self.classifier(x)


def build_weighted_sampler(dataset: CarInteriorDataset) -> WeightedRandomSampler:
    counts = dataset.label_counts()
    n_total = sum(counts.values())
    class_weight = {label: n_total / count for label, count in counts.items()}
    sample_weights = [class_weight[int(dataset.df.iloc[i]["label_id"])] for i in range(len(dataset))]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for imgs, seats, labels in loader:
            imgs, seats = imgs.to(device), seats.to(device)
            logits = model(imgs, seats)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    report = classification_report(
        all_labels, all_preds, target_names=["clean", "attention"], output_dict=True, zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")

    attention_recall = report["attention"]["recall"]
    attention_precision = report["attention"]["precision"]
    # F2：recall 權重是 precision 的 2 倍，符合「漏抓比誤判代價高」的業務考量，
    # 但仍會懲罰 precision 過低的退化模型（例如全部都猜 attention 的情況）
    attention_f2 = fbeta_score(all_labels, all_preds, beta=2, pos_label=1, zero_division=0)
    return report, cm, auc, attention_recall, attention_f2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits_dir", default="./splits")
    parser.add_argument("--output_dir", default="./runs/exp1")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=224)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")

    splits_dir = Path(args.splits_dir)
    train_ds = CarInteriorDataset(splits_dir / "train.csv", args.image_size, train=True)
    val_ds = CarInteriorDataset(splits_dir / "val.csv", args.image_size, train=False)
    test_ds = CarInteriorDataset(splits_dir / "test.csv", args.image_size, train=False)

    print("訓練集標籤分布:", train_ds.label_counts())
    print("驗證集標籤分布:", val_ds.label_counts())
    print("測試集標籤分布:", test_ds.label_counts())

    sampler = build_weighted_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = SeatAwareClassifier().to(device)

    # 注意：已改用 WeightedRandomSampler 處理類別不平衡，這裡不再對損失函數額外加權，
    # 避免兩種矯正手法疊加，把模型推向「全部都猜 attention」的退化解
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_f2 = -1.0
    patience, patience_limit = 0, 8

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for imgs, seats, labels in train_loader:
            imgs, seats, labels = imgs.to(device), seats.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs, seats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        train_loss = running_loss / len(train_ds)
        report, cm, auc, attention_recall, attention_f2 = evaluate(model, val_loader, device)
        scheduler.step(attention_f2)

        print(
            f"[Epoch {epoch}] train_loss={train_loss:.4f}  "
            f"val_attention_recall={attention_recall:.3f}  "
            f"val_attention_precision={report['attention']['precision']:.3f}  "
            f"val_attention_f2={attention_f2:.3f}  "
            f"val_auc={auc:.3f}"
        )

        if attention_f2 > best_f2:
            best_f2 = attention_f2
            patience = 0
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            print(f"  -> 新最佳模型已儲存 (val_attention_f2={best_f2:.3f})")
        else:
            patience += 1
            if patience >= patience_limit:
                print("  -> 驗證集 F2 已無改善，提前停止訓練")
                break

    print("\n=== 用最佳模型在測試集上評估 ===")
    model.load_state_dict(torch.load(out_dir / "best_model.pt"))
    report, cm, auc, attention_recall, attention_f2 = evaluate(model, test_loader, device)
    print("混淆矩陣 (列=實際, 欄=預測, 順序=[clean, attention]):")
    print(cm)
    print(f"\nAUC: {auc:.3f}")
    print(f"attention recall: {attention_recall:.3f}")
    print(f"attention precision: {report['attention']['precision']:.3f}")
    print(f"attention f1: {report['attention']['f1-score']:.3f}")
    print(f"attention f2: {attention_f2:.3f}")


if __name__ == "__main__":
    main()
