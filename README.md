# 車內整潔度分類模型 — 訓練 pipeline

## 環境安裝

```bash
pip install torch torchvision opencv-python pandas scikit-learn pillow
```

若有 GPU，先依照 https://pytorch.org/get-started/locally/ 安裝對應 CUDA 版本的 PyTorch，訓練速度會快非常多。

## 資料夾結構假設

你的照片資料夾底下要能對應 CSV 裡的 `filename` 欄位，例如：

```
image_root/
├── 前車內/
│   ├── H47214759_10_20260422162613.png
│   └── ...
└── 後車內/
    ├── H47260296_11_20260422081836.png
    └── ...
```

## 執行順序

### 1. 切分資料集

```bash
python prepare_split.py \
    --front_csv 前車內標註.csv \
    --rear_csv 後車內標註.csv \
    --image_root /你的照片資料夾路徑 \
    --output_dir ./splits
```

會輸出 `splits/train.csv`、`splits/val.csv`、`splits/test.csv`，並印出各集合的車輛數與標籤分布，
確認一下三個集合裡都有 attention 樣本（目前正類極少，若某個集合分到 0 筆，代表資料量還不夠支撐這個切分比例，
可以調整 `--val_size` `--test_size`，或先擴充標註量）。

### 2. 訓練模型

```bash
python train.py --splits_dir ./splits --output_dir ./runs/exp1 --epochs 30
```

訓練過程每個 epoch 會印出驗證集的 attention recall/precision/AUC，模型選擇依據是
**attention recall 最高的那個 checkpoint**（而非 loss 最低或 accuracy 最高），
因為業務上「漏抓髒車」的代價比「誤判乾淨車」高。

結束後會自動載入最佳模型，在測試集上印出混淆矩陣與完整指標。

### 3. 單張照片預測（demo 用）

```bash
python predict.py --checkpoint ./runs/exp1/best_model.pt --image path/to/photo.png --seat rear
```

## 已知限制 / 後續建議

- 目前 attention 樣本極少（前座22筆、後座32筆），第一輪訓練結果**僅供驗證 pipeline 是否跑得通**，
  不代表最終可上線的準確度。務必持續擴充標註量，尤其是 attention 類別。
- 若持續訓練後 attention recall 仍不理想，可以考慮：
  - 改用 focal loss 取代 class-weighted CrossEntropy，進一步強化對難分類正類樣本的關注
  - 對 attention 樣本做更激進的資料增強（例如複製多份套用不同增強）
  - 改用異常偵測（只用 clean 樣本訓練 autoencoder，抓重建誤差高的當異常）作為輔助或備案模型
- 目前模型完全沒看過「拍錯/無效」照片，若要在正式流程串接，前面應該有一個獨立的「車內 vs 車外」
  前置過濾器（可以用你已標註的 188+21 筆 invalid 資料訓練），避免拍錯的照片流入這個整潔度分類器。
