"""
dataset.py

自訂 PyTorch Dataset：
- 讀取照片並做 CLAHE 曝光/對比度正規化（緩解你資料中「室內昏暗 vs 戶外過曝」的光線落差）
- 訓練集套用資料增強（因為正類樣本極少，需要靠增強放大有效樣本數）
- seat（前/後座）轉成整數索引，供模型當作輔助特徵
"""

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

SEAT_TO_IDX = {"front": 0, "rear": 1}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def imread_unicode(path: str) -> np.ndarray:
    """
    Windows 上 cv2.imread() 對中文路徑支援不完整，容易讀取失敗或無聲跳過。
    改用 np.fromfile + cv2.imdecode 繞過這個限制。
    """
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def apply_clahe(img_bgr: np.ndarray) -> np.ndarray:
    """在 LAB 色彩空間的 L 通道做 CLAHE，避免破壞色彩資訊（髒污顏色本身也是判斷線索）。"""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


class CarInteriorDataset(Dataset):
    def __init__(self, csv_path: str, image_size: int = 224, train: bool = False):
        self.df = pd.read_csv(csv_path)
        self.image_size = image_size
        self.train = train

        # 一般增強：套用在多數類別（clean），強度溫和，避免把本來就夠用的資料過度扭曲
        self.aug = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15),
                transforms.RandomRotation(degrees=8),
                transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
            ]
        )

        # 強化增強：只套用在稀少的 attention 類別，用更大幅度的變換
        # 從 39 張照片盡量榨出更多「看起來不同」的訓練樣本，降低過擬合速度
        self.aug_strong = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05),
                transforms.RandomRotation(degrees=15),
                transforms.RandomResizedCrop(image_size, scale=(0.65, 1.0)),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            ]
        )
        # RandomErasing 模擬局部遮擋/雜訊，僅套用在強化版；要在 ToTensor 之後才能用，
        # 所以標記一個 flag，在 __getitem__ 裡對 tensor 做後處理
        self.erasing = transforms.RandomErasing(p=0.4, scale=(0.02, 0.12), value="random")

        self.eval_resize = transforms.Compose([transforms.Resize((image_size, image_size))])

        self.to_tensor_norm = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = int(row["label_id"])

        img_bgr = imread_unicode(row["filepath"])
        if img_bgr is None:
            raise FileNotFoundError(f"讀取失敗，請確認路徑存在: {row['filepath']}")

        img_bgr = apply_clahe(img_bgr)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        from PIL import Image

        pil_img = Image.fromarray(img_rgb)

        if self.train and label == 1:
            pil_img = self.aug_strong(pil_img)
        elif self.train:
            pil_img = self.aug(pil_img)
        else:
            pil_img = self.eval_resize(pil_img)

        if pil_img.size != (self.image_size, self.image_size):
            pil_img = pil_img.resize((self.image_size, self.image_size))

        tensor_img = self.to_tensor_norm(pil_img)

        if self.train and label == 1:
            tensor_img = self.erasing(tensor_img)

        seat_idx = SEAT_TO_IDX[row["seat"]]

        return tensor_img, torch.tensor(seat_idx, dtype=torch.long), torch.tensor(label, dtype=torch.long)

    def label_counts(self):
        return self.df["label_id"].value_counts().to_dict()
