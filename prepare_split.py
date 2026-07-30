"""
prepare_split.py

讀取前座/後座標註 CSV，合併成單一資料表，過濾掉「拍錯/無效」照片，
並「依車輛 ID」切分 train/val/test，避免同一台車的前後座照片
分散到不同集合造成 data leakage。

同時針對「是否含有 attention 標籤」做車輛層級的分層抽樣，
確保三個集合裡都有一定比例的正類樣本可供訓練/驗證/測試。

使用方式：
    python prepare_split.py \
        --front_csv 前車內標註.csv \
        --rear_csv 後車內標註.csv \
        --image_root /path/to/your/image/folder \
        --output_dir ./splits
"""

import argparse
import re
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

CAR_ID_PATTERN = re.compile(r"(H\d+)_")

LABEL_MAP = {"clean": 0, "attention": 1}


def extract_car_id(filename: str) -> str:
    m = CAR_ID_PATTERN.search(filename)
    if not m:
        raise ValueError(f"無法從檔名解析車輛 ID: {filename}")
    return m.group(1)


def load_and_merge(front_csv: str, rear_csv: str) -> pd.DataFrame:
    front = pd.read_csv(front_csv)
    front["seat"] = "front"
    rear = pd.read_csv(rear_csv)
    rear["seat"] = "rear"

    df = pd.concat([front, rear], ignore_index=True)

    # 過濾拍錯/無效照片，不參與訓練
    before = len(df)
    df = df[df["label"] != "invalid"].copy()
    print(f"過濾無效照片：{before} -> {len(df)} 筆")

    df["car_id"] = df["filename"].apply(extract_car_id)
    df["label_id"] = df["label"].map(LABEL_MAP)

    if df["label_id"].isnull().any():
        bad = df[df["label_id"].isnull()]["label"].unique()
        raise ValueError(f"發現未知標籤值: {bad}，請確認 CSV 內容")

    return df


def split_by_car(df: pd.DataFrame, val_size: float, test_size: float, seed: int) -> dict:
    """
    以車輛 ID 為單位切分，並依「該車是否曾出現 attention 標籤」做分層，
    確保各集合都有正類樣本。
    """
    car_has_attention = (
        df.groupby("car_id")["label_id"].max().rename("has_attention").reset_index()
    )

    train_ids, temp_ids = train_test_split(
        car_has_attention["car_id"],
        test_size=(val_size + test_size),
        stratify=car_has_attention["has_attention"],
        random_state=seed,
    )

    temp_df = car_has_attention[car_has_attention["car_id"].isin(temp_ids)]
    relative_test_size = test_size / (val_size + test_size)
    val_ids, test_ids = train_test_split(
        temp_df["car_id"],
        test_size=relative_test_size,
        stratify=temp_df["has_attention"],
        random_state=seed,
    )

    return {
        "train": df[df["car_id"].isin(train_ids)].copy(),
        "val": df[df["car_id"].isin(val_ids)].copy(),
        "test": df[df["car_id"].isin(test_ids)].copy(),
    }


def report(name: str, split_df: pd.DataFrame):
    n_cars = split_df["car_id"].nunique()
    n_total = len(split_df)
    dist = split_df["label"].value_counts().to_dict()
    print(f"[{name}] 車輛數={n_cars}  照片數={n_total}  標籤分布={dist}")


def filter_missing_files(df: pd.DataFrame) -> pd.DataFrame:
    """檢查每張照片的檔案是否真的存在，剔除缺檔的列並回報數量。"""
    exists_mask = df["filepath"].apply(lambda p: Path(p).exists())
    n_missing = (~exists_mask).sum()
    if n_missing > 0:
        print(f"\n⚠️  發現 {n_missing} 筆照片在磁碟上找不到，將從資料集中剔除。")
        missing_sample = df.loc[~exists_mask, "filepath"].head(10).tolist()
        print("缺檔範例（最多列出10筆）：")
        for p in missing_sample:
            print(f"  - {p}")
        # 把完整缺檔清單存成 CSV，方便之後回頭核對
        missing_out = Path("missing_files.csv")
        df.loc[~exists_mask, ["filename", "filepath"]].to_csv(missing_out, index=False)
        print(f"完整缺檔清單已存到 {missing_out.resolve()}\n")
    return df[exists_mask].copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--front_csv", required=True)
    parser.add_argument("--rear_csv", required=True)
    parser.add_argument("--image_root", required=True, help="照片資料夾根目錄（CSV 內 filename 是相對此路徑）")
    parser.add_argument("--output_dir", default="./splits")
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_and_merge(args.front_csv, args.rear_csv)
    df["filepath"] = df["filename"].apply(lambda f: str(Path(args.image_root) / f))
    df = filter_missing_files(df)

    splits = split_by_car(df, args.val_size, args.test_size, args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, split_df in splits.items():
        report(name, split_df)
        cols = ["filepath", "seat", "label", "label_id", "car_id"]
        split_df[cols].to_csv(out_dir / f"{name}.csv", index=False)

    print(f"\n切分完成，已輸出到 {out_dir}/train.csv, val.csv, test.csv")


if __name__ == "__main__":
    main()
