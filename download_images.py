import os
import pandas as pd
import requests
from tqdm import tqdm

# ======================================
# 專案路徑設定
# ======================================

# 取得目前程式所在資料夾
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Excel 路徑
EXCEL_PATH = os.path.join(BASE_DIR, "data", "車外車內照片上傳清單.xlsx")

# 圖片儲存位置
SAVE_FOLDER = os.path.join(BASE_DIR, "images")

# 建立 images 資料夾
os.makedirs(SAVE_FOLDER, exist_ok=True)

print("目前工作目錄：", os.getcwd())
print("程式位置：", BASE_DIR)
print()

# ======================================
# 讀取 Excel
# ======================================

print("正在讀取 Excel...")

df = pd.read_excel(EXCEL_PATH)

# 去除欄位前後空白
df.columns = df.columns.str.strip()

print(f"Excel 共 {len(df)} 筆資料")

# ======================================
# 只保留 前車內、後車內
# ======================================

df = df[df["上傳分類(常有誤)"].isin(["前車內", "後車內"])]

print(f"需要下載 {len(df)} 張圖片")
print()

# ======================================
# 開始下載
# ======================================

success = 0
failed = 0

for _, row in tqdm(df.iterrows(), total=len(df)):

    url = str(row["下載網址"]).strip()

    # 直接從網址取得檔名，比讀 Excel 更安全
    filename = os.path.basename(url)

    category = row["上傳分類(常有誤)"]

    # 建立分類資料夾
    folder = os.path.join(SAVE_FOLDER, category)
    os.makedirs(folder, exist_ok=True)

    save_path = os.path.join(folder, filename)

    # 已存在則跳過
    if os.path.exists(save_path):
        success += 1
        continue

    try:

        response = requests.get(url, timeout=20)

        if response.status_code == 200:

            with open(save_path, "wb") as f:
                f.write(response.content)

            success += 1

        else:

            print(f"HTTP {response.status_code}：{filename}")
            failed += 1

    except Exception as e:

        print(f"下載失敗：{filename}")
        print(e)
        failed += 1

# ======================================
# 完成
# ======================================

print()
print("=" * 40)
print("下載完成")
print("=" * 40)

print(f"成功：{success}")
print(f"失敗：{failed}")

print()
print("圖片儲存位置：")
print(SAVE_FOLDER)