"""
predict.py

給單張照片路徑 + 座位別，輸出「乾淨/需留意」預測結果與信心分數。
適合在 demo 時串接展示，或之後包成 API 端點。

使用方式：
    python predict.py --checkpoint ./runs/exp1/best_model.pt --image path/to/photo.png --seat rear
"""

import argparse

import cv2
import torch
from PIL import Image
from torchvision import transforms

from dataset import apply_clahe, imread_unicode, IMAGENET_MEAN, IMAGENET_STD, SEAT_TO_IDX
from train import SeatAwareClassifier

LABEL_NAMES = {0: "乾淨 (clean)", 1: "需留意 (attention)"}


def load_model(checkpoint_path: str, device: torch.device) -> SeatAwareClassifier:
    model = SeatAwareClassifier().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def preprocess(image_path: str, image_size: int = 224) -> torch.Tensor:
    img_bgr = imread_unicode(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"讀取失敗: {image_path}")
    img_bgr = apply_clahe(img_bgr)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb).resize((image_size, image_size))

    to_tensor_norm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    )
    return to_tensor_norm(pil_img).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--seat", choices=["front", "rear"], required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    img_tensor = preprocess(args.image).to(device)
    seat_tensor = torch.tensor([SEAT_TO_IDX[args.seat]], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(img_tensor, seat_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred = probs.argmax().item()

    print(f"預測結果: {LABEL_NAMES[pred]}")
    print(f"信心分數: clean={probs[0]:.3f}  attention={probs[1]:.3f}")


if __name__ == "__main__":
    main()
