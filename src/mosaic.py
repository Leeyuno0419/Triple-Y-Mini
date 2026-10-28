"""얼굴 모자이크 (B 조영준). 입출력 규칙은 역할별_구현명세.md B장 ①을 따른다.

사용법 (프로젝트 루트가 sys.path에 있어야 함):
  from src.mosaic import apply_mosaic
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from src.config import BLOCKS, PAD
from src.types import Box


def apply_mosaic(
    img: np.ndarray, boxes: list[Box], pad: float = PAD, blocks: int = BLOCKS, shape: str = "ellipse"
) -> np.ndarray:
    """BGR 이미지의 얼굴 박스에 모자이크를 씌운 새 이미지를 돌려준다. 입력 이미지는 바꾸지 않는다.

    박스 폭·높이에 pad 비율만큼 여백을 더해 이미지 범위로 자르고, 크기가 0이면 건너뛴다.
    긴 변을 blocks칸으로 나눠 블록 크기가 얼굴 크기에 비례한다. shape은 "ellipse" 또는 "rect".
    """
    if shape not in ("ellipse", "rect"):
        raise ValueError(f"shape은 'ellipse' 또는 'rect': {shape}")

    out = img.copy()
    img_h, img_w = img.shape[:2]
    for box in boxes:
        x1, y1, x2, y2 = box["xyxy"]
        px, py = (x2 - x1) * pad, (y2 - y1) * pad
        x1, y1 = max(math.floor(x1 - px), 0), max(math.floor(y1 - py), 0)
        x2, y2 = min(math.ceil(x2 + px), img_w), min(math.ceil(y2 + py), img_h)
        if x2 <= x1 or y2 <= y1:
            continue

        roi = out[y1:y2, x1:x2]  # out의 뷰라 쓰면 out에 반영된다
        h, w = roi.shape[:2]
        scale = blocks / max(w, h)  # 긴 변을 blocks칸으로 → 블록 크기가 얼굴 크기에 비례
        small = cv2.resize(roi, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
        pixel = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

        if shape == "ellipse":
            mask = np.zeros((h, w), np.uint8)
            cv2.ellipse(mask, (w // 2, h // 2), (w // 2, h // 2), 0, 0, 360, 255, -1)
            roi[mask > 0] = pixel[mask > 0]
        else:
            roi[:] = pixel
    return out
