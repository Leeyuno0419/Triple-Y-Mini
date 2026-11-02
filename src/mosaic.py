"""얼굴 모자이크 (B 조영준). 입출력 규칙은 역할별_구현명세.md B장 ①을 따른다.

사용법 (프로젝트 루트가 sys.path에 있어야 함):
  from src.mosaic import apply_mosaic
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from src.config import BLOCKS, BLUR_EDGE_RATIO, BLUR_SIGMA_RATIO, BLUR_WORK_SIDE, PAD
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


def apply_blur(
    img: np.ndarray, boxes: list[Box], pad: float = PAD,
    sigma_ratio: float = BLUR_SIGMA_RATIO, edge_ratio: float = BLUR_EDGE_RATIO, work_side: int = BLUR_WORK_SIDE,
) -> np.ndarray:
    """BGR 이미지의 얼굴 박스에 부드러운 블러를 씌운 새 이미지를 돌려준다. 입력 이미지는 바꾸지 않는다.

    apply_mosaic 과 같은 방식으로 박스에 pad 비율만큼 여백을 더하고 이미지 범위로 자르며, 크기가 0이면 건너뛴다.
    영역은 타원이고 가장자리는 서서히 섞어 경계선이 티 나지 않게 한다. 세기는 가림 영역 긴 변 x sigma_ratio 다.
    큰 가우시안 블러는 느려서, 가림 영역을 긴 변 work_side 픽셀까지 줄인 해상도에서 흐리고(시그마도 같은 비율로 줄인다) 원래 크기로 키운다.
    """
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
        scale = min(1.0, work_side / max(w, h))
        sw, sh = max(1, round(w * scale)), max(1, round(h * scale))
        small = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_AREA) if scale < 1.0 else roi
        blurred = cv2.GaussianBlur(small, (0, 0), max(w, h) * sigma_ratio * scale)

        mask = np.zeros((sh, sw), np.float32)
        cv2.ellipse(mask, (sw // 2, sh // 2), (sw // 2, sh // 2), 0, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), max(1.0, min(w, h) * edge_ratio) * scale)
        if scale < 1.0:
            blurred = cv2.resize(blurred, (w, h), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = mask[..., None]
        roi[:] = (roi * (1.0 - mask) + blurred * mask).astype(np.uint8)
    return out
