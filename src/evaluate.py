"""평가 스크립트 (A 이유노). 입출력 규칙은 역할별_구현명세.md A장 ⑤⑥을 따른다.

사용법 (프로젝트 루트가 sys.path에 있어야 함):
  from src.evaluate import evaluate_miss_rate, benchmark_fps
"""
from __future__ import annotations

import math
import platform
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.config import CONF, MIN_EVAL_SIZE, PAD
from src.types import Box

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
_WARMUP_FRAMES = 10  # 첫 추론은 모델 초기화로 느려서 FPS 측정에서 뺀다


def _read_yolo_labels(path: Path, img_w: int, img_h: int) -> list[tuple[int, int, int, int]]:
    """YOLO 라벨(클래스 중심x 중심y 너비 높이, 0~1 비율)을 픽셀 xyxy 목록으로 바꾼다. 파일이 없으면 []."""
    if not path.exists():
        return []
    boxes = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cx, cy, bw, bh = (float(v) for v in parts[1:5])
        x1, y1 = max(round((cx - bw / 2) * img_w), 0), max(round((cy - bh / 2) * img_h), 0)
        x2, y2 = min(round((cx + bw / 2) * img_w), img_w), min(round((cy + bh / 2) * img_h), img_h)
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return boxes


def _mosaic_mask(boxes: list[Box], img_w: int, img_h: int) -> np.ndarray:
    """apply_mosaic(shape="ellipse")이 실제로 덮는 영역을 bool 마스크로 돌려준다.

    여백·자르기·타원 계산은 src/mosaic.py와 같다. mosaic.py를 바꾸면 여기도 맞춰야 한다.
    """
    mask = np.zeros((img_h, img_w), np.uint8)
    for box in boxes:
        x1, y1, x2, y2 = box["xyxy"]
        px, py = (x2 - x1) * PAD, (y2 - y1) * PAD
        x1, y1 = max(math.floor(x1 - px), 0), max(math.floor(y1 - py), 0)
        x2, y2 = min(math.ceil(x2 + px), img_w), min(math.ceil(y2 + py), img_h)
        if x2 <= x1 or y2 <= y1:
            continue
        w, h = x2 - x1, y2 - y1
        ellipse = np.zeros((h, w), np.uint8)
        cv2.ellipse(ellipse, (w // 2, h // 2), (w // 2, h // 2), 0, 0, 360, 255, -1)
        mask[y1:y2, x1:x2] |= ellipse  # 박스끼리 겹쳐도 합집합
    return mask > 0


def evaluate_miss_rate(
    detect_fn: Callable[[np.ndarray], list[Box]],
    test_dir: str,
    cover_thr: float = 0.9,
    min_size: int = MIN_EVAL_SIZE,
) -> dict:
    """테스트셋에서 모자이크가 덮지 못한 정답 얼굴의 비율(누락률)을 잰다.

    test_dir 아래 images/와 labels/(YOLO 형식, Roboflow 내보내기 구조)를 읽는다.
    정답 박스 넓이의 cover_thr 이상이 실제 모자이크 영역(여백 + 타원)에 덮이면 "가림"이다.
    짧은 변이 min_size 미만인 정답 얼굴은 small_faces·small_missed로 따로 센다.
    missed_files에는 min_size 이상 얼굴을 놓친 이미지 파일 이름만 담는다.
    """
    root = Path(test_dir)
    total = missed = small = small_missed = 0
    missed_files: list[str] = []
    for img_path in sorted((root / "images").iterdir()):
        if img_path.suffix.lower() not in _IMG_EXTS:
            continue
        img = cv2.imread(str(img_path))
        img_h, img_w = img.shape[:2]
        gts = _read_yolo_labels(root / "labels" / f"{img_path.stem}.txt", img_w, img_h)
        if not gts:
            continue

        covered = _mosaic_mask(detect_fn(img), img_w, img_h)
        file_missed = False
        for x1, y1, x2, y2 in gts:
            hit = covered[y1:y2, x1:x2].mean() >= cover_thr
            if min(x2 - x1, y2 - y1) < min_size:
                small += 1
                small_missed += not hit
            else:
                total += 1
                if not hit:
                    missed += 1
                    file_missed = True
        if file_missed:
            missed_files.append(img_path.name)

    return {
        "total_faces": total,
        "missed": missed,
        "miss_rate": missed / total if total else 0.0,
        "missed_files": missed_files,
        "small_faces": small,
        "small_missed": small_missed,
    }


def benchmark_fps(
    weights: str,
    video_path: str,
    imgsz: int = 640,
    max_frames: int = 300,
    ref_embs: Optional[np.ndarray] = None,
) -> dict:
    """영상 앞 max_frames 프레임으로 탐지만 / 전체 / 전체 + 인물 제외 세 단계 FPS를 잰다.

    det_fps는 weights 모델을 직접 불러 imgsz로 탐지만 잰다 (처음 몇 프레임은 워밍업으로 뺌).
    pipeline_fps·exclude_fps는 B의 process_video가 돌려주는 fps이며, process_video는
    config.WEIGHTS 모델을 쓰므로 weights·imgsz 인자가 반영되지 않는다.
    B의 video.py가 없으면 pipeline_fps, C의 exclude.py가 없거나 ref_embs가 None이면 exclude_fps가 None이다.
    """
    cap = cv2.VideoCapture(video_path)
    fps_src = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while len(frames) < max_frames:
        ok, img = cap.read()
        if not ok:
            break
        frames.append(img)
    cap.release()

    model = YOLO(weights)
    for img in frames[:_WARMUP_FRAMES]:
        model(img, imgsz=imgsz, conf=CONF, verbose=False)
    start = time.perf_counter()
    for img in frames[_WARMUP_FRAMES:]:
        model(img, imgsz=imgsz, conf=CONF, verbose=False)
    det_fps = (len(frames) - _WARMUP_FRAMES) / (time.perf_counter() - start)

    pipeline_fps = exclude_fps = None
    try:
        from src.video import process_video  # B 모듈이 생기기 전에도 det_fps는 잴 수 있게 지연 import
    except ModuleNotFoundError:
        process_video = None
    if process_video is not None:
        with tempfile.TemporaryDirectory() as tmp:
            clip = str(Path(tmp) / "clip.mp4")  # 같은 프레임 수로 맞추려고 잘라 저장
            h, w = frames[0].shape[:2]
            writer = cv2.VideoWriter(clip, cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (w, h))
            for img in frames:
                writer.write(img)
            writer.release()
            pipeline_fps = process_video(clip, str(Path(tmp) / "out.mp4"))["fps"]
            if ref_embs is not None:
                try:
                    exclude_fps = process_video(clip, str(Path(tmp) / "out_ex.mp4"), ref_embs=ref_embs)["fps"]
                except ModuleNotFoundError:  # C의 exclude.py가 아직 없음
                    exclude_fps = None

    return {
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor() or "CPU",
        "model": Path(weights).stem,
        "imgsz": imgsz,
        "det_fps": det_fps,
        "pipeline_fps": pipeline_fps,
        "exclude_fps": exclude_fps,
    }
