"""얼굴 탐지·추적 (A 이유노). 입출력 규칙은 역할별_구현명세.md A장 ①②를 따른다.

사용법 (프로젝트 루트가 sys.path에 있어야 함):
  from src.detector import detect_faces, reset_tracker
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from src.config import CONF, WEIGHTS
from src.types import Box

_WEIGHTS_PATH = str(Path(__file__).resolve().parent.parent / WEIGHTS)  # 실행 폴더와 무관하게 루트 기준

_model = YOLO(_WEIGHTS_PATH)  # import 시 한 번만 로드


def detect_faces(img: np.ndarray, track: bool = False) -> list[Box]:
    """BGR 이미지에서 얼굴을 찾아 박스 목록을 돌려준다. 얼굴이 없으면 [].

    track=True면 ByteTrack으로 프레임 간 같은 사람에 track_id를 붙인다 (영상·실시간용).
    좌표는 바깥쪽으로 반올림(x1·y1 내림, x2·y2 올림)해 이미지 범위로 자르고, 크기가 0인 박스는 버린다.
    """
    if track:
        result = _model.track(img, persist=True, tracker="bytetrack.yaml", conf=CONF, verbose=False)[0]
    else:
        result = _model(img, conf=CONF, verbose=False)[0]

    img_h, img_w = img.shape[:2]
    n = len(result.boxes)
    ids = result.boxes.id.int().tolist() if track and result.boxes.id is not None else [None] * n

    boxes: list[Box] = []
    for (x1, y1, x2, y2), conf, track_id in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), ids):
        x1, y1 = max(math.floor(x1), 0), max(math.floor(y1), 0)
        x2, y2 = min(math.ceil(x2), img_w), min(math.ceil(y2), img_h)
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append({"xyxy": (x1, y1, x2, y2), "conf": float(conf), "track_id": track_id})
    return boxes


def reset_tracker() -> None:
    """추적 상태를 비우고 track_id를 1부터 다시 매긴다. 새 영상을 시작할 때 호출한다."""
    for tracker in getattr(_model.predictor, "trackers", []):  # 아직 track을 안 불렀으면 빈 목록
        tracker.reset()
