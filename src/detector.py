"""얼굴 탐지·추적 (A 이유노). 입출력 규칙은 역할별_구현명세.md A장 ①②를 따른다.

사용법 (프로젝트 루트가 sys.path에 있어야 함):
  from src.detector import detect_faces, reset_tracker
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import IterableSimpleNamespace

from src.config import CONF, TRACKER, WEIGHTS
from src.types import Box

_WEIGHTS_PATH = str(Path(__file__).resolve().parent.parent / WEIGHTS)  # 실행 폴더와 무관하게 루트 기준

_model = YOLO(_WEIGHTS_PATH)  # import 시 한 번만 로드
# model.track()은 track을 못 받은 탐지(새로 나타난 얼굴 등)를 결과에서 빼서 그 얼굴이 노출된다.
# 추적기를 직접 돌려 모든 탐지를 돌려주고, track이 붙은 것에만 track_id를 단다.
_tracker = BYTETracker(IterableSimpleNamespace(**TRACKER))


def detect_faces(img: np.ndarray, track: bool = False) -> list[Box]:
    """BGR 이미지에서 얼굴을 찾아 박스 목록을 돌려준다. 얼굴이 없으면 [].

    track=True면 ByteTrack으로 프레임 간 같은 사람에 track_id를 붙인다 (영상·실시간용).
    점수가 CONF 이상인 탐지는 track을 못 받아도 track_id=None으로 돌려준다 (모자이크 누락 방지).
    CONF 미만 탐지는 기존 track에 이어 붙었을 때만 돌려준다.
    좌표는 바깥쪽으로 반올림(x1·y1 내림, x2·y2 올림)해 이미지 범위로 자르고, 크기가 0인 박스는 버린다.
    """
    if track:
        result = _model(img, conf=TRACKER["track_low_thresh"], verbose=False)[0]  # 2단계 매칭용 약한 탐지까지
        tracks = _tracker.update(result.boxes.cpu().numpy(), img)
        id_by_idx = {int(t[7]): int(t[4]) for t in tracks}  # t = [x1, y1, x2, y2, track_id, score, cls, idx]
        ids = [id_by_idx.get(i) for i in range(len(result.boxes))]
    else:
        result = _model(img, conf=CONF, verbose=False)[0]
        ids = [None] * len(result.boxes)

    img_h, img_w = img.shape[:2]
    boxes: list[Box] = []
    for (x1, y1, x2, y2), conf, track_id in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), ids):
        if track_id is None and conf < CONF:
            continue
        x1, y1 = max(math.floor(x1), 0), max(math.floor(y1), 0)
        x2, y2 = min(math.ceil(x2), img_w), min(math.ceil(y2), img_h)
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append({"xyxy": (x1, y1, x2, y2), "conf": float(conf), "track_id": track_id})
    return boxes


def reset_tracker() -> None:
    """추적 상태를 비우고 track_id를 1부터 다시 매긴다. 새 영상을 시작할 때 호출한다."""
    _tracker.reset()
