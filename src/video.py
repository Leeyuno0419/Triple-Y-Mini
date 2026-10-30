"""영상 얼굴 추적·모자이크 처리 (B 조영준).

프로젝트 루트에서 ``from src.video import process_video``로 사용한다.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from src.config import HOLD_FRAMES, UNTRACKED_HOLD_FRAMES, UNTRACKED_MATCH_IOU
from src.types import Box


@dataclass
class _BoxState:
    """마지막 탐지 박스와 직전 위치를 저장해 짧은 이동을 예측한다."""

    box: Box
    frame_idx: int
    previous_box: Optional[Box] = None
    previous_frame_idx: Optional[int] = None


class BoxHolder:
    """탐지가 잠깐 끊긴 얼굴 박스를 추적 ID 또는 위치로 연결해 유지한다.

    track_id가 있는 박스는 max_age만큼, ID가 없는 박스는 IoU로 같은 위치에
    매칭된 경우에만 untracked_max_age만큼 유지한다.
    """

    def __init__(
        self,
        max_age: int = HOLD_FRAMES,
        untracked_max_age: int = UNTRACKED_HOLD_FRAMES,
        match_iou: float = UNTRACKED_MATCH_IOU,
    ) -> None:
        if max_age < 0 or untracked_max_age < 0:
            raise ValueError("유지 프레임 수는 0 이상이어야 합니다.")
        if not 0.0 <= match_iou <= 1.0:
            raise ValueError("match_iou는 0과 1 사이여야 합니다.")
        self.max_age = max_age
        self.untracked_max_age = untracked_max_age
        self.match_iou = match_iou
        self._last_seen: dict[int, _BoxState] = {}
        self._untracked: dict[int, _BoxState] = {}
        self._next_untracked_id = 0

    @staticmethod
    def _iou(box_a: Box, box_b: Box) -> float:
        """두 얼굴 박스의 교차 영역 비율(IoU)을 계산한다."""
        ax1, ay1, ax2, ay2 = box_a["xyxy"]
        bx1, by1, bx2, by2 = box_b["xyxy"]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - intersection
        return intersection / union if union else 0.0

    def _find_match(
        self,
        box: Box,
        cache: dict[int, _BoxState],
        used: set[int],
        frame_idx: int,
        max_age: int,
    ) -> Optional[int]:
        """최근 박스 중 IoU가 가장 높은 미사용 항목을 찾는다."""
        best_key: Optional[int] = None
        best_iou = self.match_iou
        for key, state in cache.items():
            age = frame_idx - state.frame_idx
            if key in used or age < 0 or age > max_age:
                continue
            overlap = self._iou(box, state.box)
            if overlap >= best_iou:
                best_key, best_iou = key, overlap
        return best_key

    @staticmethod
    def _predict_box(state: _BoxState, frame_idx: int) -> Box:
        """최근 두 위치의 이동량으로 누락 프레임의 박스 위치를 보정한다."""
        age = frame_idx - state.frame_idx
        previous = state.previous_box
        previous_frame = state.previous_frame_idx
        if age <= 0 or previous is None or previous_frame is None:
            return state.box

        elapsed = state.frame_idx - previous_frame
        if elapsed <= 0:
            return state.box

        x1, y1, x2, y2 = state.box["xyxy"]
        px1, py1, px2, py2 = previous["xyxy"]
        width, height = max(x2 - x1, 1), max(y2 - y1, 1)
        last_cx, last_cy = (x1 + x2) / 2, (y1 + y2) / 2
        previous_cx, previous_cy = (px1 + px2) / 2, (py1 + py2) / 2

        # Limit per-frame motion so a bad match cannot project a box far away.
        vx = max(-width * 0.5, min(width * 0.5, (last_cx - previous_cx) / elapsed))
        vy = max(-height * 0.5, min(height * 0.5, (last_cy - previous_cy) / elapsed))
        dx, dy = round(vx * age), round(vy * age)
        return {**state.box, "xyxy": (x1 + dx, y1 + dy, x2 + dx, y2 + dy)}

    @staticmethod
    def _store(
        cache: dict[int, _BoxState],
        key: int,
        box: Box,
        frame_idx: int,
        max_age: int,
        fallback: Optional[_BoxState] = None,
    ) -> None:
        """새 위치를 저장하고, 짧은 간격이면 직전 관측을 이동 예측용으로 남긴다."""
        previous = cache.get(key) or fallback
        if previous is not None and 0 < frame_idx - previous.frame_idx <= max_age:
            cache[key] = _BoxState(box, frame_idx, previous.box, previous.frame_idx)
        else:
            cache[key] = _BoxState(box, frame_idx)

    def update(self, boxes: list[Box], frame_idx: int) -> list[Box]:
        """현재 박스와 짧은 탐지 공백을 메우는 최근 박스들을 반환한다."""
        current = list(boxes)
        seen_track_ids: set[int] = set()
        seen_untracked_ids: set[int] = set()

        for box in boxes:
            track_id = box["track_id"]
            if track_id is not None:
                # 정식 ID가 붙으면 같은 위치의 임시 박스를 정식 트랙으로 승격한다.
                anonymous_id = self._find_match(
                    box, self._untracked, seen_untracked_ids, frame_idx, self.untracked_max_age
                )
                anonymous_state = None
                if anonymous_id is not None:
                    anonymous_state = self._untracked.pop(anonymous_id)
                seen_track_ids.add(track_id)
                self._store(self._last_seen, track_id, box, frame_idx, self.max_age, anonymous_state)
                continue

            # ID가 잠깐 사라졌더라도 기존 트랙과 위치가 이어지면 그 트랙을 갱신한다.
            tracked_id = self._find_match(box, self._last_seen, seen_track_ids, frame_idx, self.max_age)
            if tracked_id is not None:
                tracked_box: Box = {**box, "track_id": tracked_id}
                self._store(self._last_seen, tracked_id, tracked_box, frame_idx, self.max_age)
                seen_track_ids.add(tracked_id)
                continue

            # 아직 ID가 없는 새 얼굴은 별도의 짧은 위치 기반 임시 트랙에 보관한다.
            anonymous_id = self._find_match(
                box, self._untracked, seen_untracked_ids, frame_idx, self.untracked_max_age
            )
            if anonymous_id is None:
                anonymous_id = self._next_untracked_id
                self._next_untracked_id += 1
            self._store(self._untracked, anonymous_id, box, frame_idx, self.untracked_max_age)
            seen_untracked_ids.add(anonymous_id)

        held: list[Box] = []
        for track_id, state in list(self._last_seen.items()):
            age = frame_idx - state.frame_idx
            if age > self.max_age:
                del self._last_seen[track_id]
            elif track_id not in seen_track_ids and age > 0:
                held.append(self._predict_box(state, frame_idx))

        for anonymous_id, state in list(self._untracked.items()):
            age = frame_idx - state.frame_idx
            if age > self.untracked_max_age:
                del self._untracked[anonymous_id]
            elif anonymous_id not in seen_untracked_ids and age > 0:
                held.append(self._predict_box(state, frame_idx))

        return current + held


def _load_exclusion_filter():
    """제외 모듈이 있으면 필터를 반환하고, 모듈 부재는 None으로 알린다."""
    try:
        from src.exclude import filter_excluded
    except ModuleNotFoundError as err:
        if err.name != "src.exclude":
            raise
        return None
    return filter_excluded


def process_video(
    in_path: str,
    out_path: str,
    ref_embs: Optional[np.ndarray] = None,
    on_progress: Optional[Callable[[float], None]] = None,
) -> dict:
    """영상의 얼굴을 추적·모자이크하고 결과 정보 dict를 반환한다.

    반환 키는 ``faces``, ``frames``, ``fps``, ``seconds``이다. ``faces``는
    track ID가 있는 고유 인원 수이며, FPS와 시간은 출력 인코딩까지 포함한다.
    """
    from src.detector import detect_faces, reset_tracker
    from src.mosaic import apply_mosaic

    in_file = Path(in_path)
    out_file = Path(out_path)
    if not in_file.is_file():
        raise ValueError(f"입력 영상을 찾을 수 없습니다: {in_path}")
    if in_file.resolve() == out_file.resolve():
        raise ValueError("출력 경로는 입력 영상 경로와 달라야 합니다.")

    filter_excluded = _load_exclusion_filter() if ref_embs is not None else None
    if ref_embs is not None and filter_excluded is None:
        warnings.warn(
            "src.exclude를 찾을 수 없어 등록 인물도 포함해 모든 얼굴을 모자이크합니다.",
            RuntimeWarning,
            stacklevel=2,
        )

    out_file.parent.mkdir(parents=True, exist_ok=True)
    reset_tracker()
    capture = cv2.VideoCapture(str(in_file))
    if not capture.isOpened():
        capture.release()
        raise ValueError("영상을 열 수 없습니다. 파일이 깨졌는지 확인해 주세요.")

    fps_src = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps_src <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise ValueError("영상의 프레임 속도 또는 해상도를 읽을 수 없습니다.")

    holder = BoxHolder()
    exclude_cache: dict = {}
    track_ids: set[int] = set()
    processed_frames = 0
    start = time.perf_counter()

    try:
        if on_progress is not None:
            on_progress(0.0)
        with tempfile.TemporaryDirectory(prefix="triple_y_video_") as temp_dir:
            intermediate = Path(temp_dir) / "mosaiced.mp4"
            writer = cv2.VideoWriter(
                str(intermediate), cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (width, height)
            )
            if not writer.isOpened():
                writer.release()
                raise RuntimeError("처리 영상을 저장할 코덱을 열 수 없습니다.")

            try:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if frame.shape[1] != width or frame.shape[0] != height:
                        raise ValueError("영상 프레임의 해상도가 일정하지 않습니다.")

                    boxes = detect_faces(frame, track=True)
                    for box in boxes:
                        if box["track_id"] is not None:
                            track_ids.add(box["track_id"])
                    boxes = holder.update(boxes, processed_frames)

                    if filter_excluded is not None:
                        boxes = filter_excluded(frame, boxes, ref_embs, cache=exclude_cache)
                    writer.write(apply_mosaic(frame, boxes))
                    processed_frames += 1

                    if on_progress is not None and total_frames > 0:
                        on_progress(min(processed_frames / total_frames, 1.0))
            finally:
                writer.release()
                capture.release()

            if processed_frames == 0:
                raise ValueError("영상에서 처리할 프레임을 읽지 못했습니다.")

            merge_audio_h264(str(intermediate), str(in_file), str(out_file))
    except Exception:
        capture.release()
        raise

    if on_progress is not None:
        on_progress(1.0)

    seconds = time.perf_counter() - start
    return {
        "faces": len(track_ids),
        "frames": processed_frames,
        "fps": processed_frames / seconds if seconds > 0 else 0.0,
        "seconds": seconds,
    }
