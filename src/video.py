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

from src.config import (
    DETECT_PROGRESS,
    DETECT_PROGRESS_EXCLUDE,
    HOLD_FRAMES,
    MAX_FILL_GAP,
    SCENE_CUT_CORR,
    SCENE_CUT_MIN_GAP,
    SCENE_FLAT_STD,
    SCENE_THUMB_SIZE,
    UNTRACKED_HOLD_FRAMES,
    UNTRACKED_MATCH_IOU,
)
from src.track_fill import interpolate_gaps, stitch_tracks
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


@dataclass
class VideoAnalysis:
    """영상을 한 번 처리한 뒤 남기는, 결과를 다시 만들 때 쓰는 값. 프레임 이미지는 담지 않아 가볍다.

    boxes[f]            f번째 프레임의 박스(보간·박스 유지까지 끝났고 인물 제외 판정 전). track_id 는 탐지기가 준 원래 값이다.
    auto_excluded[f]    f번째 프레임에서 등록 인물(ref_embs)이라 모자이크에서 빠진 박스의 위치(boxes[f] 안의 번호).
    alias               끊겨서 새 ID 가 된 track -> 같은 사람으로 이어 붙인 대표 track. 사람 단위 편집(unmask·mask)에 쓴다.
    """

    boxes: list[list[Box]]
    auto_excluded: list[frozenset]
    alias: dict[int, int]
    fps: float
    width: int
    height: int
    cuts: frozenset = frozenset()    # 장면 전환(새 장면이 시작하는 프레임 번호). 참고용이다


def person_of(analysis: VideoAnalysis, track_id: Optional[int]) -> Optional[int]:
    """track_id 가 속한 사람 번호(끊긴 track 을 이어 붙인 대표 track). track_id 가 없으면 None."""
    return None if track_id is None else analysis.alias.get(track_id, track_id)


def edits_at(edits: tuple, frame_idx: int) -> tuple[dict, list]:
    """frame_idx 프레임에 걸리는 편집을 정리한다. (사람 번호 -> "unmask" | "mask", 이 프레임에서 가릴 직접 영역 목록)

    edits 는 (start, end, kind, value) 의 순서 있는 목록이고 start~end 프레임(양끝 포함)에만 적용된다. 같은 대상은 뒤의 편집이 앞을 덮는다.
      kind "unmask" / "mask"     : value 는 사람 번호. 그 사람을 가리지 않음 / 등록 인물이어도 가림.
      kind "region" / "unregion" : value 는 (x1, y1, x2, y2) 원본 픽셀. 그 자리를 가림(직접 영역) / 그 직접 영역을 뺌.
    """
    decision: dict[int, str] = {}
    regions: dict[tuple, bool] = {}    # 순서를 지키려고 dict 를 쓴다
    for start, end, kind, value in edits:
        if start <= frame_idx <= end:
            if kind in ("unmask", "mask"):
                decision[value] = kind
            else:
                regions[tuple(value)] = kind == "region"
    return decision, [region for region, on in regions.items() if on]


def visible_boxes(analysis: VideoAnalysis, frame_idx: int, edits: tuple = ()) -> list[Box]:
    """frame_idx 프레임에서 실제로 가릴 박스 목록. 사진의 resolve_boxes 와 같은 규칙이다.

    기본은 인물 제외가 빼지 않은 모든 박스다. 이 프레임에 걸린 편집(edits_at)에서 사람이 "unmask" 면 그 사람의 박스를 가리지 않고(해제),
    "mask" 면 등록 인물이어도 가린다(해제했던 것을 다시 가림). 직접 영역 (x1, y1, x2, y2) 은 그 자리를 가린다.
    ID 가 없는 박스는 사람 단위로 고를 수 없으므로 기본 규칙(등록 인물 판정)을 따른다.
    """
    auto = analysis.auto_excluded[frame_idx]
    decision, regions = edits_at(edits, frame_idx)
    out: list[Box] = []
    for i, box in enumerate(analysis.boxes[frame_idx]):
        person = person_of(analysis, box["track_id"])
        if person is not None and decision.get(person) == "unmask":
            continue
        if (person is not None and decision.get(person) == "mask") or i not in auto:
            out.append(box)
    for x1, y1, x2, y2 in regions:
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(analysis.width, int(x2)), min(analysis.height, int(y2))
        if x2 - x1 >= 2 and y2 - y1 >= 2:
            out.append({"xyxy": (x1, y1, x2, y2), "conf": 1.0, "track_id": None})
    return out


def _find_ffmpeg() -> Optional[str]:
    """ffmpeg 경로를 반환한다. PATH에 없으면 imageio-ffmpeg에 들어 있는 실행 파일을 쓰고, 둘 다 없으면 None."""
    found = shutil.which("ffmpeg")
    if found is not None:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # 패키지가 없거나 실행 파일을 못 찾으면 ffmpeg 없음으로 본다
        return None


def merge_audio_h264(video_path: str, audio_src: str, out_path: str) -> None:
    """처리 영상에 원본의 오디오를 선택적으로 합치고 H.264 MP4로 저장한다.

    입력에 오디오 트랙이 없어도 영상만 출력한다. ffmpeg가 없거나 인코딩에
    실패하면 RuntimeError를 발생시킨다.
    """
    ffmpeg = _find_ffmpeg()
    if ffmpeg is None:
        raise RuntimeError("ffmpeg를 찾을 수 없습니다. imageio-ffmpeg를 설치하거나 ffmpeg를 PATH에 등록해 주세요.")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.resolve() in (Path(video_path).resolve(), Path(audio_src).resolve()):
        raise ValueError("출력 경로는 입력 영상 경로와 달라야 합니다.")

    fd, temp_name = tempfile.mkstemp(prefix=f"{out.stem}_", suffix=out.suffix or ".mp4", dir=out.parent)
    os.close(fd)
    encoded_temp = Path(temp_name)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        video_path,
        "-i",
        audio_src,
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-shortest",
        "-movflags",
        "+faststart",
        str(encoded_temp),
    ]
    try:
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=False)
        except OSError as err:
            raise RuntimeError(f"ffmpeg를 실행할 수 없습니다: {err}") from err
        if result.returncode != 0 or not encoded_temp.is_file() or encoded_temp.stat().st_size == 0:
            detail = result.stderr.strip()[-1200:] or "알 수 없는 ffmpeg 오류"
            raise RuntimeError(f"ffmpeg 영상 인코딩에 실패했습니다: {detail}")
        encoded_temp.replace(out)
    finally:
        encoded_temp.unlink(missing_ok=True)


def _frame_thumb(frame: np.ndarray) -> np.ndarray:
    """장면 전환 감지용: 프레임을 32x18 흑백으로 줄여 1차원 벡터로 반환한다."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, SCENE_THUMB_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32).ravel()


def _is_scene_cut(prev_thumb: Optional[np.ndarray], thumb: np.ndarray) -> bool:
    """앞 프레임과 밝기 구조의 상관계수가 SCENE_CUT_CORR 보다 낮으면 장면 전환으로 본다.

    상관계수는 전체 밝기·대비 변화(무대 조명)에는 거의 변하지 않고, 화면 구성이 바뀔 때만 크게 떨어진다.
    한쪽이 단색 화면이면 상관을 잴 수 없어 장면 전환으로 보지 않는다.
    """
    if prev_thumb is None or prev_thumb.std() < SCENE_FLAT_STD or thumb.std() < SCENE_FLAT_STD:
        return False
    return float(np.corrcoef(prev_thumb, thumb)[0, 1]) < SCENE_CUT_CORR


class _SceneCutDetector:
    """프레임을 차례로 받아 장면 전환인지 판정한다 (process_video·process_stream 공용).

    단색 화면(암전 등)은 비교 기준으로 삼지 않아, 암전 앞 화면과 뒤 화면을 비교한다.
    전환 직후 min_gap 프레임 안에서 다시 잡힌 전환은 흔들림·잔상 효과로 보고 무시한다.
    """

    def __init__(self, min_gap: int = SCENE_CUT_MIN_GAP) -> None:
        self.min_gap = min_gap
        self._prev_thumb: Optional[np.ndarray] = None
        self._frames_since_cut = min_gap  # 첫 전환은 바로 인정한다

    def update(self, frame: np.ndarray) -> bool:
        """이번 프레임이 장면 전환이면 True를 반환한다."""
        self._frames_since_cut += 1
        thumb = _frame_thumb(frame)
        if thumb.std() < SCENE_FLAT_STD:
            return False
        cut = _is_scene_cut(self._prev_thumb, thumb) and self._frames_since_cut > self.min_gap
        self._prev_thumb = thumb
        if cut:
            self._frames_since_cut = 0
        return cut


def _load_exclusion_filter():
    """제외 모듈이 있으면 필터를 반환하고, 모듈 부재는 None으로 알린다."""
    try:
        from src.exclude import filter_excluded
    except ModuleNotFoundError as err:
        if err.name != "src.exclude":
            raise
        return None
    return filter_excluded


def process_video_edit(
    in_path: str,
    out_path: str,
    ref_embs: Optional[np.ndarray] = None,
    on_progress: Optional[Callable[[float], None]] = None,
    effect: str = "mosaic",
) -> tuple[dict, VideoAnalysis]:
    """영상의 얼굴을 추적·가리고(모자이크 또는 블러) (결과 정보 dict, VideoAnalysis)를 반환한다.

    VideoAnalysis 로 탐지 없이 결과를 다시 만들 수 있다(render_video). 정보 dict 만 필요하면 process_video 를 쓴다.

    effect 는 ``"mosaic"``(기본) 또는 ``"blur"``이다. 그 밖의 값이면 ValueError.

    영상을 두 번 읽는다. 1패스에서 탐지·추적만 해 박스를 모으고, 끊긴 공백을 앞뒤 박스로 보간한 뒤(src.track_fill),
    2패스에서 박스 유지·인물 제외·모자이크·저장을 한다. 진행률은 1패스가 앞쪽 DETECT_PROGRESS(인물 제외를 쓰면 DETECT_PROGRESS_EXCLUDE)를 차지한다.
    반환 키는 ``faces``, ``frames``, ``fps``, ``seconds``이다. ``faces``는
    끊겨서 ID가 바뀐 track을 이어 붙인 뒤의 고유 인원 수이며, FPS와 시간은 출력 인코딩까지 포함한다.
    """
    from src.detector import detect_faces, reset_tracker
    from src.mosaic import apply_blur, apply_mosaic

    if effect not in ("mosaic", "blur"):
        raise ValueError(f"effect는 'mosaic' 또는 'blur'여야 합니다: {effect}")
    apply_effect = apply_blur if effect == "blur" else apply_mosaic

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

    # 인물 제외를 쓰면 2패스에서 얼굴 인식이 돌아 오래 걸리므로, 진행률에서 1패스의 몫을 줄여 막대가 멈춘 것처럼 보이지 않게 한다.
    detect_share = DETECT_PROGRESS_EXCLUDE if filter_excluded is not None and len(ref_embs) > 0 else DETECT_PROGRESS
    holder = BoxHolder()
    exclude_cache: dict = {}
    track_ids: set[int] = set()
    base_boxes: list[list[Box]] = []
    auto_excluded: list[frozenset] = []
    per_frame: list[list[Box]] = []
    scene_cut = _SceneCutDetector()
    cut_frames: set[int] = set()    # 1패스에서 잡은 장면 전환. 보간과 2패스 초기화가 같은 컷을 쓰도록 여기서 한 번만 판정한다
    id_offset = 0    # 장면이 바뀌어 추적기를 비우면 track_id 가 1 부터 다시 매겨지므로, 영상 전체에서 겹치지 않게 이어 붙이는 값
    max_track_id = 0
    processed_frames = 0
    start = time.perf_counter()

    try:
        if on_progress is not None:
            on_progress(0.0)

        # 1패스: 탐지·추적만 한다. 프레임 이미지는 쌓지 않고 박스 목록만 모은다.
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape[1] != width or frame.shape[0] != height:
                raise ValueError("영상 프레임의 해상도가 일정하지 않습니다.")

            if scene_cut.update(frame):
                # 장면이 바뀌면 추적기를 비운다. (남겨 두면 이전 장면의 번호가 같은 자리에 나온 다른 사람에게 붙는다)
                reset_tracker()
                id_offset = max_track_id
                cut_frames.add(len(per_frame))

            boxes = detect_faces(frame, track=True)
            for box in boxes:
                if box["track_id"] is not None:
                    box["track_id"] += id_offset    # 추적기를 비우면 번호가 1 부터 다시 매겨지니 영상 전체에서 겹치지 않게 이어 붙인다
                    max_track_id = max(max_track_id, box["track_id"])
                    track_ids.add(box["track_id"])
            per_frame.append(boxes)

            if on_progress is not None and total_frames > 0:
                on_progress(min(len(per_frame) / total_frames, 1.0) * detect_share)
        capture.release()

        if not per_frame:
            raise ValueError("영상에서 처리할 프레임을 읽지 못했습니다.")

        # 뒤쪽 박스까지 알고 나서, 끊긴 공백을 앞뒤 박스로 보간한다. track_id 는 바꾸지 않고 대응표로만 인원 수를 센다.
        # 장면 전환을 가로질러 잇거나 채우면 다른 장면의 허공·몸통에 박스가 생기므로, 컷에서는 끊는다.
        cuts = frozenset(cut_frames)
        alias = stitch_tracks(per_frame, MAX_FILL_GAP, cuts=cuts)
        per_frame, _ = interpolate_gaps(per_frame, alias, MAX_FILL_GAP, cuts=cuts)
        faces = len({alias.get(track_id, track_id) for track_id in track_ids})

        # 2패스: 영상을 다시 열어 박스 유지 -> 인물 제외 -> 모자이크 -> 저장
        capture = cv2.VideoCapture(str(in_file))
        if not capture.isOpened():
            raise ValueError("영상을 다시 열 수 없습니다. 처리 중 파일이 바뀌었는지 확인해 주세요.")
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
                    if processed_frames >= len(per_frame):
                        raise ValueError("영상을 다시 읽는 중 프레임 수가 달라졌습니다.")  # 박스 없이 내보내면 얼굴이 노출된다

                    if processed_frames in cuts:
                        holder = BoxHolder()    # 새 장면이니 이전 장면에서 유지하던 박스를 버린다
                        exclude_cache.clear()    # 이전 장면의 제외 판정도 버린다 (같은 자리에 나온 다른 사람이 '제외'를 이어받지 않게)
                    held_boxes = holder.update(per_frame[processed_frames], processed_frames)
                    boxes = held_boxes
                    if filter_excluded is not None:
                        boxes = filter_excluded(frame, held_boxes, ref_embs, cache=exclude_cache)
                    kept_ids = {id(box) for box in boxes}    # filter_excluded 는 남긴 박스를 같은 객체로 돌려준다
                    base_boxes.append(held_boxes)
                    auto_excluded.append(frozenset(i for i, box in enumerate(held_boxes) if id(box) not in kept_ids))
                    writer.write(apply_effect(frame, boxes))
                    processed_frames += 1

                    if on_progress is not None:
                        on_progress(detect_share + (1.0 - detect_share) * processed_frames / len(per_frame))
            finally:
                writer.release()
                capture.release()

            if processed_frames != len(per_frame):
                raise ValueError("영상을 다시 읽는 중 프레임 수가 달라졌습니다.")

            merge_audio_h264(str(intermediate), str(in_file), str(out_file))
    except Exception:
        capture.release()
        raise

    if on_progress is not None:
        on_progress(1.0)

    seconds = time.perf_counter() - start
    info = {
        "faces": faces,
        "frames": processed_frames,
        "fps": processed_frames / seconds if seconds > 0 else 0.0,
        "seconds": seconds,
    }
    return info, VideoAnalysis(base_boxes, auto_excluded, alias, fps_src, width, height, cuts)


def process_video(
    in_path: str,
    out_path: str,
    ref_embs: Optional[np.ndarray] = None,
    on_progress: Optional[Callable[[float], None]] = None,
    effect: str = "mosaic",
) -> dict:
    """영상의 얼굴을 추적·가리고(모자이크 또는 블러) 결과 정보 dict를 반환한다. 자세한 동작은 process_video_edit 와 같다."""
    info, _ = process_video_edit(in_path, out_path, ref_embs, on_progress, effect)
    return info


def render_video(
    in_path: str,
    out_path: str,
    analysis: VideoAnalysis,
    effect: str = "mosaic",
    edits: tuple = (),
    on_progress: Optional[Callable[[float], None]] = None,
) -> dict:
    """process_video_edit 가 남긴 analysis 로 얼굴을 다시 찾지 않고 결과 영상을 다시 만든다. 반환 키는 process_video 와 같다.

    edits 는 구간별 편집 (start, end, kind, value) 목록이다(사람 해제·다시 가림, 직접 영역 추가·빼기). 형식과 규칙은 edits_at·visible_boxes 를 보라.
    영상을 읽어 가리고 저장하는 일뿐이라 처리보다 훨씬 빠르다.
    읽은 프레임 수가 analysis 와 다르면 ValueError (박스 없이 내보내면 얼굴이 노출된다).
    """
    from src.mosaic import apply_blur, apply_mosaic

    if effect not in ("mosaic", "blur"):
        raise ValueError(f"effect는 'mosaic' 또는 'blur'여야 합니다: {effect}")
    apply_effect = apply_blur if effect == "blur" else apply_mosaic
    in_file, out_file = Path(in_path), Path(out_path)
    if not in_file.is_file():
        raise ValueError(f"입력 영상을 찾을 수 없습니다: {in_path}")
    if in_file.resolve() == out_file.resolve():
        raise ValueError("출력 경로는 입력 영상 경로와 달라야 합니다.")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(in_file))
    if not capture.isOpened():
        capture.release()
        raise ValueError("영상을 열 수 없습니다. 파일이 깨졌는지 확인해 주세요.")

    total = len(analysis.boxes)
    persons: set[int] = set()
    processed_frames = 0
    start = time.perf_counter()
    try:
        if on_progress is not None:
            on_progress(0.0)
        with tempfile.TemporaryDirectory(prefix="triple_y_video_") as temp_dir:
            intermediate = Path(temp_dir) / "mosaiced.mp4"
            writer = cv2.VideoWriter(
                str(intermediate), cv2.VideoWriter_fourcc(*"mp4v"), analysis.fps, (analysis.width, analysis.height)
            )
            if not writer.isOpened():
                writer.release()
                raise RuntimeError("처리 영상을 저장할 코덱을 열 수 없습니다.")
            try:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if processed_frames >= total or frame.shape[1] != analysis.width or frame.shape[0] != analysis.height:
                        raise ValueError("영상을 다시 읽는 중 프레임 수나 크기가 달라졌습니다.")
                    boxes = visible_boxes(analysis, processed_frames, edits)
                    persons.update(p for p in (person_of(analysis, box["track_id"]) for box in boxes) if p is not None)
                    writer.write(apply_effect(frame, boxes))
                    processed_frames += 1
                    if on_progress is not None:
                        on_progress(min(processed_frames / total, 1.0))
            finally:
                writer.release()
                capture.release()
            if processed_frames != total:
                raise ValueError("영상을 다시 읽는 중 프레임 수가 달라졌습니다.")
            merge_audio_h264(str(intermediate), str(in_file), str(out_file))
    except Exception:
        capture.release()
        raise

    seconds = time.perf_counter() - start
    return {
        "faces": len(persons),
        "frames": processed_frames,
        "fps": processed_frames / seconds if seconds > 0 else 0.0,
        "seconds": seconds,
    }
