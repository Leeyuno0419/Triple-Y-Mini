"""옆얼굴 미검출·ID 튐 진단 도구: 프레임 로그, 파라미터 비교, 끊긴 track 이어 붙이기·선형 보간.

영상에서 인물이 고개를 돌려 옆얼굴이 될 때 생기는 세 가지 문제를 한 파일에서 다룬다.
  1) 탐지 누락    : YOLO(conf·imgsz)와 InsightFace SCRFD(det_thresh·det_size)를 나란히 돌려 어느 쪽이 놓치는지 본다.
  2) ID 튐        : ByteTrack 결과에서 track 이 끊기거나 같은 자리에 새 ID 가 생긴 프레임을 찾아 낸다.
  3) 박스 끊김     : src/track_fill.py 의 stitch_tracks·interpolate_gaps 로 끊긴 공백을 앞뒤 박스로 선형 보간한다 (process_video 와 같은 코드).

사용법 (프로젝트 루트에서):
  # 1) 문제 프레임만 터미널에 출력 + 요약. --all 이면 모든 프레임, --json 이면 표준 JSON 도 저장
  python tools\\debug_side_face.py log data\\raw\\test\\video.mp4 --json data\\processed\\debug_log.json

  # 2) 파라미터 비교: 구간(프레임 범위)을 정해 conf·imgsz·det_thresh·det_size 조합별 검출 프레임 비율을 비교
  python tools\\debug_side_face.py sweep data\\raw\\test\\video.mp4 --start 300 --end 420

  # 3) 이어 붙이기 + 보간을 적용한 결과 영상. --draw 면 모자이크 대신 박스(초록=검출, 주황=보간, 파랑=ID없음)를 그린다
  python tools\\debug_side_face.py render data\\raw\\test\\video.mp4 data\\processed\\debug_out.mp4 --draw

원칙 (CLAUDE.md)
  - 이 도구의 숫자 옵션은 '실험용'이다. 좋은 값을 찾으면 src/config.py 에 옮기고 팀에 알린다 (공통 파일).
  - 기본값은 src/config.py 의 CONF · TRACKER · DET_THRESH · DET_SIZE 를 그대로 쓴다. 파이프라인과 같은 조건에서 재현된다.
  - 입력 영상(data/raw)은 읽기만 한다. 출력은 data/processed 처럼 git 제외 폴더에 둔다.
  - 한계: render 는 인물 제외(src.exclude) 를 거치지 않는다. 등록 인물도 모자이크되니 '박스가 이어지는지' 확인용으로만 쓴다.
"""
from __future__ import annotations

try:    # Windows 에서는 torch 보다 onnx 를 먼저 불러야 한다 (InsightFace 가 쓰는 onnx 의 DLL 오류 방지)
    import onnx  # noqa: F401
except ImportError:
    pass

import argparse
import json
import math
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator, Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CONF, DET_SIZE, DET_THRESH, JUDGE_MAX_TURN, TRACKER, WEIGHTS  # noqa: E402

XYXY = Tuple[float, float, float, float]

if hasattr(sys.stdout, "reconfigure"):    # 콘솔 인코딩이 한글을 못 쓰는 환경에서도 멈추지 않게 한다
    sys.stdout.reconfigure(errors="replace")


# ============================================================
# ① 데이터와 기하 도우미
# ============================================================

@dataclass(frozen=True)
class Det:
    """한 프레임의 얼굴 하나. src: Y=YOLO만, I=InsightFace만, YI=둘 다, interp=보간으로 채운 박스."""

    xyxy: XYXY
    conf: float
    src: str = "Y"
    track_id: Optional[int] = None
    yolo_conf: float = 0.0
    insight_conf: float = 0.0
    turn: Optional[float] = None    # 얼굴 돌림 지표(정면 0, 클수록 옆얼굴). InsightFace 가 찾은 얼굴만 값이 있다


def iou(a: XYXY, b: XYXY) -> float:
    """두 박스의 IoU."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def shift_ratio(a: XYXY, b: XYXY) -> float:
    """두 박스 중심 사이 거리 ÷ 앞 박스의 긴 변. 얼굴이 박스 크기의 몇 배만큼 움직였는지 본다."""
    dx = (a[0] + a[2]) / 2 - (b[0] + b[2]) / 2
    dy = (a[1] + a[3]) / 2 - (b[1] + b[3]) / 2
    return math.hypot(dx, dy) / max(a[2] - a[0], a[3] - a[1], 1.0)


def size_ratio(a: XYXY, b: XYXY) -> float:
    """두 박스 넓이의 비(작은 쪽 ÷ 큰 쪽)."""
    area_a = max((a[2] - a[0]) * (a[3] - a[1]), 1.0)
    area_b = max((b[2] - b[0]) * (b[3] - b[1]), 1.0)
    return min(area_a, area_b) / max(area_a, area_b)


def iter_frames(path: Path, start: int = 0, end: Optional[int] = None, step: int = 1) -> Iterator[tuple[int, np.ndarray]]:
    """영상 프레임을 (프레임 번호, BGR 이미지) 로 차례로 돌려준다. [start, end) 구간만 읽는다."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"영상을 열 수 없습니다: {path}")
    try:
        if start > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        idx = start
        while end is None or idx < end:
            ok, frame = cap.read()
            if not ok:
                break
            if (idx - start) % step == 0:
                yield idx, frame
            idx += 1
    finally:
        cap.release()


def video_info(path: Path) -> tuple[float, int, int, int]:
    """(fps, 전체 프레임 수, 폭, 높이)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"영상을 열 수 없습니다: {path}")
    info = (cap.get(cv2.CAP_PROP_FPS) or 0.0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return info


# ============================================================
# ② 탐지기: YOLO + InsightFace(SCRFD), 파라미터를 호출마다 바꿀 수 있게 감싼다
# ============================================================

def face_turn(kps: Optional[np.ndarray]) -> Optional[float]:
    """코가 두 눈 가운데에서 벗어난 정도 ÷ 눈 사이 거리 (src/exclude.py 의 _face_turn 과 같은 식). 정면은 0 근처."""
    if kps is None or len(kps) < 3:
        return None
    gap = abs(float(kps[1][0]) - float(kps[0][0]))
    if gap < 1.0:
        return float("inf")
    return abs(float(kps[2][0]) - (float(kps[0][0]) + float(kps[1][0])) / 2) / gap


class Detectors:
    """YOLO 와 InsightFace 탐지기를 한 번씩만 만들고, 호출마다 conf·imgsz·det_thresh·det_size 를 바꿔 쓴다."""

    def __init__(self, use_insight: bool = True, tta: bool = False) -> None:
        from ultralytics import YOLO

        self.yolo_model = YOLO(str(ROOT / WEIGHTS))
        self.tta = tta
        self.det_model: Any = None
        if use_insight:
            import onnxruntime
            from insightface.app import FaceAnalysis

            gpu = "CUDAExecutionProvider" in onnxruntime.get_available_providers()
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if gpu else ["CPUExecutionProvider"]
            app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"], providers=providers)
            app.prepare(ctx_id=0 if gpu else -1, det_thresh=DET_THRESH, det_size=(DET_SIZE, DET_SIZE))
            self.det_model = app.det_model

    def yolo(self, frame: np.ndarray, conf: float, imgsz: int) -> list[tuple[XYXY, float]]:
        """YOLO 얼굴 탐지. conf 이상인 (박스, 점수) 목록. imgsz 는 32 의 배수가 좋다. tta 면 좌우 반전·다중 크기를 섞는다."""
        res = self.yolo_model(frame, conf=conf, imgsz=imgsz, augment=self.tta, verbose=False)[0]
        return [(tuple(b), c) for b, c in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist())]

    def insight(self, frame: np.ndarray, det_thresh: float, det_size: int) -> list[tuple[XYXY, float, Optional[float]]]:
        """SCRFD 얼굴 탐지. det_thresh 이상인 (박스, 점수, 돌림 지표) 목록. det_size 가 클수록 작은 얼굴에 강하지만 느리다."""
        if self.det_model is None:
            return []
        self.det_model.det_thresh = det_thresh
        bboxes, kpss = self.det_model.detect(frame, input_size=(det_size, det_size), max_num=0, metric="default")
        out = []
        for i, b in enumerate(bboxes):
            out.append(((float(b[0]), float(b[1]), float(b[2]), float(b[3])), float(b[4]),
                        face_turn(kpss[i] if kpss is not None else None)))
        return out


def fuse(yolo: list[tuple[XYXY, float]], insight: list[tuple[XYXY, float, Optional[float]]], fuse_iou: float) -> list[Det]:
    """두 탐지 결과를 합친다. 같은 얼굴(IoU>=fuse_iou)이면 YOLO 박스를 쓰되 점수는 둘 중 큰 값으로 올린다.

    옆얼굴에서 YOLO 점수만 낮아지는 경우, SCRFD 점수가 높으면 추적기의 1단계 매칭(높은 점수)을 계속 통과한다.
    YOLO 가 못 찾고 SCRFD 만 찾은 얼굴은 SCRFD 박스를 그대로 추가한다.
    """
    dets = [Det(box, conf, "Y", yolo_conf=conf) for box, conf in yolo]
    for box, conf, turn in insight:
        best, best_iou = -1, fuse_iou
        for i, d in enumerate(dets):
            overlap = iou(box, d.xyxy)
            if overlap >= best_iou:
                best, best_iou = i, overlap
        if best >= 0:
            d = dets[best]
            dets[best] = replace(d, src="YI", conf=max(d.conf, conf), insight_conf=conf, turn=turn)
        else:
            dets.append(Det(box, conf, "I", insight_conf=conf, turn=turn))
    return dets


# ============================================================
# ③ 추적: 파이프라인(src/detector.py)과 같은 ByteTrack 설정
# ============================================================

class _TrackerInput:
    """BYTETracker.update 가 요구하는 모양(conf·cls·xywh, 불리언 인덱싱)만 갖춘 입력."""

    def __init__(self, xywh: np.ndarray, conf: np.ndarray) -> None:
        self.xywh, self.conf, self.cls = xywh, conf, np.zeros(len(conf), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, mask: np.ndarray) -> "_TrackerInput":
        return _TrackerInput(self.xywh[mask], self.conf[mask])


def track_frame(tracker: Any, dets: list[Det], frame: np.ndarray, conf_min: float) -> list[Det]:
    """합친 탐지를 ByteTrack 에 넣어 track_id 를 붙인다. 좌표는 detector.py 처럼 바깥쪽 반올림해 이미지 범위로 자른다.

    track 을 못 받은 탐지는 conf_min 이상일 때만 track_id=None 으로 남긴다 (모자이크 누락 방지, detector.py 와 같은 규칙).
    """
    h, w = frame.shape[:2]
    if dets:
        xyxy = np.array([d.xyxy for d in dets], dtype=np.float32)
        xywh = np.stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2,
                         xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]], axis=1)
        inp = _TrackerInput(xywh, np.array([d.conf for d in dets], dtype=np.float32))
    else:
        inp = _TrackerInput(np.zeros((0, 4), np.float32), np.zeros(0, np.float32))
    tracks = tracker.update(inp, frame)
    id_by_idx = {int(t[7]): int(t[4]) for t in tracks}    # t = [x1, y1, x2, y2, track_id, score, cls, idx]

    out: list[Det] = []
    for i, d in enumerate(dets):
        tid = id_by_idx.get(i)
        if tid is None and d.conf < conf_min:
            continue
        x1, y1 = max(math.floor(d.xyxy[0]), 0), max(math.floor(d.xyxy[1]), 0)
        x2, y2 = min(math.ceil(d.xyxy[2]), w), min(math.ceil(d.xyxy[3]), h)
        if x2 > x1 and y2 > y1:
            out.append(replace(d, xyxy=(x1, y1, x2, y2), track_id=tid))
    return out


def make_tracker(conf: float, low_conf: float, track_buffer: int, match_thresh: float) -> Any:
    """config 의 TRACKER 설정에 명령줄 값을 덮어쓴 BYTETracker 를 만든다."""
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.utils import IterableSimpleNamespace

    cfg = dict(TRACKER, track_high_thresh=conf, new_track_thresh=conf, track_low_thresh=low_conf,
               track_buffer=track_buffer, match_thresh=match_thresh)
    return BYTETracker(IterableSimpleNamespace(**cfg))


# ============================================================
# ④ 프레임 로그: 미검출·낮은 신뢰도·ID 튐을 터미널에 자세히 출력
# ============================================================

class FrameLogger:
    """프레임마다 이상 징후(flag)를 찾아 출력하고, 끝에서 요약한다.

    flag 의미
      EMPTY          얼굴이 0개 (직전 프레임에는 있었을 때만 표시. --all 이면 항상)
      DROP           직전 프레임보다 얼굴 수가 줄었다
      LOW            신뢰도가 warn_conf 미만인 얼굴이 있다
      SIDE           돌림 지표가 JUDGE_MAX_TURN 이상인 얼굴(옆얼굴)이 있다
      YOLO_MISS      SCRFD 만 찾고 YOLO 는 놓친 얼굴이 있다  -> YOLO conf 를 낮추거나 imgsz 를 키울 후보
      INSIGHT_MISS   YOLO 만 찾고 SCRFD 는 놓친 얼굴이 있다  -> det_thresh 를 낮추거나 det_size 를 키울 후보
      BOX_JUMP       같은 track_id 의 박스가 직전 위치와 IoU 가 jump_iou 미만으로 튀었다
      ID_SWITCH?     방금 사라진 track 의 자리에 새 track_id 가 생겼다 (같은 사람이 다른 ID 로 바뀐 의심)
    """

    def __init__(self, fps: float, warn_conf: float, jump_iou: float, switch_iou: float, memory: int,
                 show_all: bool, use_insight: bool) -> None:
        self.fps, self.warn_conf, self.jump_iou, self.switch_iou = fps or 1.0, warn_conf, jump_iou, switch_iou
        self.memory, self.show_all, self.use_insight = memory, show_all, use_insight
        self.prev_count = 0
        self.last: dict[int, tuple[int, XYXY]] = {}    # track_id -> (마지막으로 본 프레임, 박스)
        self.seen_ids: set[int] = set()
        self.flag_counts: dict[str, int] = {}
        self.obs: dict[int, list[int]] = {}            # track_id -> 관측 프레임 번호
        self.empty_runs: list[list[int]] = []
        self.switches: list[tuple[int, int, int]] = []    # (프레임, 옛 ID, 새 ID)

    def feed(self, idx: int, dets: list[Det]) -> None:
        """한 프레임을 검사하고, 이상이 있으면(또는 --all 이면) 자세히 출력한다."""
        flags: list[str] = []
        n = len(dets)
        if n == 0 and (self.prev_count > 0 or self.show_all):
            flags.append("EMPTY")
        if n < self.prev_count:
            flags.append("DROP")
        if any(d.conf < self.warn_conf for d in dets):
            flags.append("LOW")
        if any(d.turn is not None and d.turn >= JUDGE_MAX_TURN for d in dets):
            flags.append("SIDE")
        if self.use_insight and any(d.src == "I" for d in dets):
            flags.append("YOLO_MISS")
        if self.use_insight and any(d.src == "Y" for d in dets):
            flags.append("INSIGHT_MISS")

        notes: dict[int, str] = {}
        for i, d in enumerate(dets):
            tid = d.track_id
            if tid is None:
                continue
            prev = self.last.get(tid)
            if prev is not None and idx - prev[0] <= self.memory and iou(prev[1], d.xyxy) < self.jump_iou:
                flags.append("BOX_JUMP")
                notes[i] = f"BOX_JUMP iou={iou(prev[1], d.xyxy):.2f} (f{prev[0]})"
            if tid not in self.seen_ids:
                for old, (seen_f, old_box) in self.last.items():
                    here = any(x.track_id == old for x in dets)
                    if not here and 0 < idx - seen_f <= self.memory and iou(old_box, d.xyxy) >= self.switch_iou:
                        flags.append("ID_SWITCH?")
                        notes[i] = f"ID_SWITCH? #{old}->#{tid} (gap={idx - seen_f - 1}f)"
                        self.switches.append((idx, old, tid))
                        break
        for d in dets:
            if d.track_id is not None:
                self.seen_ids.add(d.track_id)
                self.last[d.track_id] = (idx, d.xyxy)
                self.obs.setdefault(d.track_id, []).append(idx)

        if n == 0:
            if self.empty_runs and self.empty_runs[-1][1] == idx - 1:
                self.empty_runs[-1][1] = idx
            else:
                self.empty_runs.append([idx, idx])
        self.prev_count = n
        for f in set(flags):
            self.flag_counts[f] = self.flag_counts.get(f, 0) + 1

        if flags or self.show_all:
            n_y = sum(1 for d in dets if "Y" in d.src)
            n_i = sum(1 for d in dets if "I" in d.src)
            uniq = ",".join(sorted(set(flags))) or "-"
            print(f"[F{idx:06d} t={idx / self.fps:7.2f}s] faces={n} yolo={n_y} insight={n_i} flags={uniq}")
            for i, d in enumerate(dets):
                x1, y1, x2, y2 = (int(v) for v in d.xyxy)
                turn = "  n/a" if d.turn is None else f"{d.turn:5.2f}"
                tid = "-" if d.track_id is None else str(d.track_id)
                print(f"    #{i} id={tid:>3} src={d.src:<2} conf={d.conf:.3f} (yolo={d.yolo_conf:.3f} insight={d.insight_conf:.3f}) "
                      f"turn={turn} xyxy=({x1},{y1},{x2},{y2}) size={x2 - x1}x{y2 - y1}"
                      + (f"  <-- {notes[i]}" if i in notes else ""))

    def summary(self, total: int, max_gap: int) -> None:
        """전체 요약: flag 횟수, track 수, 끊김 구간, 얼굴 0개 구간, 다음에 해볼 조치."""
        print("\n" + "=" * 70)
        print(f"요약: {total} 프레임, track {len(self.seen_ids)}개, ID 교체 의심 {len(self.switches)}건")
        for name in sorted(self.flag_counts):
            print(f"  {name:<13} {self.flag_counts[name]:>6} 프레임")
        gaps = []
        for tid, frames in self.obs.items():
            for a, b in zip(frames, frames[1:]):
                if b - a > 1:
                    gaps.append((b - a - 1, tid, a + 1, b - 1))
        gaps.sort(reverse=True)
        fillable = sum(1 for g in gaps if g[0] <= max_gap)
        print(f"  track 내부 끊김 {len(gaps)}곳 (보간 가능 {max_gap}프레임 이하: {fillable}곳). 긴 순서 상위 10:")
        for length, tid, s, e in gaps[:10]:
            print(f"    id={tid} F{s}-F{e} ({length}프레임)")
        long_empty = [r for r in self.empty_runs if r[1] - r[0] + 1 >= 1][:10]
        if long_empty:
            print("  얼굴 0개 구간 (처음 10개): " + ", ".join(f"F{a}-F{b}" for a, b in long_empty))
        print("  조치 가이드:")
        if self.flag_counts.get("YOLO_MISS"):
            print("    - YOLO_MISS 있음: --conf/--low-conf 낮추기, --imgsz 키우기(960·1280), --tta 시험. sweep 으로 비교")
        if self.flag_counts.get("INSIGHT_MISS"):
            print("    - INSIGHT_MISS 있음: --det-thresh 낮추기, --det-size 키우기. sweep 으로 비교")
        if self.switches or self.flag_counts.get("BOX_JUMP"):
            print("    - ID 교체/박스 튐 있음: --track-buffer 늘리기, render 의 stitch_tracks 로 이어 붙이기")
        if gaps:
            print("    - 끊김 있음: render 의 interpolate_gaps 로 앞뒤 박스를 선형 보간")


# ============================================================
# ⑤ 검출 → 추적 한 번 돌리기 (log·render 공용)
# ============================================================

def collect(path: Path, args: argparse.Namespace, logger: Optional[FrameLogger] = None) -> list[list[Det]]:
    """영상 전체를 검출·합치기·추적해 프레임별 Det 목록을 만든다. 프레임 이미지는 저장하지 않는다(메모리 절약)."""
    detectors = Detectors(use_insight=not args.no_insight, tta=args.tta)
    tracker = make_tracker(args.conf, args.low_conf, args.track_buffer, args.match_thresh)
    fps, total, _, _ = video_info(path)
    stop = args.max_frames if args.max_frames else None
    frames: list[list[Det]] = []
    began = time.perf_counter()
    for idx, frame in iter_frames(path, end=stop):
        yolo = detectors.yolo(frame, args.low_conf, args.imgsz)
        insight = detectors.insight(frame, args.det_thresh, args.det_size)
        dets = track_frame(tracker, fuse(yolo, insight, args.fuse_iou), frame, args.conf)
        frames.append(dets)
        if logger is not None:
            logger.feed(idx, dets)
        elif idx % 100 == 0:
            print(f"  검출 {idx}/{total} ({(idx + 1) / (time.perf_counter() - began):.1f} fps)", flush=True)
    return frames


# ============================================================
# ⑥ ID 이어 붙이기와 선형 보간 (본체는 src/track_fill.py, process_video 가 쓰는 것과 같다)
# ============================================================

def fill_gaps(frames: list[list[Det]], max_gap: int) -> tuple[list[list[Det]], int, int]:
    """Det 목록에 src.track_fill 의 이어 붙이기·보간을 적용한다. 반환: (보간 박스가 src="interp" 로 추가된 목록, 이어 붙인 수, 채운 수)."""
    from src.track_fill import interpolate_gaps, stitch_tracks

    boxes = [[{"xyxy": tuple(int(v) for v in d.xyxy), "conf": d.conf, "track_id": d.track_id} for d in dets] for dets in frames]
    alias = stitch_tracks(boxes, max_gap)
    filled_boxes, filled = interpolate_gaps(boxes, alias, max_gap)
    out = []
    for dets, original, grown in zip(frames, boxes, filled_boxes):
        extra = [Det(b["xyxy"], b["conf"], "interp", b["track_id"]) for b in grown[len(original):]]    # 보간 박스는 목록 맨 뒤에 붙는다
        out.append(list(dets) + extra)
    return out, len(alias), filled


# ============================================================
# ⑦ 하위 명령: log / sweep / render
# ============================================================

def cmd_log(args: argparse.Namespace) -> None:
    """프레임별 진단 로그를 출력하고, 요약과 (선택) 표준 JSON 을 남긴다."""
    path = Path(args.video)
    fps, _, _, _ = video_info(path)
    logger = FrameLogger(fps, args.warn_conf, args.jump_iou, args.switch_iou, args.max_gap + 1, args.all, not args.no_insight)
    frames = collect(path, args, logger)
    logger.summary(len(frames), args.max_gap)
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        doc = {
            "schema_version": 1, "video": path.name, "model": WEIGHTS, "fps": fps,
            "params": {"conf": args.conf, "low_conf": args.low_conf, "imgsz": args.imgsz,
                       "det_thresh": args.det_thresh, "det_size": args.det_size, "insight": not args.no_insight},
            "frames": [{"frame_idx": i, "detections": [
                {"class": "face", "conf": round(d.conf, 4), "bbox": [int(v) for v in d.xyxy],    # 원본 픽셀 절대좌표 xyxy
                 "track_id": d.track_id, "src": d.src, "turn": None if d.turn is None or math.isinf(d.turn) else round(d.turn, 3)}
                for d in dets]} for i, dets in enumerate(frames)],
        }
        out.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        print(f"JSON 저장: {out}")


def _run_grid(detectors: Detectors, frames: list[tuple[int, np.ndarray]], args: argparse.Namespace) -> None:
    """sweep 본체: 설정 조합별로 얼굴이 1개 이상 검출된 프레임 비율과 속도를 표로 출력한다."""
    n = len(frames)
    rows: list[tuple[str, set[int], int, float]] = []    # (이름, 검출된 프레임 인덱스, 얼굴 수, ms/프레임)
    detectors.yolo(frames[0][1], min(args.sweep_conf), args.sweep_imgsz[0])    # 첫 호출은 모델 준비 시간이 섞여 속도 비교가 틀어진다
    detectors.insight(frames[0][1], min(args.sweep_det_thresh), args.sweep_det_size[0])
    for imgsz in args.sweep_imgsz:
        began = time.perf_counter()
        raw = [detectors.yolo(f, min(args.sweep_conf), imgsz) for _, f in frames]
        ms = (time.perf_counter() - began) * 1000 / n
        for conf in args.sweep_conf:
            kept = [[c for c in r if c[1] >= conf] for r in raw]
            rows.append((f"YOLO imgsz={imgsz} conf={conf}", {i for i, k in enumerate(kept) if k}, sum(map(len, kept)), ms))
    if detectors.det_model is not None:
        for size in args.sweep_det_size:
            began = time.perf_counter()
            raw_i = [detectors.insight(f, min(args.sweep_det_thresh), size) for _, f in frames]
            ms = (time.perf_counter() - began) * 1000 / n
            for th in args.sweep_det_thresh:
                kept_i = [[c for c in r if c[1] >= th] for r in raw_i]
                rows.append((f"SCRFD det_size={size} det_thresh={th}", {i for i, k in enumerate(kept_i) if k}, sum(map(len, kept_i)), ms))

    union = set().union(*(r[1] for r in rows)) if rows else set()
    print(f"\n구간 {frames[0][0]}~{frames[-1][0]} ({n} 프레임, step={args.step}), 어느 설정이든 얼굴을 찾은 프레임: {len(union)}")
    print(f"{'설정':<40}{'검출 프레임':>10}{'커버율':>9}{'얼굴 수':>9}{'ms/프레임':>11}")
    for name, hit, count, ms in rows:
        cover = len(hit) / len(union) * 100 if union else 0.0
        print(f"{name:<40}{len(hit):>10}{cover:>8.1f}%{count:>9}{ms:>11.1f}")
    print("해석: 커버율 = (그 설정이 얼굴을 찾은 프레임) ÷ (어느 설정이든 찾은 프레임). 정답 라벨이 없으니 오탐은 세지 못한다.")
    print("      커버율이 오르는 설정이 있으면 render --draw 로 그 구간을 눈으로 확인해 오탐(배경·손)이 늘지 않았는지 본다.")


def cmd_sweep(args: argparse.Namespace) -> None:
    """지정한 프레임 구간에서 파라미터 조합을 비교한다."""
    path = Path(args.video)
    frames = list(iter_frames(path, args.start, args.end, args.step))
    if not frames:
        raise SystemExit("지정한 구간에서 읽은 프레임이 없습니다.")
    _run_grid(Detectors(use_insight=not args.no_insight, tta=args.tta), frames, args)


def cmd_render(args: argparse.Namespace) -> None:
    """검출 -> ID 이어 붙이기 -> 선형 보간 후 모자이크(또는 박스 표시) 영상을 저장한다."""
    from src.mosaic import apply_mosaic
    from src.video import merge_audio_h264

    src, dst = Path(args.video), Path(args.output)
    if src.resolve() == dst.resolve():
        raise SystemExit("출력 경로는 입력 영상과 달라야 합니다.")
    frames = collect(src, args)
    frames, stitched, filled = fill_gaps(frames, args.max_gap)
    print(f"이어 붙일 수 있는 track {stitched}개, 보간으로 채운 박스 {filled}개 (track_id 는 바꾸지 않는다)")

    fps, _, width, height = video_info(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="triple_y_debug_") as tmp:
        mid = Path(tmp) / "mid.mp4"
        writer = cv2.VideoWriter(str(mid), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise SystemExit("처리 영상을 저장할 코덱을 열 수 없습니다.")
        try:
            for idx, frame in iter_frames(src, end=len(frames)):
                dets = frames[idx]
                if args.draw:
                    frame = frame.copy()
                    for d in dets:
                        x1, y1, x2, y2 = (int(v) for v in d.xyxy)
                        color = (0, 165, 255) if d.src == "interp" else ((255, 128, 0) if d.track_id is None else (0, 200, 0))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"{d.track_id if d.track_id is not None else '-'}:{d.src} {d.conf:.2f}",
                                    (x1, max(y1 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                    writer.write(frame)
                else:
                    boxes = [{"xyxy": tuple(int(v) for v in d.xyxy), "conf": float(d.conf), "track_id": d.track_id} for d in dets]
                    writer.write(apply_mosaic(frame, boxes))
        finally:
            writer.release()
        merge_audio_h264(str(mid), str(src), str(dst))
    print(f"저장: {dst}")


# ============================================================
# ⑧ 명령줄
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    """명령줄 옵션. 기본값은 src/config.py 값을 따른다."""
    p = argparse.ArgumentParser(description="옆얼굴 미검출·ID 튐 진단 도구", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("video", help="입력 영상 (읽기만 한다)")
        sp.add_argument("--conf", type=float, default=CONF, help="추적기 1단계/새 track 점수, 이 미만이고 track 도 못 받으면 버림")
        sp.add_argument("--low-conf", type=float, default=TRACKER["track_low_thresh"], help="YOLO 원시 탐지 하한 = 추적기 2단계 점수. 옆얼굴의 낮은 점수를 살린다")
        sp.add_argument("--imgsz", type=int, default=640, help="YOLO 입력 크기(px, 32의 배수). 클수록 작은 얼굴·옆얼굴에 유리, 느림")
        sp.add_argument("--det-thresh", type=float, default=DET_THRESH, help="SCRFD 탐지 신뢰도 하한")
        sp.add_argument("--det-size", type=int, default=DET_SIZE, help="SCRFD 입력 크기(px). 클수록 작은 얼굴에 유리, 느림")
        sp.add_argument("--no-insight", action="store_true", help="InsightFace 를 끄고 YOLO 만 쓴다")
        sp.add_argument("--tta", action="store_true", help="YOLO 테스트 시점 증강(augment). 느리지만 옆얼굴을 더 찾을 수 있다")
        sp.add_argument("--fuse-iou", type=float, default=0.4, help="YOLO·SCRFD 박스를 같은 얼굴로 보는 IoU")
        sp.add_argument("--track-buffer", type=int, default=TRACKER["track_buffer"], help="사라진 track 을 기억하는 프레임 수")
        sp.add_argument("--match-thresh", type=float, default=TRACKER["match_thresh"], help="track 연결 기준 (높을수록 너그러움)")
        sp.add_argument("--max-frames", type=int, default=0, help="앞에서 이 프레임까지만 처리 (0=끝까지)")
        sp.add_argument("--max-gap", type=int, default=15, help="이어 붙이기·보간을 허용하는 최대 공백(프레임)")

    lg = sub.add_parser("log", help="프레임별 진단 로그")
    common(lg)
    lg.add_argument("--all", action="store_true", help="이상 없는 프레임도 모두 출력")
    lg.add_argument("--warn-conf", type=float, default=0.5, help="이 미만이면 LOW 표시")
    lg.add_argument("--jump-iou", type=float, default=0.3, help="같은 ID 의 직전 박스와 IoU 가 이 미만이면 BOX_JUMP")
    lg.add_argument("--switch-iou", type=float, default=0.3, help="사라진 track 자리에 새 ID 가 생기면 ID_SWITCH? 로 보는 IoU")
    lg.add_argument("--json", help="프레임별 결과를 표준 JSON 으로 저장할 경로")
    lg.set_defaults(func=cmd_log)

    sw = sub.add_parser("sweep", help="파라미터 조합 비교")
    sw.add_argument("video")
    sw.add_argument("--start", type=int, default=0)
    sw.add_argument("--end", type=int, default=None)
    sw.add_argument("--step", type=int, default=1, help="n 프레임마다 하나만 쓴다")
    sw.add_argument("--sweep-conf", type=float, nargs="+", default=[0.25, 0.15, 0.1, 0.05])
    sw.add_argument("--sweep-imgsz", type=int, nargs="+", default=[640, 960, 1280])
    sw.add_argument("--sweep-det-thresh", type=float, nargs="+", default=[0.5, 0.3, 0.2, 0.1])
    sw.add_argument("--sweep-det-size", type=int, nargs="+", default=[320, 640, 960])
    sw.add_argument("--no-insight", action="store_true")
    sw.add_argument("--tta", action="store_true")
    sw.set_defaults(func=cmd_sweep)

    rd = sub.add_parser("render", help="이어 붙이기 + 보간 적용 영상")
    common(rd)
    rd.add_argument("output", help="출력 영상 경로 (입력과 달라야 함)")
    rd.add_argument("--draw", action="store_true", help="모자이크 대신 박스를 그린다 (초록=검출, 주황=보간, 파랑=ID 없음)")
    rd.set_defaults(func=cmd_render)
    return p


def main() -> None:
    """명령줄 진입점."""
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
