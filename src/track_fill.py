"""영상에서 끊긴 얼굴 track 이어 붙이기·선형 보간 (B 조영준).

옆얼굴·가림으로 탐지가 몇 프레임 끊겨도 앞뒤에 잡힌 박스를 이어 모자이크가 끊기지 않게 한다.
앞뒤 박스를 모두 알아야 하므로 영상 전체를 먼저 탐지한 뒤(1패스) 박스 목록만 가지고 처리한다.
process_video 가 사용한다. 프레임 이미지는 다루지 않는다.

  from src.track_fill import stitch_tracks, interpolate_gaps

개인정보 원칙: Box 의 track_id 는 절대 바꾸지 않는다. 인물 제외(src/exclude.py)는 track_id 별 판정을 캐시하므로,
잘못 이어 붙인 다른 사람이 등록 인물의 '제외' 판정을 물려받아 노출되는 일이 없어야 한다.
이어 붙인 대응표(alias)는 보간 때 앞뒤 박스를 짝지을 때와 고유 인원 수를 셀 때만 쓴다.
"""
from __future__ import annotations

from math import hypot
from typing import Optional, Tuple

from src.config import FILL_DUP_IOU, FILL_MAX_SHIFT, MAX_FILL_GAP, STITCH_MIN_SIZE_RATIO
from src.types import Box

XYXY = Tuple[int, int, int, int]


def _iou(a: XYXY, b: XYXY) -> float:
    """두 박스의 교차 영역 비율(IoU)."""
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _shift_ratio(a: XYXY, b: XYXY) -> float:
    """두 박스 중심 사이 거리 ÷ 앞 박스의 긴 변. 얼굴이 박스 크기의 몇 배만큼 움직였는지."""
    dx = (a[0] + a[2]) / 2 - (b[0] + b[2]) / 2
    dy = (a[1] + a[3]) / 2 - (b[1] + b[3]) / 2
    return hypot(dx, dy) / max(a[2] - a[0], a[3] - a[1], 1)


def _size_ratio(a: XYXY, b: XYXY) -> float:
    """두 박스 넓이의 비 (작은 쪽 ÷ 큰 쪽)."""
    area_a = max((a[2] - a[0]) * (a[3] - a[1]), 1)
    area_b = max((b[2] - b[0]) * (b[3] - b[1]), 1)
    return min(area_a, area_b) / max(area_a, area_b)


def _cut_between(cuts: frozenset, start: int, end: int) -> bool:
    """start 프레임과 end 프레임 사이(start < 컷 <= end)에 장면 전환이 있으면 True. cuts 는 새 장면이 시작하는 프레임 번호 모음이다."""
    return any(start < c <= end for c in cuts)


def stitch_tracks(
    frames: list[list[Box]],
    max_gap: int = MAX_FILL_GAP,
    max_shift: float = FILL_MAX_SHIFT,
    min_size_ratio: float = STITCH_MIN_SIZE_RATIO,
    cuts: frozenset = frozenset(),
) -> dict[int, int]:
    """끊겨서 새 ID 가 된 track 을 옛 track 에 이어 붙일 대응표 {새 ID: 옛 ID} 를 만든다. 박스는 바꾸지 않는다.

    새 track 이 시작하는 자리가 '이미 끝난 track 의 마지막 박스'에서 max_gap 프레임 안에 있고, 프레임당
    max_shift(박스 긴 변의 비율) 이하로만 움직였고, 크기가 비슷하면 같은 사람으로 본다.
    시간이 겹치는(동시에 존재한) track 끼리는 잇지 않는다. 후보가 여럿이면 가장 가까운 하나만 고른다.
    애매하면 잇지 않는 쪽으로 정했다. 장면 전환(cuts)을 가로지르는 연결도 하지 않는다(다른 장면의 다른 위치일 뿐이다). 체인(A->B->C)이면 모두 맨 앞 ID 로 연결된다.
    """
    first: dict[int, tuple[int, XYXY]] = {}
    last: dict[int, tuple[int, XYXY]] = {}
    for f, boxes in enumerate(frames):
        for box in boxes:
            tid = box["track_id"]
            if tid is not None:
                first.setdefault(tid, (f, box["xyxy"]))
                last[tid] = (f, box["xyxy"])

    alias: dict[int, int] = {}
    open_ends: dict[int, tuple[int, XYXY]] = {}    # 이어 붙일 수 있는(이미 끝난) 대표 track -> 마지막 (프레임, 박스)
    for tid in sorted(first, key=lambda t: first[t][0]):
        start_f, start_box = first[tid]
        best: Optional[int] = None
        best_cost = float("inf")
        for canon, (end_f, end_box) in open_ends.items():
            gap = start_f - end_f - 1
            if gap < 0 or gap > max_gap or _cut_between(cuts, end_f, start_f):
                continue
            cost = _shift_ratio(end_box, start_box)
            if cost <= max_shift * (gap + 1) and _size_ratio(end_box, start_box) >= min_size_ratio and cost < best_cost:
                best, best_cost = canon, cost
        if best is not None:
            alias[tid] = best
            del open_ends[best]
            open_ends[best] = last[tid]
        else:
            open_ends[tid] = last[tid]
    return alias


def interpolate_gaps(
    frames: list[list[Box]],
    alias: Optional[dict[int, int]] = None,
    max_gap: int = MAX_FILL_GAP,
    max_shift: float = FILL_MAX_SHIFT,
    dup_iou: float = FILL_DUP_IOU,
    cuts: frozenset = frozenset(),
) -> tuple[list[list[Box]], int]:
    """같은 얼굴이 앞뒤 프레임에서는 보이고 사이가 비었을 때, 빈 프레임에 박스를 선형 보간해 채운다.

    같은 얼굴 = 같은 track_id, 또는 alias 로 이어 붙인 track. 앞 박스와 뒤 박스의 좌표·점수를 (k / (gap+1)) 비율로 섞는다.
    같은 track_id 안의 공백은 그 track_id 를 달고, ID 가 바뀌는 공백(이어 붙인 경우)은 track_id=None 으로 둔다.
    (인물 제외가 다른 track 의 판정을 물려받지 않게 하려는 것이다. None 박스는 매 프레임 따로 판정되고, 판정 불가면 모자이크된다)
    채우지 않는 경우: 공백이 max_gap 보다 길다 / 앞뒤 박스가 프레임당 max_shift 보다 멀다 / 그 프레임에 이미 겹치는 박스(IoU>=dup_iou)가 있다 /
    공백 사이에 장면 전환(cuts)이 있다(앞뒤가 서로 다른 장면이라 그 사이를 이으면 허공에 박스가 생긴다).
    반환: (보간이 추가된 프레임별 목록, 채운 박스 수). 보간 박스는 각 프레임 목록의 맨 뒤에 붙는다. 입력은 바꾸지 않는다.
    """
    alias = alias or {}
    observed: dict[int, list[tuple[int, Box]]] = {}
    for f, boxes in enumerate(frames):
        for box in boxes:
            tid = box["track_id"]
            if tid is not None:
                observed.setdefault(alias.get(tid, tid), []).append((f, box))

    out = [list(boxes) for boxes in frames]
    filled = 0
    for obs in observed.values():
        for (f0, b0), (f1, b1) in zip(obs, obs[1:]):
            gap = f1 - f0 - 1
            if gap <= 0 or gap > max_gap or _cut_between(cuts, f0, f1) or _shift_ratio(b0["xyxy"], b1["xyxy"]) > max_shift * (gap + 1):
                continue
            tid = b0["track_id"] if b0["track_id"] == b1["track_id"] else None
            for k in range(1, gap + 1):
                t = k / (gap + 1)
                xyxy = tuple(round(a + (b - a) * t) for a, b in zip(b0["xyxy"], b1["xyxy"]))
                if xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]:
                    continue
                if any(_iou(xyxy, other["xyxy"]) >= dup_iou for other in frames[f0 + k]):
                    continue
                out[f0 + k].append({"xyxy": xyxy, "conf": b0["conf"] + (b1["conf"] - b0["conf"]) * t, "track_id": tid})
                filled += 1
    return out, filled
