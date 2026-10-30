"""인물 등록·제외 (C 나예린). 입출력 규칙은 역할별_구현명세.md C장 ①~④를 따른다.

등록할 사람의 얼굴 벡터(InsightFace)를 만들고(`list_candidates`, `register_person`),
탐지된 얼굴 중 그 사람의 박스를 모자이크 대상에서 빼 준다(`filter_excluded`).

사용법 (프로젝트 루트가 sys.path에 있어야 함):
  from src.exclude import get_face_app, list_candidates, register_person, filter_excluded

insightface·onnxruntime 은 처음 쓸 때(get_face_app) 불러온다. 설치돼 있지 않으면 ImportError 가 나고,
웹 화면(app/app.py)이 이를 받아 안내를 띄운다. 모델(buffalo_l)은 처음 한 번 인터넷에서 내려받는다(~/.insightface).
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Optional, Union

try:    # Windows 에서는 torch(ultralytics)보다 onnx 를 먼저 불러야 한다. 거꾸로면 'DLL load failed' 로 InsightFace 를 못 쓴다
    import onnx  # noqa: F401
except ImportError:    # 설치돼 있지 않으면 get_face_app 에서 안내한다
    pass

import cv2
import numpy as np

from src.config import (
    CANDIDATE_MAX_COUNT,
    CANDIDATE_MAX_FRAMES,
    CANDIDATE_MIN_SCORE,
    CANDIDATE_MIN_SIZE,
    CANDIDATE_SAMPLE_FPS,
    DET_SIZE,
    DET_THRESH,
    EXCLUDE_GRACE_FRAMES,
    JUDGE_CANVAS_SCALE,
    JUDGE_MAX_TURN,
    JUDGE_RETRY_FRAMES,
    MIN_VERIFY_SIZE,
    NEGATIVE_CONFIRM_HITS,
    NEGATIVE_RECHECK_FRAMES,
    PAD,
    RECHECK_FRAMES,
    REF_MAX_EMBS,
    SIM_THRESHOLD,
)
from src.types import Box

if TYPE_CHECKING:    # 타입 표시용으로만 쓴다. 실행할 때는 get_face_app 안에서 불러온다
    from insightface.app import FaceAnalysis

_log = logging.getLogger(__name__)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
_FRAME_KEY = "_frame"    # filter_excluded 의 cache 안에서 지금까지 부른 횟수(프레임 번호)를 세는 자리. track_id(정수)와 겹치지 않는다

_MODULES = ["detection", "recognition"]    # 얼굴 찾기와 벡터 만들기만 쓴다. 랜드마크·성별·나이 모델은 불러오지 않아 빠르고 가볍다
_app: Optional[Any] = None
_app_lock = threading.Lock()


# ============================================================
# ① 얼굴 분석기 (한 번만 만들어 재사용)
# ============================================================

def get_face_app() -> "FaceAnalysis":
    """InsightFace `FaceAnalysis(name="buffalo_l")` 를 한 번만 만들어 돌려준다. 이후에는 같은 객체를 재사용한다.

    GPU(CUDAExecutionProvider)가 있으면 `prepare(ctx_id=0)` 으로 GPU 를, 없거나 GPU 준비에 실패하면
    `ctx_id=-1` 로 CPU 를 쓴다. insightface·onnxruntime 이 없으면 ImportError 를 그대로 올린다.
    여러 스레드가 동시에 처음 부르더라도 분석기는 하나만 만들어진다.
    """
    global _app
    if _app is not None:
        return _app
    with _app_lock:
        if _app is None:
            _app = _build_face_app()
        return _app


def _build_face_app() -> "FaceAnalysis":
    """GPU -> CPU 순서로 분석기를 만든다. get_face_app 이 자물쇠를 잡은 채 한 번만 부른다."""
    import onnxruntime
    from insightface.app import FaceAnalysis

    det_size = (DET_SIZE, DET_SIZE)
    if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
        try:
            app = FaceAnalysis(name="buffalo_l", allowed_modules=_MODULES, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            app.prepare(ctx_id=0, det_thresh=DET_THRESH, det_size=det_size)
            return app
        except Exception:    # CUDA·cuDNN 버전이 안 맞는 등. 멈추지 말고 CPU 로 이어 간다
            _log.warning("GPU 로 InsightFace 를 준비하지 못해 CPU 로 바꿉니다.", exc_info=True)
    app = FaceAnalysis(name="buffalo_l", allowed_modules=_MODULES, providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_thresh=DET_THRESH, det_size=det_size)
    return app


# ============================================================
# 보조 얼굴 탐지 (박스만)
# ============================================================

def detect_faces_insight(img: np.ndarray) -> list[Box]:
    """InsightFace 탐지기로 얼굴 박스만 빠르게 찾는다. 얼굴 벡터는 만들지 않는다. 얼굴이 없으면 [].

    A파트 detect_faces(YOLO)가 놓치는 작고 흐린 얼굴을 보완하는 용도다. DET_THRESH 이상이면 모두 돌려주고,
    좌표는 이미지 범위 안의 정수이며 track_id 는 None 이다. (WIDER 40장 기준 YOLO 91% -> 이 탐지기 97%)
    """
    app = get_face_app()
    h, w = img.shape[:2]
    try:
        bboxes, _ = app.det_model.detect(img, max_num=0, metric="default")    # 탐지만 한다 (app.get 은 얼굴마다 벡터까지 만들어 느리다)
        raw = [(b[0], b[1], b[2], b[3], b[4]) for b in bboxes]
    except (AttributeError, TypeError):    # insightface 버전에 따라 탐지기 호출 모양이 다를 때는 get 으로 대신한다
        raw = [(*f.bbox[:4], float(getattr(f, "det_score", 1.0))) for f in app.get(img)]
    boxes: list[Box] = []
    for x1, y1, x2, y2, score in raw:
        x1, y1 = max(int(np.floor(x1)), 0), max(int(np.floor(y1)), 0)
        x2, y2 = min(int(np.ceil(x2)), w), min(int(np.ceil(y2)), h)
        if x2 > x1 and y2 > y1:
            boxes.append({"xyxy": (x1, y1, x2, y2), "conf": float(score), "track_id": None})
    return boxes


# ============================================================
# 공용 도우미
# ============================================================

def _face_box(face: Any, width: int, height: int) -> tuple[int, int, int, int]:
    """InsightFace 얼굴의 bbox 를 이미지 범위 안의 정수 (x1, y1, x2, y2) 로 바꾼다."""
    x1, y1, x2, y2 = (float(v) for v in face.bbox[:4])
    return (max(int(np.floor(x1)), 0), max(int(np.floor(y1)), 0),
            min(int(np.ceil(x2)), width), min(int(np.ceil(y2)), height))


def _face_embedding(face: Any) -> Optional[np.ndarray]:
    """얼굴의 길이 1 짜리(정규화된) 임베딩 벡터를 돌려준다. 벡터가 없거나 0 이면 None."""
    emb = getattr(face, "normed_embedding", None)
    if emb is None:
        emb = getattr(face, "embedding", None)
    if emb is None:
        return None
    emb = np.asarray(emb, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(emb))
    return emb / norm if norm > 0 else None


def _unit_rows(embs: Any) -> np.ndarray:
    """(k, d) 벡터들을 행마다 길이 1 로 맞춘다. 길이가 0 인 행은 뺀다."""
    arr = np.asarray(embs, dtype=np.float32)
    arr = arr.reshape(1, -1) if arr.ndim == 1 else arr.reshape(-1, arr.shape[-1])
    norms = np.linalg.norm(arr, axis=1)
    return arr[norms > 0] / norms[norms > 0, None]


def _crop(img: np.ndarray, xyxy: tuple[int, int, int, int], pad: float = PAD) -> np.ndarray:
    """박스에 pad 비율만큼 여백을 더해 이미지 범위로 자른 사진을 돌려준다. (복사본이 아닌 뷰)"""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = xyxy
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    return img[max(int(y1 - py), 0):min(int(np.ceil(y2 + py)), h), max(int(x1 - px), 0):min(int(np.ceil(x2 + px)), w)]


def _iter_frames(src: str) -> Iterator[np.ndarray]:
    """사진이면 그 사진 한 장을, 영상이면 초당 CANDIDATE_SAMPLE_FPS 프레임(최대 CANDIDATE_MAX_FRAMES 장)을 BGR 로 내보낸다.

    Windows 에서 한글 경로도 읽히도록 사진은 바이트로 읽어 디코딩한다. 읽을 수 없으면 ValueError.
    """
    path = Path(src)
    ext = path.suffix.lower()
    if ext in _IMAGE_EXTS:
        img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("이미지를 읽을 수 없습니다. 파일이 깨졌는지 확인해 주세요.")
        yield img
        return
    if ext not in _VIDEO_EXTS:
        raise ValueError(f"지원하지 않는 파일 형식입니다: {ext or '(확장자 없음)'}")

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError("영상을 열 수 없습니다. 파일이 깨졌는지 확인해 주세요.")
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 0 and np.isfinite(fps) else 30.0
        step = max(1, round(fps / CANDIDATE_SAMPLE_FPS))
        index, taken = 0, 0
        while taken < CANDIDATE_MAX_FRAMES and cap.grab():    # grab 은 디코딩만 하고 변환하지 않아 빠르다
            if index % step == 0:
                ok, frame = cap.retrieve()
                if ok:
                    yield frame
                    taken += 1
            index += 1
    finally:
        cap.release()


# ============================================================
# ② 후보 찾기 / ③ 등록
# ============================================================

def list_candidates(src: Union[str, list[str]]) -> list[dict]:
    """사진 또는 영상에서 등장하는 사람(후보)을 찾아 점수 높은 순으로 돌려준다. 얼굴이 하나도 없으면 [].

    src 는 파일 경로 하나이거나, 사진 경로 목록이다. 사진을 여러 장 주면(정면 1장 + 좌우 측면 각 1장처럼) 사진마다 한 프레임으로
    보고, 서로 닮은 얼굴은 한 사람(후보 하나)으로 묶어서 그 사람의 여러 각도 얼굴 벡터를 모은다.

    반환: [{"thumb": 대표 얼굴(BGR), "embs": (k, d) 벡터(행마다 길이 1), "frames": 등장 횟수, "score": float}, ...]
    영상은 초당 CANDIDATE_SAMPLE_FPS 프레임만 보고, 얼굴 벡터의 유사도가 SIM_THRESHOLD 이상이면 같은 사람으로 묶는다.
    같은 프레임의 서로 다른 얼굴은 같은 사람으로 묶지 않는다. score 는 등장 횟수 × 평균 얼굴 크기(√(가로×세로))다.
    짧은 변이 CANDIDATE_MIN_SIZE 미만이거나 탐지 신뢰도가 CANDIDATE_MIN_SCORE 미만인 얼굴은 후보에서 뺀다(너무 작고 흐려 사람을 알아보기 어렵다).
    MIN_VERIFY_SIZE 보다 작은 얼굴도 후보로는 보이지만, 영상·사진에서 제외 판정을 할 때는 그 크기 미만이면 인식 없이 모자이크한다.
    """
    groups: list[dict] = []
    sources = [str(src)] if isinstance(src, (str, Path)) else [str(path) for path in src]
    for frame in (frame for source in sources for frame in _iter_frames(source)):
        h, w = frame.shape[:2]
        used: set = set()    # 이 프레임에서 이미 얼굴이 배정된 묶음
        for face in sorted(get_face_app().get(frame), key=lambda f: _quality(f, w, h), reverse=True):
            x1, y1, x2, y2 = _face_box(face, w, h)
            emb = _face_embedding(face)
            if emb is None or min(x2 - x1, y2 - y1) < CANDIDATE_MIN_SIZE or float(getattr(face, "det_score", 1.0)) < CANDIDATE_MIN_SCORE:
                continue
            quality = _quality(face, w, h)
            size = float(np.sqrt((x2 - x1) * (y2 - y1)))
            gi = _best_group(groups, emb, used)
            if gi is None:
                groups.append({"members": [], "frames": 0, "size_sum": 0.0, "vec_sum": np.zeros_like(emb), "best": (-1.0, None)})
                gi = len(groups) - 1
            group = groups[gi]
            group["members"].append((quality, emb))
            group["frames"] += 1
            group["size_sum"] += size
            group["vec_sum"] = group["vec_sum"] + emb
            if quality > group["best"][0]:
                group["best"] = (quality, _crop(frame, (x1, y1, x2, y2)).copy())
            used.add(gi)

    groups = _merge_groups(groups)
    candidates = []
    for group in groups:
        members = _pick_diverse(group["members"], REF_MAX_EMBS)    # 또렷하되 각도가 서로 다른 얼굴을 남긴다 (옆얼굴도 알아보려면 정면만으로는 모자란다)
        candidates.append({
            "thumb": group["best"][1],
            "embs": np.stack([emb for _, emb in members]).astype(np.float32),
            "frames": group["frames"],
            "score": float(group["size_sum"]),    # = 등장 횟수 × 평균 얼굴 크기
        })
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[:CANDIDATE_MAX_COUNT]


def _quality(face: Any, width: int, height: int) -> float:
    """얼굴이 얼마나 또렷한지 점수: 탐지 신뢰도 × 얼굴 크기. 큰 얼굴을 대표 사진과 저장 벡터로 먼저 고르는 데 쓴다."""
    x1, y1, x2, y2 = _face_box(face, width, height)
    return float(getattr(face, "det_score", 1.0)) * float(np.sqrt(max(0, x2 - x1) * max(0, y2 - y1)))


def _pick_diverse(members: list[tuple[float, np.ndarray]], limit: int) -> list[tuple[float, np.ndarray]]:
    """묶음의 얼굴 벡터 중 limit 개를 고른다. 가장 또렷한 것부터 시작해, 이미 고른 것들과 가장 다른 것을 차례로 더한다.

    또렷한 순으로만 고르면 거의 같은 정면 얼굴만 남아서, 고개를 돌린 얼굴을 같은 사람으로 알아보지 못한다.
    다만 흐린 얼굴이 뽑히지 않도록 또렷한 쪽 절반(최소 limit 의 두 배) 안에서만 고른다.
    """
    ranked = sorted(members, key=lambda m: m[0], reverse=True)
    if len(ranked) <= limit:
        return ranked
    pool = ranked[: max(limit * 2, len(ranked) // 2)]
    chosen = [pool[0]]
    rest = pool[1:]
    while len(chosen) < limit and rest:
        far = min(range(len(rest)), key=lambda i: max(float(rest[i][1] @ c[1]) for c in chosen))    # 고른 것들과의 최대 유사도가 가장 낮은 것
        chosen.append(rest.pop(far))
    return chosen


def _centroid(vec_sum: np.ndarray) -> np.ndarray:
    """벡터 합을 길이 1 로 맞춘 묶음의 중심 벡터."""
    norm = float(np.linalg.norm(vec_sum))
    return vec_sum / norm if norm > 0 else vec_sum


def _best_group(groups: list[dict], emb: np.ndarray, used: set) -> Optional[int]:
    """emb 와 가장 닮은 묶음 번호. SIM_THRESHOLD 미만이거나 이 프레임에서 이미 쓴 묶음뿐이면 None."""
    best, best_sim = None, SIM_THRESHOLD
    for gi, group in enumerate(groups):
        if gi in used:
            continue
        sim = float(_centroid(group["vec_sum"]) @ emb)
        if sim >= best_sim:
            best, best_sim = gi, sim
    return best


def _merge_groups(groups: list[dict]) -> list[dict]:
    """중심 벡터가 SIM_THRESHOLD 이상 닮은 묶음끼리 합친다. 고개를 돌려서 따로 묶였던 같은 사람을 하나로 모으려는 것이다."""
    merged = True
    while merged:
        merged = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if float(_centroid(groups[i]["vec_sum"]) @ _centroid(groups[j]["vec_sum"])) >= SIM_THRESHOLD:
                    a, b = groups[i], groups[j]
                    a["members"] += b["members"]
                    a["frames"] += b["frames"]
                    a["size_sum"] += b["size_sum"]
                    a["vec_sum"] = a["vec_sum"] + b["vec_sum"]
                    a["best"] = max(a["best"], b["best"], key=lambda t: t[0])
                    del groups[j]
                    merged = True
                    break
            if merged:
                break
    return groups


def register_person(src: Union[str, list[str]], index: int = 0) -> np.ndarray:
    """사진·영상에서 찾은 후보 중 index 번째 사람의 얼굴 벡터(k, d)를 돌려준다. 기본 0 은 가장 크게·오래 나온 사람이다.

    후보가 없거나 index 가 범위를 벗어나면 ValueError.
    """
    candidates = list_candidates(src)
    if not candidates:
        raise ValueError("얼굴을 찾지 못했습니다. 정면 얼굴이 잘 보이는 파일로 다시 시도해 주세요.")
    if not 0 <= index < len(candidates):
        raise ValueError(f"index 는 0~{len(candidates) - 1} 이어야 합니다: {index}")
    return candidates[index]["embs"]


# ============================================================
# ④ 제외 인물 걸러내기
# ============================================================

def filter_excluded(
    img: np.ndarray,
    boxes: list[Box],
    ref_embs: np.ndarray,
    threshold: float = SIM_THRESHOLD,
    cache: Optional[dict] = None,
) -> list[Box]:
    """boxes 중 모자이크할 박스만 남겨 돌려준다. 등록 인물(ref_embs)로 판정된 박스는 뺀다. 입력 박스는 바꾸지 않는다.

    cache 는 track_id 별 판정을 담는 dict 로, 영상마다 새 빈 dict 를 만들어 프레임마다 같은 것을 넘긴다. (사진은 None)
    cache 안의 모양(C가 정함): cache[track_id] = {"exclude": 제외 여부(True/False, 옆얼굴이라 미룬 중이면 None), "checked": 판정한 프레임 번호,
    "ok": 제외로 확실히 확인한 마지막 프레임, "hits": '다른 사람' 판정 뒤 재확인에서 연속 통과한 횟수}, cache["_frame"] = 부른 횟수.
    박스마다 위에서부터 차례로 본다. 애매하면 언제나 모자이크하는 쪽(남기는 쪽)으로 정한다.
      1. 짧은 변이 MIN_VERIFY_SIZE 미만: 인식하지 않고 모자이크한다. 캐시에도 저장하지 않는다.
      2. track_id 가 있는(영상) 박스가 다른 박스와 겹침(IoU > 0): 그 track 의 캐시를 지우고 이번 프레임은 모자이크한다.
         (겹치는 순간 track_id 가 서로 바뀌면 모르는 사람이 '제외' 판정을 넘겨받을 수 있기 때문이다.
         사진은 track_id 가 없어 이런 위험이 없으므로, 얼굴이 겹쳐 있어도 이 규칙 없이 바로 판정한다)
      3. 캐시에 "제외"는 RECHECK_FRAMES, "모자이크"(정면에서 다른 사람으로 봄)는 NEGATIVE_RECHECK_FRAMES 프레임마다 다시 판정한다.
         모자이크 쪽은 재확인에서 NEGATIVE_CONFIRM_HITS 번 연속 통과해야 제외로 뒤집는다. 옆얼굴이라 미룬 track 은 JUDGE_RETRY_FRAMES 프레임마다 다시 판정한다. (옆얼굴은 같은 사람도 유사도가 크게 떨어져서, 그 순간의
         낮은 값으로 '다른 사람'을 확정하면 정면이 나와도 영영 모자이크되기 때문이다)
      4. 캐시에 없으면 판정하고, track_id 가 있으면 결과를 캐시에 저장한다. 제외로 확정된 track 이 고개를 돌려 다시 판정이 미뤄지면
         마지막 확실한 판정 뒤 EXCLUDE_GRACE_FRAMES 프레임까지는 제외를 유지한다.
    판정: 박스를 여백(PAD)과 함께 잘라 얼굴 벡터를 구하고, ref_embs 와의 코사인 유사도 최댓값이 threshold 이상이면 제외.
    잘라낸 곳에서 얼굴을 못 찾거나 벡터를 못 만들면 제외하지 않는다(모자이크 유지). 이런 '판정 불가'는 캐시에 저장하지 않아
    다음 프레임에 다시 시도한다.
    """
    frame_no = 0
    if cache is not None:
        frame_no = cache.get(_FRAME_KEY, 0) + 1
        cache[_FRAME_KEY] = frame_no
    ref = _unit_rows(ref_embs) if ref_embs is not None and len(ref_embs) else np.zeros((0, 0), np.float32)
    if not boxes or len(ref) == 0:    # 등록한 사람이 없으면 아무도 제외하지 않는다
        return list(boxes)

    get_face_app()    # 모델이 없으면(ImportError) 여기서 바로 알린다. 아래 판정 중의 오류는 '모자이크 유지'로 처리한다
    kept: list[Box] = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box["xyxy"]
        track_id = box.get("track_id")

        if min(x2 - x1, y2 - y1) < MIN_VERIFY_SIZE:                  # 1. 작은 얼굴
            kept.append(box)
            continue
        if track_id is not None and _overlaps_other(box, boxes, i):    # 2. 겹친 얼굴 (추적 ID 가 바뀔 수 있는 영상에서만)
            if cache is not None:
                cache.pop(track_id, None)
            kept.append(box)
            continue

        excluded = _cached_verdict(cache, track_id, frame_no)         # 3. 캐시
        if excluded is None:                                          # 4. 새로 판정
            excluded, turned = _judge_face(img, box, ref, threshold)
            if cache is not None and track_id is not None:
                prev = cache.get(track_id)
                if prev is not None and prev["exclude"] is False:     # '다른 사람'으로 봤던 track 의 재확인: 연속으로 통과해야 뒤집는다
                    hits = prev.get("hits", 0) + 1 if excluded is True else (0 if excluded is False else prev.get("hits", 0))
                    if hits >= NEGATIVE_CONFIRM_HITS:
                        cache[track_id] = {"exclude": True, "checked": frame_no, "ok": frame_no}
                    else:
                        prev.update(checked=frame_no, hits=hits)      # 아직은 모자이크 유지, 다음 재확인에서 다시 본다
                        excluded = None
                elif excluded is not None:                            # 확실한 판정: 제외는 '확실했던 프레임'(ok)을 기록해 둔다
                    cache[track_id] = {"exclude": excluded, "checked": frame_no, "ok": frame_no if excluded else None}
                elif turned:                                          # 옆얼굴이라 미룸
                    if prev is not None and prev["exclude"] and prev.get("ok") is not None and frame_no - prev["ok"] <= EXCLUDE_GRACE_FRAMES:
                        excluded = True                               # 방금까지 등록 인물로 확인된 track 이 고개를 돌렸을 뿐이다 -> 잠깐 제외 유지
                        prev["checked"] = frame_no - RECHECK_FRAMES + JUDGE_RETRY_FRAMES
                    else:
                        cache[track_id] = {"exclude": None, "checked": frame_no, "ok": None}
        if excluded is not True:                                      # 제외로 확정된 것만 뺀다
            kept.append(box)
    return kept


def _overlaps_other(box: Box, boxes: list[Box], index: int) -> bool:
    """box 가 다른 박스와 겹치는 넓이가 0 보다 크면(IoU > 0) True. 모서리만 맞닿은 것은 겹침이 아니다."""
    x1, y1, x2, y2 = box["xyxy"]
    for j, other in enumerate(boxes):
        if j == index:
            continue
        ox1, oy1, ox2, oy2 = other["xyxy"]
        if min(x2, ox2) - max(x1, ox1) > 0 and min(y2, oy2) - max(y1, oy1) > 0:
            return True
    return False


def _cached_verdict(cache: Optional[dict], track_id: Optional[int], frame_no: int) -> Optional[bool]:
    """캐시로 정할 수 있으면 제외 여부(True/False)를, 새로 판정해야 하면 None 을 돌려준다."""
    if cache is None or track_id is None:
        return None
    entry = cache.get(track_id)
    if entry is None:
        return None
    if entry["exclude"] is None:                      # 옆얼굴이라 판정을 미룬 track: 잠깐은 모자이크하고, 간격이 지나면 다시 판정한다
        return False if frame_no - entry["checked"] < JUDGE_RETRY_FRAMES else None
    # 제외 판정도, '다른 사람' 판정도 주기적으로 다시 확인한다 (한 프레임의 흐림·각도 때문에 틀렸을 수 있다)
    interval = RECHECK_FRAMES if entry["exclude"] else NEGATIVE_RECHECK_FRAMES
    if frame_no - entry["checked"] >= interval:
        return None
    return bool(entry["exclude"])


def _face_turn(face: Any) -> Optional[float]:
    """얼굴이 얼마나 옆으로 돌았는지: 코가 두 눈 가운데에서 벗어난 정도 ÷ 눈 사이 거리. 정면은 0 에 가깝다. 랜드마크가 없으면 None."""
    kps = getattr(face, "kps", None)
    if kps is None or len(kps) < 3:
        return None
    left_eye, right_eye, nose = kps[0], kps[1], kps[2]
    gap = abs(float(right_eye[0]) - float(left_eye[0]))
    if gap < 1.0:    # 두 눈이 거의 한 점으로 겹치면 완전한 옆얼굴이다
        return float("inf")
    return (float(nose[0]) - (float(left_eye[0]) + float(right_eye[0])) / 2) / gap


def _on_canvas(crop: np.ndarray) -> np.ndarray:
    """잘라낸 얼굴을 JUDGE_CANVAS_SCALE 배 큰 회색 정사각형 캔버스 한가운데에 놓는다.

    InsightFace 탐지기는 얼굴이 사진을 꽉 채우면 거의 못 찾는다(여백 15% 크롭 그대로는 표본의 8%만 성공,
    4배 캔버스는 90% 성공). 그래서 얼굴이 화면의 1/4 정도로 보이게 키운 캔버스에 놓고 다시 찾는다.
    """
    h, w = crop.shape[:2]
    side = int(max(h, w) * JUDGE_CANVAS_SCALE)
    canvas = np.full((side, side, 3), 128, np.uint8)
    y0, x0 = (side - h) // 2, (side - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = crop
    return canvas


def _judge_face(img: np.ndarray, box: Box, ref: np.ndarray, threshold: float) -> tuple[Optional[bool], bool]:
    """박스 안의 얼굴이 등록 인물인지 판정한다. (판정, 옆얼굴이라 미룸)을 돌려준다.

    판정: 등록 인물이면 True, 아니면 False, 판단할 수 없으면 None.
    유사도가 임계값 이상이면 얼굴이 돌아가 있어도 True 다. 임계값 미만인데 얼굴이 JUDGE_MAX_TURN 보다 많이 돌아가 있으면
    옆얼굴은 같은 사람도 유사도가 크게 떨어지므로 '다른 사람'으로 단정하지 않고 (None, True) 로 판정을 미룬다. 얼굴을 못 찾았거나
    오류가 났을 때는 (None, False). 어느 경우든 None 이면 호출하는 쪽이 모자이크를 유지한다.
    """
    try:
        crop = _crop(img, tuple(box["xyxy"]))
        if crop.size == 0:
            return None, False
        canvas = _on_canvas(crop)
        faces = get_face_app().get(canvas)
        if not faces:
            return None, False                        # 잘라낸 곳에서 얼굴을 못 찾음 -> 제외하지 않는다
        side = canvas.shape[0]

        def off_center(f: Any) -> float:             # 대상 얼굴은 캔버스 한가운데에 있다. 크롭에 같이 찍힌 이웃 얼굴이 아니라 한가운데 얼굴을 고른
            x1, y1, x2, y2 = _face_box(f, side, side)
            return float(np.hypot((x1 + x2) / 2 - side / 2, (y1 + y2) / 2 - side / 2))

        face = min(faces, key=off_center)
        emb = _face_embedding(face)
        if emb is None:
            return None, False
        if float(np.max(ref @ emb)) >= threshold:
            return True, False
        turn = _face_turn(face)
        if turn is not None and abs(turn) > JUDGE_MAX_TURN:
            return None, True                         # 옆얼굴: 낮은 유사도를 믿을 수 없다 -> 모자이크는 유지하되 나중에 다시 본다
        return False, False
    except Exception:    # 인식 중 오류는 '제외하지 않음'(모자이크 유지)으로 안전하게 처리하고 기록만 남긴다
        _log.exception("인물 판정 중 오류가 나서 이 박스는 모자이크합니다.")
        return None, False


def _judge(img: np.ndarray, box: Box, ref: np.ndarray, threshold: float) -> Optional[bool]:
    """박스 안의 얼굴이 등록 인물이면 True, 아니면 False, 판단할 수 없으면 None 을 돌려준다. (_judge_face 의 판정만)"""
    return _judge_face(img, box, ref, threshold)[0]
