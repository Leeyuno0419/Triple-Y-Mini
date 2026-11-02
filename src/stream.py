"""Live video capture, face tracking, and mosaic processing."""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Iterator, Optional
from urllib.parse import urlparse

import cv2
import numpy as np

from src.video import BoxHolder, _SceneCutDetector, _load_exclusion_filter

_DETECT_MAX_SIDE = 640


def resolve_stream_url(url: str) -> str:
    """Resolve YouTube watch/live URLs to a direct media URL; pass other URLs through."""
    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower()
    if host == "youtu.be" or host.endswith("youtube.com"):
        try:
            import yt_dlp
        except ImportError as err:
            raise ImportError("YouTube 주소를 사용하려면 yt-dlp를 설치해야 합니다.") from err
        options = {"quiet": True, "no_warnings": True, "noplaylist": True}
        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                info = ydl.extract_info(url, download=False)
        except yt_dlp.utils.DownloadError as err:
            raise RuntimeError(f"YouTube 스트림 정보를 가져오지 못했습니다: {err}") from err

        stream_url = info.get("url")
        if not stream_url:
            # Some live broadcasts expose video-only formats but yt-dlp's automatic
            # selector reports no match. OpenCV needs only the video track.
            formats = [f for f in info.get("formats", []) if f.get("url") and f.get("vcodec") not in (None, "none")]
            if formats:
                under_720 = [f for f in formats if (f.get("height") or 0) <= 720]
                candidates = under_720 or formats
                chosen = max(candidates, key=lambda f: (f.get("height") or 0, f.get("tbr") or 0))
                stream_url = chosen["url"]
        if not stream_url:
            raise RuntimeError("이 YouTube 영상에서 재생 가능한 영상 스트림을 찾지 못했습니다.")
        return stream_url
    return url


def process_stream(url: str, ref_embs: Optional[np.ndarray] = None) -> Iterator[tuple[np.ndarray, dict]]:
    """Yield mosaiced BGR frames and live counts. The capture worker retains only its newest frame."""
    from src.detector import detect_faces, reset_tracker
    from src.mosaic import apply_mosaic

    source = resolve_stream_url(url)
    filter_excluded = _load_exclusion_filter() if ref_embs is not None else None
    reset_tracker()
    lock = threading.Lock()
    changed = threading.Event()
    # "cut": 처리 루프가 마지막으로 가져간 뒤 장면 전환이 있었는지. 처리 루프는 프레임을 건너뛰므로
    # 판정은 모든 프레임을 읽는 캡처 스레드에서 바로 붙은 프레임끼리 한다 (SCENE_CUT_CORR 가 그 기준).
    state: dict = {"frame": None, "cut": False, "error": None, "done": False}
    stop = threading.Event()
    scene_cut = _SceneCutDetector()

    def capture_loop() -> None:
        failures = 0
        capture = None
        try:
            while not stop.is_set():
                capture = cv2.VideoCapture(source)
                if not capture.isOpened():
                    capture.release()
                    capture = None
                    failures += 1
                    if failures > 3:
                        raise RuntimeError("스트림 연결에 실패했습니다. 주소와 네트워크를 확인해 주세요.")
                    stop.wait(0.5)
                    continue
                source_fps = capture.get(cv2.CAP_PROP_FPS)
                frame_interval = 1.0 / source_fps if 1.0 <= source_fps <= 60.0 else None
                next_frame_at: Optional[float] = None
                while not stop.is_set():
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        break
                    if frame_interval is not None:
                        now = time.perf_counter()
                        if next_frame_at is None or now - next_frame_at > frame_interval * 2:
                            next_frame_at = now + frame_interval
                        else:
                            next_frame_at += frame_interval
                            if stop.wait(max(0.0, next_frame_at - now)):
                                break
                    cut = scene_cut.update(frame)
                    with lock:
                        state["frame"] = frame
                        state["cut"] = state["cut"] or cut
                    changed.set()
                capture.release()
                capture = None
                if stop.is_set():
                    break
                failures += 1
                if failures > 3:
                    raise RuntimeError("스트림이 중단되어 3회 재연결을 시도했지만 실패했습니다.")
                stop.wait(0.5)
        except Exception as err:
            with lock:
                state["error"] = err
        finally:
            if capture is not None:
                capture.release()
            with lock:
                state["done"] = True
            changed.set()

    worker = threading.Thread(target=capture_loop, name="stream-capture", daemon=True)
    worker.start()
    holder = BoxHolder()
    exclude_cache: dict = {}
    track_ids: set[tuple[int, int]] = set()  # (장면 번호, track_id): 장면 전환 때 track_id 가 1 부터 다시 매겨진다
    scene = 0
    frame_idx = 0
    processed = 0
    yield_times: deque[float] = deque(maxlen=15)
    last_frame = None
    try:
        while True:
            changed.wait(0.5)
            changed.clear()
            with lock:
                frame = state["frame"]
                error = state["error"]
                done = state["done"]
                if frame is not None and frame is last_frame:
                    frame = None
                cut = frame is not None and state["cut"]
                if frame is not None:
                    state["cut"] = False
            if error is not None:
                raise error
            if frame is not None:
                last_frame = frame
                # 장면이 바뀌면 이전 장면의 추적·유지 박스·제외 판정을 버린다 (process_video 와 같음).
                if cut:
                    reset_tracker()
                    holder = BoxHolder()
                    exclude_cache.clear()
                    scene += 1
                height, width = frame.shape[:2]
                scale = min(1.0, _DETECT_MAX_SIDE / max(height, width))
                if scale < 1.0:
                    detect_frame = cv2.resize(
                        frame,
                        (max(1, round(width * scale)), max(1, round(height * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                else:
                    detect_frame = frame
                boxes = detect_faces(detect_frame, track=True)
                if scale < 1.0:
                    inv_scale = 1.0 / scale
                    boxes = [
                        {
                            **box,
                            "xyxy": (
                                max(0, round(box["xyxy"][0] * inv_scale)),
                                max(0, round(box["xyxy"][1] * inv_scale)),
                                min(width, round(box["xyxy"][2] * inv_scale)),
                                min(height, round(box["xyxy"][3] * inv_scale)),
                            ),
                        }
                        for box in boxes
                    ]
                track_ids.update((scene, b["track_id"]) for b in boxes if b["track_id"] is not None)
                boxes = holder.update(boxes, frame_idx)
                if filter_excluded is not None:
                    boxes = filter_excluded(frame, boxes, ref_embs, cache=exclude_cache)
                result = apply_mosaic(frame, boxes)
                processed += 1
                frame_idx += 1
                yield_times.append(time.perf_counter())
                elapsed = yield_times[-1] - yield_times[0]
                fps = (len(yield_times) - 1) / elapsed if len(yield_times) > 1 and elapsed > 0 else 0.0
                yield result, {"faces_now": len(boxes), "faces_total": len(track_ids),
                               "fps": fps}
            elif done:
                break
    finally:
        stop.set()
        changed.set()
        worker.join(timeout=2.0)
