"""Live video capture, face tracking, and mosaic processing."""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, Iterator, Optional, Union
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

import cv2
import numpy as np

from src.config import (
    STREAM_OPEN_TIMEOUT_MS,
    STREAM_READ_TIMEOUT_MS,
    STREAM_RESOLVE_TIMEOUT,
    STREAM_CAPTURE_FRAMES,
    STREAM_CAPTURE_INTERVAL,
)
from src.video import BoxHolder, _SceneCutDetector, _load_exclusion_filter

_DETECT_MAX_SIDE = 640
_MAX_FAILURES = 3    # reconnect attempts before giving up
_ROOT = Path(__file__).resolve().parent.parent

_log = logging.getLogger(__name__)
if not _log.handlers:    # stream problems must be visible in the terminal even when nothing configured logging
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [stream] %(levelname)s %(message)s", "%H:%M:%S"))
    _log.addHandler(_handler)
    _log.setLevel(logging.INFO)
    _log.propagate = False

# What the last lookup used, so a later failure can say whether an outdated yt-dlp is the likely cause.
_resolver: dict = {"engine": None, "version": None}

# (substring of the yt-dlp error text, message shown to the user). The first match wins.
_YTDLP_FAILURES = (
    ("recording is not available", "이 라이브의 다시보기가 없거나 방송이 끝났습니다. 지금 방송 중인 라이브 주소인지 확인해 주세요."),
    ("live event will begin", "아직 시작하지 않은 라이브입니다. 방송이 시작된 뒤 다시 시도해 주세요."),
    ("this live event", "아직 시작하지 않은 라이브입니다. 방송이 시작된 뒤 다시 시도해 주세요."),
    ("private video", "비공개 영상이라 가져올 수 없습니다."),
    ("video is private", "비공개 영상이라 가져올 수 없습니다."),
    ("video is unavailable", "사용할 수 없는 영상입니다. 주소가 잘못됐거나 삭제·비공개된 영상일 수 있습니다."),
    ("video unavailable", "사용할 수 없는 영상입니다. 주소가 잘못됐거나 삭제·비공개된 영상일 수 있습니다."),
    ("members-only", "멤버십 전용 영상이라 가져올 수 없습니다."),
    ("join this channel", "멤버십 전용 영상이라 가져올 수 없습니다."),
    ("sign in to confirm", "유튜브가 로그인(봇 확인)을 요구해 가져오지 못했습니다. 잠시 후 다시 시도하거나 다른 주소를 써 주세요."),
    ("http error 429", "요청이 너무 많아 유튜브가 잠시 막았습니다. 몇 분 뒤에 다시 시도해 주세요."),
    ("unsupported url", "지원하지 않는 주소입니다. 유튜브 영상·라이브 주소를 확인해 주세요."),
    ("timed out", "유튜브 응답이 너무 늦습니다. 네트워크를 확인하고 다시 시도해 주세요."),
    ("getaddrinfo failed", "인터넷에 연결할 수 없습니다. 네트워크를 확인해 주세요."),
    ("name or service not known", "인터넷에 연결할 수 없습니다. 네트워크를 확인해 주세요."),
    ("network is unreachable", "인터넷에 연결할 수 없습니다. 네트워크를 확인해 주세요."),
)
_STATUS_HINTS = {
    403: " (접근 거절: 주소가 만료됐거나 yt-dlp 가 오래됐을 수 있습니다)",
    404: " (주소가 만료됐거나 방송이 끝났습니다)",
    410: " (주소가 만료됐거나 방송이 끝났습니다)",
    429: " (요청이 너무 많습니다)",
}


class StreamError(RuntimeError):
    """A stream failure whose message is Korean and safe to show to the user as is."""


def _is_youtube(url: str) -> bool:
    host = (urlparse(url.strip()).hostname or "").lower()
    return host == "youtu.be" or host.endswith("youtube.com")


def _find_ytdlp_python() -> Optional[str]:
    """Python of a separate environment that has a current yt-dlp (tools/setup_ytdlp.py makes one), or None.

    yt-dlp dropped Python 3.8 after 2024.10.22 and YouTube rejects that version, so a newer Python runs it for us.
    """
    candidates = [os.environ.get("YTDLP_PYTHON", ""), str(_ROOT / ".venv-ytdlp" / "Scripts" / "python.exe"),
                  str(_ROOT / ".venv-ytdlp" / "bin" / "python")]
    return next((path for path in candidates if path and Path(path).is_file()), None)


def _classify(text: str) -> Optional[str]:
    lowered = text.lower()
    return next((message for key, message in _YTDLP_FAILURES if key in lowered), None)


def _tail(text: str, limit: int = 400) -> str:
    text = " ".join(text.split())
    return text[-limit:]


def _extract_external(python: str, url: str) -> dict:
    """Ask a separate, current yt-dlp (JSON on stdout). Raises StreamError when the failure is one the user can act on."""
    command = [python, "-m", "yt_dlp", "-J", "--no-playlist", "--socket-timeout", "15"]
    if shutil.which("node"):    # recent yt-dlp solves YouTube's signature challenge with a JS runtime
        command += ["--js-runtimes", "node"]
    command.append(url)
    try:
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                timeout=STREAM_RESOLVE_TIMEOUT, env=dict(os.environ, PYTHONUTF8="1"))
    except subprocess.TimeoutExpired as err:
        raise StreamError(f"유튜브 주소를 확인하는 데 {STREAM_RESOLVE_TIMEOUT}초가 넘게 걸려 중단했습니다. 네트워크를 확인하고 다시 시도해 주세요.") from err
    except OSError as err:
        raise RuntimeError(f"외부 yt-dlp 를 실행하지 못했습니다: {err}") from err
    for line in result.stderr.splitlines():
        if "WARNING" in line:
            _log.warning("yt-dlp: %s", _tail(line))
    if result.returncode != 0:
        _log.warning("외부 yt-dlp 실패 (종료코드 %d): %s", result.returncode, _tail(result.stderr))
        message = _classify(result.stderr)
        if message:
            raise StreamError(message)
        raise RuntimeError(f"외부 yt-dlp 실패: {_tail(result.stderr, 200)}")
    return json.loads(result.stdout)


def _extract_inprocess(url: str) -> dict:
    """Ask the yt-dlp installed next to the app (Python 3.8's last release, which YouTube may reject)."""
    try:
        import yt_dlp
    except ImportError as err:
        raise ImportError("YouTube 주소를 사용하려면 yt-dlp를 설치해야 합니다.") from err
    messages: list[str] = []

    class _Logger:
        def debug(self, message: str) -> None:
            pass

        def info(self, message: str) -> None:
            pass

        def warning(self, message: str) -> None:
            messages.append(message)

        def error(self, message: str) -> None:
            messages.append(message)

    _resolver["version"] = yt_dlp.version.__version__
    options = {"logger": _Logger(), "noplaylist": True, "socket_timeout": 15}
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=False)
    except yt_dlp.utils.DownloadError as err:
        for message in messages:
            _log.warning("yt-dlp: %s", _tail(message))
        friendly = _classify(str(err))
        raise StreamError(friendly or f"YouTube 스트림 정보를 가져오지 못했습니다: {_tail(str(err), 200)}") from err
    for message in messages:
        _log.warning("yt-dlp: %s", _tail(message))    # e.g. "Precondition check failed" means YouTube rejects this yt-dlp
    return info


def _pick_stream_url(info: dict) -> str:
    """The URL OpenCV should open: yt-dlp's own pick, else the best video-only format up to 720p."""
    if info.get("live_status") == "is_upcoming":
        raise StreamError("아직 시작하지 않은 라이브입니다. 방송이 시작된 뒤 다시 시도해 주세요.")
    stream_url = info.get("url")
    chosen = None
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
        raise StreamError("이 YouTube 영상에서 재생 가능한 영상 스트림을 찾지 못했습니다.")
    if chosen is not None:
        _log.info("선택한 포맷: id=%s %sx%s protocol=%s", chosen.get("format_id"), chosen.get("width"), chosen.get("height"), chosen.get("protocol"))
    return stream_url


def resolve_stream_url(url: str) -> str:
    """Resolve YouTube watch/live URLs to a direct media URL; pass other URLs through.

    A current yt-dlp in a separate environment is tried first; the bundled one is the fallback.
    Failures raise StreamError with a message for the user, and the details go to the log.
    """
    if not _is_youtube(url):
        return url
    started = time.perf_counter()
    info: Optional[dict] = None
    python = _find_ytdlp_python()
    if python is not None:
        try:
            info = _extract_external(python, url)
            _resolver.update(engine="external", version=(info.get("_version") or {}).get("version"))
        except StreamError:
            raise    # the video itself is the problem (private, ended...): the other yt-dlp would say the same
        except Exception as err:    # the helper environment is broken: fall back to the bundled yt-dlp
            _log.warning("최신 yt-dlp 를 쓰지 못해 기본 yt-dlp 로 다시 시도합니다: %s", err)
    else:
        _log.warning("최신 yt-dlp 환경(.venv-ytdlp)이 없습니다. `python tools\\setup_ytdlp.py` 로 만들 수 있습니다. 기본 yt-dlp 로 시도합니다.")
    if info is None:
        info = _extract_inprocess(url)
        _resolver["engine"] = "bundled"
    source = _pick_stream_url(info)
    _log.info("유튜브 주소 해석 완료: engine=%s yt-dlp=%s live=%s host=%s (%.1f초)", _resolver["engine"], _resolver["version"],
              info.get("is_live"), urlparse(source).hostname, time.perf_counter() - started)
    return source


def _outdated_hint() -> str:
    """A pointer to the likely cause when the bundled (Python 3.8) yt-dlp was the one that resolved the URL."""
    version = _resolver.get("version") or ""
    if _resolver.get("engine") == "bundled" and version[:4].isdigit() and int(version[:4]) < 2025:
        return f" 사용 중인 yt-dlp {version} 은 오래되어 유튜브가 거절했을 수 있습니다. `python tools\\setup_ytdlp.py` 로 최신 yt-dlp 를 설치해 보세요."
    return ""


def _http_get(url: str) -> tuple[Optional[int], str]:
    """(HTTP status or None, body text or the error reason) for one probe request."""
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(request, timeout=8) as response:
            return response.status, response.read(200_000).decode("utf-8", "ignore")
    except HTTPError as err:
        return err.code, ""
    except (URLError, OSError, ValueError) as err:
        return None, str(getattr(err, "reason", err))


def _diagnose_open_failure(source: str) -> str:
    """Say where the connection breaks: walks master playlist -> media playlist -> first segment and reports the first HTTP error."""
    if urlparse(source).scheme not in ("http", "https"):
        return "스트림을 열 수 없습니다. 주소와 카메라 전원, 네트워크를 확인해 주세요."
    names = ("스트림 주소", "재생 목록", "영상 조각")
    url = source
    for level in range(3):
        status, body = _http_get(url)
        if status is None:
            return f"{names[level]}에 연결하지 못했습니다 ({body})."
        if status != 200:
            return f"{names[level]} 요청이 HTTP {status} 로 거절됐습니다{_STATUS_HINTS.get(status, '')}."
        if not body.lstrip().startswith("#EXTM3U"):
            return "서버 응답은 정상(200)인데 영상을 열지 못했습니다. 지원하지 않는 형식이거나 일시적인 네트워크 문제일 수 있습니다."
        uri = next((line.strip() for line in body.splitlines() if line.strip() and not line.startswith("#")), None)
        if uri is None:
            return "재생 목록이 비어 있습니다. 방송이 끝났거나 아직 시작하지 않았을 수 있습니다."
        url = urljoin(url, uri)
    return "주소·목록·조각 요청은 모두 정상(200)인데 영상을 열지 못했습니다. 네트워크 지연이나 코덱 문제일 수 있습니다."


def _open_capture(source: str) -> cv2.VideoCapture:
    """Open the stream with bounded open/read waits so a dead stream fails in seconds instead of hanging."""
    params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, STREAM_OPEN_TIMEOUT_MS, cv2.CAP_PROP_READ_TIMEOUT_MSEC, STREAM_READ_TIMEOUT_MS]
    return cv2.VideoCapture(source, cv2.CAP_FFMPEG, params)


def sample_stream_frames(url: str, dest_dir: str, count: int = STREAM_CAPTURE_FRAMES, interval: float = STREAM_CAPTURE_INTERVAL) -> list[str]:
    """Save `count` frames of the stream, about `interval` seconds apart, as JPEG files in `dest_dir` and return their paths.

    Used to find face candidates before the live processing starts. Raises StreamError when no frame arrives.
    """
    source = resolve_stream_url(url)
    capture = _open_capture(source)
    paths: list[str] = []
    try:
        if not capture.isOpened():
            raise StreamError(f"스트림 연결에 실패했습니다. {_diagnose_open_failure(source)}{_outdated_hint()} 주소와 네트워크를 확인해 주세요.")
        folder = Path(dest_dir)
        folder.mkdir(parents=True, exist_ok=True)
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        step = max(1, round((source_fps if 1.0 <= source_fps <= 60.0 else 30.0) * interval))    # frames of video time between two samples, however fast the stream is delivered
        deadline = time.perf_counter() + count * interval * 3 + STREAM_OPEN_TIMEOUT_MS / 1000.0
        read_count = 0
        while len(paths) < count and time.perf_counter() < deadline:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            read_count += 1
            if (read_count - 1) % step:
                continue
            encoded, buffer = cv2.imencode(".jpg", frame)
            if not encoded:
                continue
            path = folder / f"stream_{len(paths):02d}.jpg"
            buffer.tofile(str(path))    # imwrite cannot write a non-ASCII path on Windows
            paths.append(str(path))
    finally:
        capture.release()
    if not paths:
        raise StreamError("스트림에서 장면을 가져오지 못했습니다. 방송 중인 주소인지 확인한 뒤 다시 시도해 주세요.")
    return paths


def _emb_rows(embs: Optional[np.ndarray]) -> set[bytes]:
    """등록 벡터들을 행(사람의 얼굴 벡터 하나) 단위로 비교할 수 있게 바이트 집합으로 만든다. None 이면 빈 집합."""
    return set() if embs is None else {np.ascontiguousarray(row).tobytes() for row in np.asarray(embs)}


def _drop_stale_verdicts(cache: dict, old: Optional[np.ndarray], new: Optional[np.ndarray]) -> None:
    """등록이 old -> new 로 바뀌었을 때 제외 cache 에서 더는 믿을 수 없는 track 판정만 지운다. (cache 모양은 src/exclude.py 의 filter_excluded)

    - 추가만 (new 가 old 의 벡터를 모두 가짐): '제외' 판정은 남긴다. 계속 등록된 사람이 다시 가려지지 않게 한다.
      나머지('다른 사람'·판정 보류)는 새로 등록한 사람일 수 있어 지우고 바로 다시 판정한다.
    - 해제만 (old 가 new 의 벡터를 모두 가짐): '제외' 판정을 지운다. 어느 track 이 해제한 사람인지 모르므로, 남은 사람도 다시 판정한다.
      '다른 사람' 판정은 남긴다 (더 적은 사람과도 맞을 리 없다).
    - 둘 다 바뀜: 모두 지운다.
    """
    old_rows, new_rows = _emb_rows(old), _emb_rows(new)
    if new_rows >= old_rows:
        stale = (False, None)
    elif new_rows <= old_rows:
        stale = (True, None)
    else:
        stale = (True, False, None)
    for key in [k for k, entry in cache.items() if isinstance(entry, dict) and entry["exclude"] in stale]:
        del cache[key]    # cache["_frame"](부른 횟수)는 판정이 아니라 그대로 둔다


def process_stream(
    url: str,
    ref_embs: Union[Optional[np.ndarray], Callable[[], Optional[np.ndarray]]] = None,
    effect: Union[str, Callable[[], str]] = "mosaic",
    should_stop: Optional[Callable[[], bool]] = None,
) -> Iterator[tuple[np.ndarray, dict]]:
    """Yield BGR frames with faces mosaiced (or blurred when effect is "blur") and live counts. The capture worker retains only its newest frame.

    `effect` may be a function returning "mosaic" or "blur"; it is asked again for every frame, so the effect can change while the stream runs.
    `ref_embs` may likewise be a function returning the registered face vectors (or None); when it returns a different object,
    the verdicts that may no longer hold are dropped (see `_drop_stale_verdicts`), so a removed person is masked again from the next frame
    while people who stay registered are not masked again.
    `should_stop` is polled at least every 0.5 s (even when no frame arrives) and ends the generator when it returns True.
    The yielded info dict also carries the untouched source frame as "original" (BGR) for a side-by-side view.
    """
    from src.detector import detect_faces, reset_tracker
    from src.mosaic import apply_blur, apply_mosaic

    def effect_now() -> str:
        name = effect() if callable(effect) else effect
        if name not in ("mosaic", "blur"):
            raise ValueError(f"effect는 'mosaic' 또는 'blur'여야 합니다: {name}")
        return name

    effect_now()    # fail at the start, not on the first frame
    source = resolve_stream_url(url)
    def refs_now() -> Optional[np.ndarray]:
        return ref_embs() if callable(ref_embs) else ref_embs

    last_refs = refs_now()
    filter_excluded = _load_exclusion_filter() if last_refs is not None else None    # fail at the start when registered from the beginning
    reset_tracker()
    lock = threading.Lock()
    changed = threading.Event()
    # "cut": 처리 루프가 마지막으로 가져간 뒤 장면 전환이 있었는지. 처리 루프는 프레임을 건너뛰므로
    # 판정은 모든 프레임을 읽는 캡처 스레드에서 바로 붙은 프레임끼리 한다 (SCENE_CUT_CORR 가 그 기준).
    state: dict = {"frame": None, "cut": False, "error": None, "done": False}
    stop = threading.Event()
    scene_cut = _SceneCutDetector()

    def capture_loop() -> None:
        nonlocal source
        failures = 0
        capture = None
        try:
            while not stop.is_set():
                if failures >= 1 and _is_youtube(url):
                    try:    # signed YouTube URLs expire, so look the live stream up again before reconnecting
                        source = resolve_stream_url(url)
                    except Exception as err:
                        _log.warning("재연결 전에 주소를 다시 확인하지 못해 이전 주소를 씁니다: %s", err)
                capture = _open_capture(source)
                if not capture.isOpened():
                    capture.release()
                    capture = None
                    failures += 1
                    cause = _diagnose_open_failure(source)
                    _log.warning("스트림 열기 실패 (%d/%d): %s", failures, _MAX_FAILURES + 1, cause)
                    if failures > _MAX_FAILURES:
                        raise StreamError(f"스트림 연결에 실패했습니다. {cause}{_outdated_hint()} 주소와 네트워크를 확인해 주세요.")
                    stop.wait(0.5)
                    continue
                source_fps = capture.get(cv2.CAP_PROP_FPS)
                frame_interval = 1.0 / source_fps if 1.0 <= source_fps <= 60.0 else None
                next_frame_at: Optional[float] = None
                got_frames = False
                while not stop.is_set():
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        break
                    got_frames = True
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
                failures = 1 if got_frames else failures + 1    # a stream that played fine before the break starts the count over
                cause = _diagnose_open_failure(source)
                _log.warning("스트림이 끊겼습니다 (재연결 %d/%d, 프레임 수신 %s): %s", failures, _MAX_FAILURES, "있음" if got_frames else "없음", cause)
                if failures > _MAX_FAILURES:
                    raise StreamError(f"스트림이 중단되어 {_MAX_FAILURES}회 재연결을 시도했지만 실패했습니다. {cause}{_outdated_hint()}")
                stop.wait(0.5)
        except Exception as err:
            _log.error("스트림 처리를 멈춥니다: %s", err)
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
            if should_stop is not None and should_stop():
                break
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
                boxes = detect_faces(detect_frame, track=True, mode="stream")
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
                refs = refs_now()
                if refs is not last_refs:    # 등록이 바뀌었다: 더는 믿을 수 없는 판정만 버린다
                    _drop_stale_verdicts(exclude_cache, last_refs, refs)
                    last_refs = refs
                if refs is not None:
                    if filter_excluded is None:
                        filter_excluded = _load_exclusion_filter()
                    boxes = filter_excluded(frame, boxes, refs, cache=exclude_cache)
                apply_effect = apply_blur if effect_now() == "blur" else apply_mosaic
                result = apply_effect(frame, boxes)
                processed += 1
                frame_idx += 1
                yield_times.append(time.perf_counter())
                elapsed = yield_times[-1] - yield_times[0]
                fps = (len(yield_times) - 1) / elapsed if len(yield_times) > 1 and elapsed > 0 else 0.0
                yield result, {"faces_now": len(boxes), "faces_total": len(track_ids),
                               "fps": fps, "original": frame}
            elif done:
                break
    finally:
        stop.set()
        changed.set()
        worker.join(timeout=2.0)
