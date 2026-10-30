"""인물 제외 오인식 점검 (C 나예린): '서로 다른 사람' 얼굴 쌍의 유사도를 재서 현재 설정에 오인식이 없는지 확인한다.

같은 사진·같은 영상 프레임에 함께 찍힌 서로 다른 얼굴은 확실히 다른 사람이다. 그런 쌍의 유사도가 SIM_THRESHOLD 이상이면,
그 사람이 등록 인물로 잘못 판정되어 모자이크 없이 노출될 수 있다(오인식). 이 도구는 그런 쌍을 세어 보고
"MIN_VERIFY_SIZE 이상인 얼굴끼리의 오인식이 0건인지" 를 PASS / FAIL 로 알려 준다.

사용법 (프로젝트 루트에서):
  python tools\\eval_exclude.py data\\raw\\test\\video.mp4 data\\raw\\test\\newjeans.jpg
  python tools\\eval_exclude.py data\\raw\\test --max-frames 60      # 폴더면 안의 사진·영상을 모두 쓴다

원칙 (기획서 8·11장, 역할별_구현명세.md C장, CLAUDE.md 데이터·AI 규칙)
  - 오인식 0건이 최우선이다. 애매하면 언제나 모자이크한다.
  - 테스트 데이터는 인터넷에 공개된 유명인 사진·영상만 쓴다. 출처(URL)를 기록하고, 개인 블로그·SNS의 일반인 사진은 쓰지 않는다.
    데이터(data/raw)는 git 에 올리지 않는다.
  - 7일 차 컷 판단(전원): 이 도구가 FAIL 인 채로 남으면 인물 제외 기능을 접고(범위 축소) C 는 웹·영상 지원으로 전환한다.
  - 한계: 거울·포스터처럼 같은 사람이 한 화면에 두 번 나오면 '다른 사람' 쌍이 높게 나올 수 있다. 높은 쌍은 화면을 직접 확인한다.
"""
from __future__ import annotations

try:    # Windows 에서는 torch 보다 onnx 를 먼저 불러야 한다 (InsightFace 가 쓰는 onnx 의 DLL 오류 방지)
    import onnx  # noqa: F401
except ImportError:
    pass

import argparse
import sys
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import MIN_VERIFY_SIZE, SIM_THRESHOLD  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
SIZE_BANDS = ((0, 20), (20, 24), (24, 30), (30, 40), (40, 60), (60, 10**6))    # 두 얼굴 중 작은 쪽의 짧은 변(px)
FLOORS = (13, 16, 20, 24, 28, 32, 40)                                           # 판정 대상으로 삼는 크기 하한 후보
MARGIN = 0.05                                                                   # 가장 높은 타인 유사도와 임계값 사이에 최소로 두고 싶은 여유


def iter_frames(path: Path, every_seconds: float, max_frames: int) -> Iterator[np.ndarray]:
    """사진이면 한 장을, 영상이면 every_seconds 초마다 한 프레임(최대 max_frames 장)을 BGR 로 내보낸다. 한글 경로도 읽는다."""
    if path.suffix.lower() in IMAGE_EXTS:
        img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            yield img
        return
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(fps * every_seconds)))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for count, index in enumerate(range(0, max(total, 1), step)):
            if count >= max_frames:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def collect_sources(paths: list[str]) -> list[Path]:
    """인자로 받은 파일·폴더에서 사진·영상 파일 목록을 만든다."""
    found: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            found += sorted(q for q in p.rglob("*") if q.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS)
        elif p.exists():
            found.append(p)
        else:
            print(f"(없는 경로라 건너뜀: {raw})")
    return found


def impostor_pairs(sources: list[Path], every_seconds: float, max_frames: int) -> tuple[list[tuple[float, int]], int, int]:
    """같은 프레임 속 서로 다른 얼굴 쌍의 (유사도, 두 얼굴 중 작은 쪽의 짧은 변)을 모은다. (쌍 목록, 본 프레임 수, 얼굴 수)"""
    from src.exclude import _face_box, _face_embedding, get_face_app    # 얼굴 인식 엔진은 필요할 때 불러온다

    app = get_face_app()
    pairs: list[tuple[float, int]] = []
    frames = faces = 0
    for source in sources:
        for frame in iter_frames(source, every_seconds, max_frames):
            h, w = frame.shape[:2]
            items = []
            for face in app.get(frame):
                emb = _face_embedding(face)
                if emb is None:
                    continue
                x1, y1, x2, y2 = _face_box(face, w, h)
                items.append((min(x2 - x1, y2 - y1), emb))
            frames += 1
            faces += len(items)
            for a in range(len(items)):
                for b in range(a + 1, len(items)):
                    pairs.append((float(items[a][1] @ items[b][1]), min(items[a][0], items[b][0])))
    return pairs, frames, faces


def main() -> int:
    parser = argparse.ArgumentParser(description="인물 제외 오인식 점검: 서로 다른 사람 얼굴 쌍의 유사도를 잰다.")
    parser.add_argument("paths", nargs="+", help="사진·영상 파일이나 그것들이 들어 있는 폴더")
    parser.add_argument("--every", type=float, default=1.0, help="영상에서 몇 초마다 한 프레임을 볼지 (기본 1초)")
    parser.add_argument("--max-frames", type=int, default=40, help="영상 하나에서 볼 프레임 수의 상한 (기본 40)")
    args = parser.parse_args()

    sources = collect_sources(args.paths)
    if not sources:
        print("사진·영상을 찾지 못했습니다.")
        return 2
    try:
        pairs, frames, faces = impostor_pairs(sources, args.every, args.max_frames)
    except ImportError as err:
        print(f"얼굴 인식 패키지를 불러오지 못했습니다: {err.name}. `pip install -r requirements.txt` 를 먼저 실행해 주세요.")
        return 2
    if not pairs:
        print(f"프레임 {frames}장, 얼굴 {faces}개: 한 화면에 얼굴이 둘 이상 나오는 장면이 없어 점검할 수 없습니다.")
        return 2

    arr = np.array(pairs)
    print(f"\n파일 {len(sources)}개 · 프레임 {frames}장 · 얼굴 {faces}개 · 서로 다른 사람 쌍 {len(arr)}개")
    print(f"현재 설정: SIM_THRESHOLD = {SIM_THRESHOLD}, MIN_VERIFY_SIZE = {MIN_VERIFY_SIZE}px\n")

    print(f"{'두 얼굴 중 작은 쪽 크기':<22}{'쌍 수':>6}{'>=0.4':>8}{'>=0.5':>8}{'최대':>8}")
    for lo, hi in SIZE_BANDS:
        mask = (arr[:, 1] >= lo) & (arr[:, 1] < hi)
        if mask.any():
            label = f"{lo}~{hi}px" if hi < 10**6 else f"{lo}px 이상"
            print(f"{label:<22}{int(mask.sum()):>6}{int((arr[mask, 0] >= 0.4).sum()):>8}{int((arr[mask, 0] >= 0.5).sum()):>8}{arr[mask, 0].max():>8.3f}")

    print("\n크기 하한별 (그 하한 이상인 얼굴끼리만 판정한다고 할 때):")
    safe_floors = []
    for floor in FLOORS:
        mask = arr[:, 1] >= floor
        top = float(arr[mask, 0].max()) if mask.any() else 0.0
        false_n = int((arr[mask, 0] >= SIM_THRESHOLD).sum())
        print(f"  하한 {floor:>2}px: 대상 쌍 {int(mask.sum()):>4}개 중 임계값 이상 {false_n}개, 가장 높은 타인 유사도 {top:.3f} (여유 {SIM_THRESHOLD - top:+.3f})")
        if false_n == 0 and SIM_THRESHOLD - top >= MARGIN:
            safe_floors.append(floor)
    if safe_floors:
        print(f"  -> 오인식 0건이면서 여유 {MARGIN} 이상인 가장 작은 하한: {safe_floors[0]}px")

    judged = arr[arr[:, 1] >= MIN_VERIFY_SIZE]
    false_matches = int((judged[:, 0] >= SIM_THRESHOLD).sum()) if len(judged) else 0
    top = float(judged[:, 0].max()) if len(judged) else 0.0
    print(f"\n현재 설정으로 판정되는 쌍 {len(judged)}개 중 오인식(타인이 임계값 이상) = {false_matches}건, 가장 높은 타인 유사도 {top:.3f}, 여유 {SIM_THRESHOLD - top:+.3f}")
    if false_matches == 0:
        print("결과: PASS (오인식 0건)")
        if len(judged) < 100:
            print("  주의: 판정된 쌍이 100개 미만이라 근거가 약합니다. 유명인 영상·사진을 더 모아 다시 재 보세요.")
        return 0
    print("결과: FAIL (오인식이 남아 있음)")
    print("  -> SIM_THRESHOLD 를 올리거나 MIN_VERIFY_SIZE 를 키운 뒤 다시 재 보세요. 그래도 남으면 7일 차 컷 판단에서")
    print("     인물 제외 기능을 접고(범위 축소) C 는 웹·영상 지원으로 전환합니다 (기획서 10·11장).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
