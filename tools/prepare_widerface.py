"""WIDER FACE train·val 이미지와 정답을 받아 YOLO 라벨로 바꾸고, 본인이 맡은 범위의 검수 이미지를 만들어 폴더를 연다.

사용법 (프로젝트 폴더에서): python tools\\prepare_widerface.py <이니셜>
  이니셜: lyn(이유노, 폴더 0~20), cyj(조영준, 21~40), nyr(나예린, 41~61)
결과: data\\processed\\widerface\\review\\<이니셜>\\train, val 에 박스를 그린 이미지 각 30장 (끝나면 폴더가 열림)

- 받은 zip과 압축 푼 원본은 data\\raw\\widerface\\ 에 둔다 (git 제외). 이미 있으면 다시 받지 않는다.
- 브라우저로 받은 zip이 있으면 data\\raw\\widerface\\ 에 넣고 실행하면 다운로드를 건너뛴다.
- 압축 풀기가 중간에 끊겼다면 data\\raw\\widerface\\ 안의 풀다 만 폴더(WIDER_train 등)를 지우고 다시 실행한다.
"""
from __future__ import annotations

import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset import convert_widerface_to_yolo  # noqa: E402
from review_labels import make_review  # noqa: E402

RAW = ROOT / "data" / "raw" / "widerface"
OUT = ROOT / "data" / "processed" / "widerface"
# Drive 대용량 파일은 일반 공유 링크로 받으면 바이러스 검사 안내 페이지(HTML)가 오므로 다운로드 전용 주소를 쓴다.
# zip 이름(확장자 제외)이 곧 압축을 푼 폴더 이름이다.
DOWNLOADS = {
    "WIDER_train.zip": "https://drive.usercontent.google.com/download?id=15hGDLhsx8bLgLcIRD5DhYt5iBxnjNF1M&export=download&confirm=t",
    "WIDER_val.zip": "https://drive.usercontent.google.com/download?id=1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q&export=download&confirm=t",
    "wider_face_split.zip": "http://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip",
}
SPLITS = ("train", "val")
RANGES = {"lyn": (0, 20), "cyj": (21, 40), "nyr": (41, 61)}


def download(url: str, dest: Path) -> None:
    """url을 dest로 받는다. .part로 받고 크기가 Content-Length와 같을 때만 이름을 바꿔, 반쪽 파일이 남지 않게 한다."""
    if dest.exists():
        print(f"이미 있음: {dest.name}")
        return
    print(f"다운로드 중: {dest.name}")
    part = dest.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=60) as res:
        if "text/html" in res.headers.get("Content-Type", ""):
            sys.exit(f"{dest.name}: 파일 대신 웹 페이지가 왔습니다 (Drive 다운로드 한도 등). "
                     f"README의 주소에서 브라우저로 받아 {RAW}에 넣고 다시 실행하세요.")
        size = int(res.headers.get("Content-Length", 0))
        with open(part, "wb") as fp:
            shutil.copyfileobj(res, fp, length=1024 * 1024)
    # 연결이 중간에 닫혀도 urllib은 예외 없이 끝나므로 크기로 확인한다
    if size and part.stat().st_size != size:
        part.unlink()
        sys.exit(f"{dest.name}: 다운로드가 중간에 끊겼습니다. 다시 실행하세요.")
    part.rename(dest)


def unzip(zip_path: Path, extracted: Path) -> None:
    """zip_path를 RAW에 푼다. 풀린 폴더 extracted가 이미 있으면 건너뛴다."""
    if extracted.exists():
        print(f"이미 풀려 있음: {extracted.name}")
        return
    print(f"압축 푸는 중: {zip_path.name}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(RAW)


def main() -> None:
    """다운로드 → 압축 풀기 → train·val YOLO 변환 → 본인 범위 검수 이미지 생성 → 검수 폴더 열기를 차례로 실행한다."""
    if len(sys.argv) != 2 or sys.argv[1] not in RANGES:
        sys.exit(f"사용법: python tools\\prepare_widerface.py <이니셜>  (이니셜: {', '.join(RANGES)})")
    initial = sys.argv[1]

    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in DOWNLOADS.items():
        download(url, RAW / name)
        unzip(RAW / name, RAW / Path(name).stem)

    review_dir = OUT / "review" / initial
    for split in SPLITS:
        img_dir = RAW / f"WIDER_{split}" / "images"
        stats = convert_widerface_to_yolo(str(RAW / "wider_face_split" / f"wider_face_{split}_bbx_gt.txt"),
                                          str(img_dir), str(OUT / split))
        print(f"{split} YOLO 라벨 변환:", stats)
        make_review(OUT / split / "labels", img_dir, review_dir / split, *RANGES[initial])

    print(f"검수 폴더: {review_dir}")
    if sys.platform == "win32":
        os.startfile(review_dir)


if __name__ == "__main__":
    main()
