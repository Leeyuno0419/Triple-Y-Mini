"""YOLO 라벨을 이미지 위에 그려 눈으로 검수할 이미지를 만든다.

사용법 (프로젝트 폴더에서):
  python tools\\review_labels.py <labels 폴더> <이미지 폴더> <출력 폴더> [폴더 번호 범위] [장수]
  예: python tools\\review_labels.py data\\processed\\widerface\\train\\labels data\\raw\\widerface\\WIDER_train\\images data\\processed\\widerface\\review\\lyn\\train 0-20 30
폴더 번호 범위는 WIDER FACE 이벤트 폴더 앞 번호(0--Parade의 0)로 거르며, 기본값은 전체(0-61), 장수 기본값은 30.
seed를 고정해서 같은 범위·장수면 누가 돌려도 같은 이미지가 뽑힌다.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SEED = 42


def draw_labels(img_path: Path, label_path: Path) -> Image.Image:
    """라벨 txt의 박스를 픽셀 좌표로 되돌려 이미지 위에 초록 사각형으로 그리고, 왼쪽 위에 얼굴 수를 적는다."""
    im = Image.open(img_path).convert("RGB")
    w, h = im.size
    draw = ImageDraw.Draw(im)
    line_w = max(1, round(max(w, h) / 500))
    rows = label_path.read_text(encoding="utf-8").splitlines()
    for row in rows:
        _, cx, cy, bw, bh = (float(t) for t in row.split())
        draw.rectangle([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h],
                       outline=(0, 255, 0), width=line_w)
    font = ImageFont.load_default(size=max(16, w // 40))
    text = f"{len(rows)} faces"
    draw.rectangle(draw.textbbox((5, 5), text, font=font), fill=(0, 0, 0))
    draw.text((5, 5), text, fill=(255, 255, 0), font=font)
    return im


def make_review(labels_dir: Path, img_dir: Path, out_dir: Path, lo: int = 0, hi: int = 61, count: int = 30) -> None:
    """폴더 번호 lo~hi 범위에서 이미지를 count장 뽑아, 박스를 그린 검수용 이미지를 out_dir에 저장한다."""
    images = {p.stem: p for p in img_dir.rglob("*.jpg")
              if lo <= int(p.parent.name.split("--")[0]) <= hi}
    labeled = sorted(s for s in images if (labels_dir / f"{s}.txt").exists())
    if not labeled:
        sys.exit(f"라벨이 있는 이미지가 없습니다. 변환을 먼저 실행했는지, 폴더 번호 범위({lo}-{hi})가 맞는지 확인하세요.")

    out_dir.mkdir(parents=True, exist_ok=True)
    for stem in random.Random(SEED).sample(labeled, min(count, len(labeled))):
        draw_labels(images[stem], labels_dir / f"{stem}.txt").save(out_dir / f"{stem}.jpg", quality=90)
    print(f"{min(count, len(labeled))}장 저장: {out_dir}")


if __name__ == "__main__":
    lo, hi = (int(t) for t in (sys.argv[4] if len(sys.argv) > 4 else "0-61").split("-"))
    make_review(*(Path(a) for a in sys.argv[1:4]), lo, hi, int(sys.argv[5]) if len(sys.argv) > 5 else 30)
