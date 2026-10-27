"""데이터셋 라벨 변환 (A 이유노). 입출력 규칙은 역할별_구현명세.md A장을 따른다.

사용법 (프로젝트 폴더에서):
  python src\\dataset.py <wider_face_train_bbx_gt.txt> <WIDER_train\\images> <출력 폴더>
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image


def convert_widerface_to_yolo(gt_txt: str, img_dir: str, out_dir: str) -> dict:
    """WIDER FACE 정답 txt를 YOLO txt 라벨로 바꾼다.

    out_dir/labels/<이미지이름>.txt에 얼굴 하나당 `0 cx cy w h`(0~1 비율, 소수점 6자리) 한 줄을 쓴다.
    invalid 플래그 박스와, 이미지 범위로 자른 뒤 너비·높이가 0 이하인 박스는 건너뛰고 skipped로 센다.
    얼굴이 0개인 이미지는 빈 txt를 만든다.
    반환: {"images": 변환한 이미지 수, "boxes": 박스 수, "skipped": 건너뛴 수}
    """
    label_dir = Path(out_dir) / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    lines = Path(gt_txt).read_text(encoding="utf-8").splitlines()
    stats = {"images": 0, "boxes": 0, "skipped": 0}

    i = 0
    while i < len(lines) and lines[i].strip():
        rel_path, n = lines[i].strip(), int(lines[i + 1])
        rows = lines[i + 2:i + 2 + n]
        i += 2 + max(n, 1)  # 얼굴이 0개여도 "0 0 0 0 ..." 더미 줄이 한 줄 있다

        with Image.open(Path(img_dir) / rel_path) as im:  # 헤더만 읽어 크기를 얻음
            img_w, img_h = im.size

        out = []
        for row in rows:
            # 한 줄: x1 y1 w h blur expression illumination invalid occlusion pose
            v = [int(t) for t in row.split()]
            x1, y1 = max(v[0], 0), max(v[1], 0)
            x2, y2 = min(v[0] + v[2], img_w), min(v[1] + v[3], img_h)
            if v[7] == 1 or x2 <= x1 or y2 <= y1:
                stats["skipped"] += 1
                continue
            cx, cy = (x1 + x2) / 2 / img_w, (y1 + y2) / 2 / img_h
            out.append(f"0 {cx:.6f} {cy:.6f} {(x2 - x1) / img_w:.6f} {(y2 - y1) / img_h:.6f}\n")

        (label_dir / f"{Path(rel_path).stem}.txt").write_text("".join(out), encoding="utf-8")
        stats["images"] += 1
        stats["boxes"] += len(out)
    return stats


if __name__ == "__main__":
    print(convert_widerface_to_yolo(*sys.argv[1:4]))
