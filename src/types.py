"""공통 데이터 타입 (역할별_구현명세.md 0장).

`from src.types import Box`처럼 패키지 경로로 불러온다.
`python src\\xxx.py`로 직접 실행하면 표준 라이브러리 `types`와 이름이 겹칠 수 있다.
"""
from __future__ import annotations

from typing import Optional, Tuple, TypedDict


class Box(TypedDict):
    """얼굴 박스 하나. 좌표는 픽셀 정수이고 이미지 범위 안, x1 < x2, y1 < y2."""

    xyxy: Tuple[int, int, int, int]
    conf: float
    track_id: Optional[int]
