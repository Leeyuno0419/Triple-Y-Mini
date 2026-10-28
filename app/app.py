"""얼굴 자동 모자이크 웹 데모 (C 나예린 담당 영역).

실행 방법 (프로젝트 루트에서):
    streamlit run app/app.py

이 파일은 두 부분으로 나뉜다.
  1) 처리 함수  : on_register / on_image / on_video / on_stream
                  (역할별_구현명세.md C장 ⑤ 의 이름·입출력을 그대로 따른다)
  2) 화면 함수  : page_main_demo / page_register_person / render_bottom_bar
                  (버튼·업로드 칸을 만들고, 위 처리 함수를 불러서 결과를 보여 준다.
                   [메인]과 [특정 인물 등록]은 서로 독립된 페이지이고,
                   화면 아래쪽 하단 바 메뉴를 누르면 페이지가 바뀐다)

팀원 모듈은 처리 함수 안에서 "필요할 때" 불러온다. 아직 없는 모듈이어도 앱은 뜨고,
그 기능을 누르면 "모듈 대기 중" 안내만 보인다.

  [A 이유노]  src/detector.py : detect_faces()        얼굴 박스 찾기
  [B 조영준]  src/mosaic.py   : apply_mosaic()        이미지 모자이크
              src/video.py    : process_video()       영상 모자이크
              src/stream.py   : process_stream()      실시간 모자이크
  [C 나예린]  src/exclude.py  : list_candidates() / filter_excluded()  특정 인물 제외
"""
from __future__ import annotations

import sys
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterator, Optional

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

if TYPE_CHECKING:    # 타입 표시용으로만 쓰고, 실행할 때는 불러오지 않는다
    from streamlit.runtime.uploaded_file_manager import UploadedFile

# `streamlit run app/app.py` 로 실행하면 app/ 폴더만 import 경로에 잡힌다.
# `from src.xxx import ...` 가 되도록 프로젝트 루트를 import 경로에 추가한다.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ============================================================
# 1. 기본 설정
# ============================================================

# 하단 바 메뉴 이름. 메뉴를 바꾸고 싶으면 여기만 고치면 된다.
MENU_MAIN = "메인"
MENU_REGISTER = "특정 인물 등록"

# 업로드를 허용하는 파일 확장자 (점 없이 소문자로 쓴다)
IMAGE_TYPES = ["jpg", "jpeg", "png"]
VIDEO_TYPES = ["mp4"]

# [메인]의 처리 종류 버튼: 사진 / 영상 (/ 실시간 주소)
MODE_IMAGE = "사진"
MODE_VIDEO = "영상"
MODE_STREAM = "실시간 주소"

# [실시간 주소] 버튼을 보일지 여부. 실시간 기능(B파트 stream.py)이 준비되면 True 로 바꾼다.
SHOW_STREAM_MODE = False

# 화면에 보여 줄 이미지의 크기 (가로, 세로 픽셀). 모든 사진을 이 크기의 캔버스에 맞춰 보여 준다.
DISPLAY_SIZE = (1280, 720)

# 영상 업로드 제한 (명세 C장 ⑤: 1분·720p). "720p"는 짧은 변이 720 이하로 해석했다.
# TODO: 숫자 설정은 src/config.py 에서만 바꾸는 것이 규칙이다. 팀에 알린 뒤 config.py 로 옮긴다.
MAX_VIDEO_SECONDS = 60
MAX_VIDEO_SHORT_SIDE = 720


# 화면 꾸미기용 CSS. 색을 바꾸고 싶으면 맨 위 --변수 값만 고치면 된다.
#   아이보리 배경(--bg) · 흰 카드(--card) · 민트(--mint) · 하늘색(--sky) · 회색(--muted, --line) · 검정(--ink) 파스텔 조합이다.
# 카드는 st.container(key="card_이름") 으로 묶고, 아래 [class*="st-key-card_"] 규칙으로 꾸민다.
APP_CSS = """
<style>
:root {
    --bg: #FDFCF7;         /* 전체 배경 (밝은 아이보리) */
    --card: #FFFFFF;       /* 카드 배경 (흰색) */
    --ink: #111418;        /* 글자 · 버튼 (검정) */
    --ink-soft: #2E3640;   /* 버튼에 마우스를 올렸을 때 */
    --muted: #6B7280;      /* 보조 글자 (회색) */
    --line: #E5E7EB;       /* 얇은 테두리 (연한 회색) */
    --mint: #CDEFE2;       /* 파스텔 민트 (배너 · 강조) */
    --mint-soft: #EEF9F4;  /* 업로드 칸 */
    --mint-line: #9FD9C2;  /* 업로드 칸 점선 · 버튼 포커스 */
    --sky: #D9ECFB;        /* 파스텔 하늘색 (상단 바 · 장식) */
    --sky-soft: #F1F8FE;   /* 연한 하늘색 (안내 상자 · 마우스 올린 메뉴) */
    --select: linear-gradient(135deg, #A3E6CD 0%, #9ED2F6 100%);   /* 선택된 메뉴 · 전환 버튼: 배경 그라데이션(민트 → 하늘색)을 더 선명하게 */
    --select-ink: #0F3B57;   /* 선택된 메뉴 · 전환 버튼의 글씨 (진한 청록 남색. 흰색보다 파스텔 위에서 또렷하다) */
}

/* 전체 배경: 아이보리 위에 민트(왼쪽 위)와 하늘색(오른쪽)이 은은하게 번진다 */
.stApp {
    background:
        radial-gradient(900px 520px at 6% 0%, rgba(205, 239, 226, 0.7), transparent 70%),
        radial-gradient(900px 560px at 96% 100%, rgba(217, 236, 251, 0.9), transparent 70%),
        radial-gradient(700px 420px at 100% 30%, rgba(234, 244, 253, 0.9), transparent 70%),
        var(--bg);
    background-attachment: fixed;
}
[data-testid="stHeader"],
[data-testid="stDecoration"],
[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"] { display: none; }   /* 위쪽 기본 막대와 사이드바는 쓰지 않는다 */
footer { visibility: hidden; }
[data-testid="stMain"] { scroll-behavior: smooth; }
/* 위쪽 여백은 상단 바 높이, 아래쪽 여백은 하단 바에 내용이 가려지지 않게 넉넉히 둔다 */
.block-container { max-width: 1080px; padding-top: 4.2rem; padding-bottom: 8rem; }

/* 상단 바: 화면 맨 위에 꽉 차게 붙고, 아래쪽 모서리가 둥근 U자 모양이다 (파스텔 그라데이션) */
.stApp::before {
    content: "TRIPLE Y MINI  ·  FACE MOSAIC";
    position: fixed; top: 0; left: 0; right: 0; height: 2.6rem; z-index: 1000;
    display: flex; align-items: center; justify-content: center;
    background: linear-gradient(90deg, var(--mint) 0%, #E3F3F1 50%, var(--sky) 100%);
    border-radius: 0 0 28px 28px;
    box-shadow: 0 6px 22px rgba(60, 110, 160, 0.12);
    font-size: .7rem; font-weight: 700; letter-spacing: .22em; color: var(--muted);
}

/* 각 페이지의 첫 화면(랜딩): 화면 높이를 꽉 채운다. 위쪽 여백을 당겨서 화면 맨 위부터 채운다. */
.landing {
    position: relative; height: 100vh; margin-top: -4.2rem; box-sizing: border-box;
    padding: 5rem 1rem 9rem;
    display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center;
}
.landing .blob { position: absolute; border-radius: 50%; opacity: .75; pointer-events: none; animation: float 7s ease-in-out infinite; }
.landing .blob.b1 { width: 220px; height: 220px; left: -6%; top: 16%; background: var(--mint); }
.landing .blob.b2 { width: 150px; height: 150px; right: -4%; top: 26%; background: var(--sky); animation-delay: -2.5s; }
.landing .blob.b3 { width: 100px; height: 100px; left: 4%; bottom: 24%; background: var(--line); animation-delay: -4.5s; }
.landing .pill { position: relative; }
.landing-title {
    position: relative; margin: 1.3rem 0 1.2rem; color: var(--ink);
    font-size: clamp(2.6rem, 8vw, 5.4rem); font-weight: 800; line-height: 1.14; letter-spacing: -.03em;
}
.landing-title mark { background: linear-gradient(transparent 62%, var(--mint) 62%); color: inherit; padding: 0 .08em; }
.landing.compact .landing-title { font-size: clamp(1.9rem, 4.6vw, 2.8rem); }
.landing-desc { position: relative; max-width: 38rem; margin: 0; word-break: keep-all !important; overflow-wrap: normal !important; text-wrap: balance; color: var(--muted); line-height: 1.7; font-size: clamp(1rem, 2.2vw, 1.2rem); }
.scroll-hint {
    position: absolute; bottom: 6.5rem; left: 0; right: 0;
    display: flex; flex-direction: column; align-items: center; gap: .3rem;
    color: var(--muted); font-size: .95rem; font-weight: 600;
    animation: bounce 1.8s ease-in-out infinite, hint-fade linear both;
    animation-timeline: auto, scroll(nearest);   /* 두 번째 애니메이션(사라짐)은 시간이 아니라 스크롤 양을 따른다 */
    animation-range: normal, 0 120px;            /* 120px 스크롤하는 동안 서서히 사라진다 */
}
@keyframes bounce { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(10px); } }
@keyframes hint-fade { from { opacity: 1; } to { opacity: 0; } }
@keyframes float { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-18px); } }
@media (prefers-reduced-motion: reduce) { .landing .blob, .scroll-hint { animation: none; } }

/* 랜딩 아래 본문의 구역 제목 */
.section-head { text-align: center; padding: 2rem 0 .6rem; }
.section-title { font-size: clamp(1.6rem, 4vw, 2.2rem); font-weight: 800; color: var(--ink); margin: 0; }
.section-sub { color: var(--muted); margin: .4rem 0 0; }

/* 카드: 흰 배경 + 둥근 모서리 + 얇은 테두리 + 부드러운 그림자 */
[class*="st-key-card_"] {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 20px;
    padding: 1.6rem 1.8rem;
    box-shadow: 0 6px 28px rgba(60, 110, 160, 0.08);
    box-sizing: border-box;
}
/* 카드 안쪽 요소가 여백을 빼고 카드 폭에 맞게 들어가도록 한다 (안 하면 오른쪽으로 삐져나온다) */
[class*="st-key-card_"] > div,
[class*="st-key-card_"] [data-testid="stMarkdown"] { width: 100% !important; max-width: 100% !important; }
[class*="st-key-card_"] [data-testid="stMarkdown"] { margin-bottom: 0 !important; }

/* 인물 등록 구역: 박스 테두리 없이, 구역마다 민트 → 하늘색 그라데이션이 가장자리로 번져 사라지게 한다.
   구역 하나가 화면 한 장을 차지하도록 높이를 넉넉히 잡는다. */
[class*="st-key-card_register"] {
    background: radial-gradient(ellipse 50% 50% at 50% 50%, rgba(214, 242, 230, 0.95) 0%, rgba(223, 239, 252, 0.65) 55%, transparent 100%);   /* 상자 가장자리에서 정확히 투명해진다 */
    border: none; border-radius: 0; box-shadow: none;
    min-height: calc(100vh - 9rem); padding: 2.5rem 2rem; justify-content: center;
}
[class*="st-key-card_register"] [data-testid="stFileUploaderDropzone"] { background: rgba(255, 255, 255, 0.6); }
[class*="st-key-card_register"] [data-testid="stAlert"] > div { background: rgba(255, 255, 255, 0.65) !important; }

.pill {
    display: inline-block; background: #fff; color: var(--ink);
    font-size: .75rem; font-weight: 600; padding: .2rem .8rem;
    border-radius: 999px; box-shadow: 0 2px 8px rgba(60, 110, 160, 0.12);
}

/* 카드 안의 작은 제목 */
.card-title { font-size: 1.25rem; font-weight: 700; color: var(--ink); margin: 0; }
.card-desc { font-size: .92rem; color: var(--muted); margin: .2rem 0 1rem; }
.st-key-card_upload .card-title, .st-key-card_upload .card-desc { text-align: center; }   /* 파일 업로드 카드 제목 가운데 정렬 */

/* 버튼: 검정 배경 + 흰 글씨 + 둥근 모서리 (모자이크 적용, 파일 선택 등) */
.stButton > button,
[data-testid="stFileUploaderDropzone"] button {
    background: var(--ink); color: #fff !important; border: none; border-radius: 999px;
    padding: .55rem 1.5rem; font-weight: 600; transition: background .15s;
}
.stButton > button *,
[data-testid="stFileUploaderDropzone"] button * { color: #fff !important; }
.stButton > button:hover,
[data-testid="stFileUploaderDropzone"] button:hover { background: var(--ink-soft); border: none; }
.stButton > button:focus:not(:active) { border: none; box-shadow: 0 0 0 3px rgba(159, 217, 194, 0.55); }

/* 파일 업로드 칸 */
[data-testid="stFileUploaderDropzone"] {
    background: var(--mint-soft); border: 1.5px dashed var(--mint-line); border-radius: 16px;
}
[data-testid="stFileUploaderDropzone"] svg { color: var(--mint-line) !important; }   /* 구름 아이콘 색 */

/* 안내 상자 · 이미지: 안내 상자는 종류(정보·성공 등)와 상관없이 연한 하늘색 한 가지로 통일한다 */
[data-testid="stAlert"] { border-radius: 14px; }
[data-testid="stAlert"] > div { background: var(--sky-soft) !important; }
[data-testid="stAlert"] * { color: var(--ink) !important; }
[data-testid="stImage"] img { border-radius: 12px; }

/* 처리 종류 버튼(사진 / 영상): 가운데에 놓이는 흰 알약 안에서 선택한 버튼만 하늘색이 되는 전환 버튼 */
[class*="st-key-mode_picker"] { margin-bottom: .6rem; }
[class*="st-key-mode_picker"] [data-testid="stRadio"] { text-align: center; }   /* 알약(inline-flex)을 가운데로 */
[class*="st-key-mode_picker"] [data-testid="stWidgetLabel"] { display: none; }
[class*="st-key-mode_picker"] [role="radiogroup"] {
    display: inline-flex; gap: .25rem; padding: .3rem; background: #fff;
    border: 1px solid var(--line); border-radius: 999px;
}
[class*="st-key-mode_picker"] label[data-baseweb="radio"] {
    margin: 0; padding: .5rem 1.8rem; border-radius: 999px; cursor: pointer; white-space: nowrap; transition: background .15s;
}
[class*="st-key-mode_picker"] label[data-baseweb="radio"] > div:first-child { display: none; }   /* 동그라미 숨김 */
[class*="st-key-mode_picker"] label[data-baseweb="radio"] > div:last-child { margin: 0 !important; padding: 0 !important; }   /* 글자 옆 기본 여백(왼쪽 패딩 8px) 제거 */
[class*="st-key-mode_picker"] label[data-baseweb="radio"]:hover { background: var(--sky-soft); }
[class*="st-key-mode_picker"] label[data-baseweb="radio"]:has(input:checked) { background: var(--select); font-weight: 700; }
[class*="st-key-mode_picker"] label[data-baseweb="radio"]:has(input:checked) * { color: var(--select-ink) !important; }

/* 하단 바: 화면 맨 아래에 꽉 차게 붙고, 위쪽 모서리가 둥근 모양이다. 세 칸으로 나눈다.
   [왼쪽: 메인] - [가운데: YYY 로고] - [오른쪽: 특정 인물 등록]
   양쪽 메뉴 칸의 폭을 같게(--nav-w) 하고, 메뉴 알약은 글자 크기에 맞춰 로고 쪽 가장자리에 붙여서
   메인 - 로고 - 등록 사이(column-gap)가 똑같이 벌어지고 로고가 바의 정중앙에 놓인다. */
[class*="st-key-bottom_bar"] {
    --nav-w: 11rem;
    position: fixed; bottom: 0; left: 0; right: 0; width: 100% !important; z-index: 1000; box-sizing: border-box;
    display: grid !important; grid-template-columns: var(--nav-w) auto var(--nav-w); justify-content: center;
    align-items: center; column-gap: 2.4rem;
    padding: .7rem 1rem .8rem;
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(234, 244, 253, 0.98));
    backdrop-filter: blur(8px);
    border-top: 1px solid var(--line); border-radius: 28px 28px 0 0;
    box-shadow: 0 -8px 30px rgba(60, 110, 160, 0.14);
}
[class*="st-key-bottom_bar"]::before {   /* 가운데 칸: 로고 이미지 (app/static/yyy_logo.png) */
    content: ""; grid-column: 2; grid-row: 1; width: 4.6rem; height: 2.1rem;
    background: url("/app/static/yyy_logo.png") no-repeat center / contain;
}
/* 라디오를 감싼 상자들을 풀어서(contents) 두 메뉴 버튼이 바의 칸에 바로 놓이게 한다 */
[class*="st-key-bottom_bar"] > div,
[class*="st-key-bottom_bar"] [data-testid="stRadio"],
[class*="st-key-bottom_bar"] [data-testid="stRadio"] > div,
[class*="st-key-bottom_bar"] [role="radiogroup"] { display: contents !important; }
[class*="st-key-bottom_bar"] [data-testid="stWidgetLabel"] { display: none; }
[class*="st-key-bottom_bar"] label[data-baseweb="radio"] {   /* 알약은 글자 크기에 맞추고, 글자는 알약 한가운데에 놓는다 */
    position: relative; isolation: isolate; color: var(--ink);
    margin: 0; padding: .6rem 1.6rem; border-radius: 999px; cursor: pointer; white-space: nowrap;
    display: flex; justify-content: center; width: fit-content;
}
[class*="st-key-bottom_bar"] label[data-baseweb="radio"] > div:last-child { margin: 0 !important; padding: 0 !important; }   /* 글자 옆 기본 여백(왼쪽 패딩 8px) 제거 */
[class*="st-key-bottom_bar"] label[data-baseweb="radio"] * { color: inherit !important; font-weight: inherit !important; }   /* 글씨색·굵기는 라벨 하나에서 정한다 */
[class*="st-key-bottom_bar"] label[data-baseweb="radio"]::before {   /* 선택 색 알약. 평소에는 투명하고 선택되면 보인다 */
    content: ""; position: absolute; inset: 0; z-index: -1; border-radius: 999px; background: var(--select); opacity: 0;
}
[class*="st-key-bottom_bar"] label[data-baseweb="radio"]:nth-of-type(1) { grid-column: 1; grid-row: 1; justify-self: end; }     /* 왼쪽 칸 (로고 쪽 끝) */
[class*="st-key-bottom_bar"] label[data-baseweb="radio"]:nth-of-type(2) { grid-column: 3; grid-row: 1; justify-self: start; }   /* 오른쪽 칸 (로고 쪽 끝) */
[class*="st-key-bottom_bar"] label[data-baseweb="radio"] > div:first-child { display: none; }   /* 동그라미 숨김 */
[class*="st-key-bottom_bar"] label[data-baseweb="radio"]:hover { background: var(--sky-soft); }
[class*="st-key-bottom_bar"] label[data-baseweb="radio"]:has(input:checked)::before { opacity: 1; }
[class*="st-key-bottom_bar"] label[data-baseweb="radio"]:has(input:checked) { color: var(--select-ink); font-weight: 700; }

/* [메인]에서 스크롤을 내리면 선택 색이 [메인]에서 [특정 인물 등록]으로 옮겨 간다. (페이지는 그대로 [메인])
   스크롤 양을 따라 움직이는 애니메이션이라, 다시 맨 위로 올리면 [메인]으로 되돌아온다.
   [메인]인지는 첫 화면(.landing)에 .compact 가 없는 것으로 알아본다. */
.stApp { timeline-scope: --page-scroll; }
[data-testid="stMain"] { scroll-timeline: --page-scroll block; }
@keyframes pick-on { from { opacity: 0; } to { opacity: 1; } }
@keyframes pick-off { from { opacity: 1; } to { opacity: 0; } }
@keyframes ink-on { from { color: var(--ink); font-weight: 400; } to { color: var(--select-ink); font-weight: 700; } }   /* 선택되는 메뉴의 글씨색 */
@keyframes ink-off { from { color: var(--select-ink); font-weight: 700; } to { color: var(--ink); font-weight: 400; } }
.stApp:has(.landing:not(.compact)) [class*="st-key-bottom_bar"] label[data-baseweb="radio"]:nth-of-type(1)::before {
    animation: pick-off linear both; animation-timeline: --page-scroll; animation-range: 0 120px;
}
.stApp:has(.landing:not(.compact)) [class*="st-key-bottom_bar"] label[data-baseweb="radio"]:nth-of-type(1) {
    animation: ink-off linear both; animation-timeline: --page-scroll; animation-range: 0 120px;
}
.stApp:has(.landing:not(.compact)) [class*="st-key-bottom_bar"] label[data-baseweb="radio"]:nth-of-type(2)::before {
    animation: pick-on linear both; animation-timeline: --page-scroll; animation-range: 0 120px;
}
.stApp:has(.landing:not(.compact)) [class*="st-key-bottom_bar"] label[data-baseweb="radio"]:nth-of-type(2) {
    animation: ink-on linear both; animation-timeline: --page-scroll; animation-range: 0 120px;
}
@media (max-width: 520px) {   /* 좁은 화면(휴대폰)에서는 칸과 간격을 줄인다 */
    [class*="st-key-bottom_bar"] { --nav-w: 8rem; column-gap: .5rem; }
    [class*="st-key-bottom_bar"]::before { width: 3.2rem; height: 1.6rem; }
    [class*="st-key-bottom_bar"] label[data-baseweb="radio"] { padding: .5rem .9rem; font-size: .85rem; }
}
</style>
"""

class BusyError(RuntimeError):
    """다른 사람이 영상·실시간 처리를 하고 있어서 지금은 시작할 수 없을 때 낸다."""


# ============================================================
# 2. 공용 도우미
# ============================================================

@st.cache_resource
def get_processing_lock() -> threading.Lock:
    """영상·실시간 처리를 한 번에 1건만 하도록 막는 자물쇠를 돌려준다.

    추적 상태가 모델 하나에 묶여 있어서, 두 사람이 동시에 처리하면 추적 ID가 섞인다.
    cache_resource 덕분에 접속한 모든 사람이 같은 자물쇠 하나를 함께 쓴다.
    """
    return threading.Lock()


@contextmanager
def processing_slot() -> Iterator[None]:
    """영상·실시간 처리 자리를 하나 잡는다. 이미 누가 쓰는 중이면 BusyError 를 낸다.

    `with processing_slot():` 블록이 끝나면(오류가 나도) 자리를 반드시 반납한다.
    """
    lock = get_processing_lock()
    if not lock.acquire(blocking=False):      # 기다리지 않고 바로 확인
        raise BusyError("다른 영상을 처리 중입니다. 잠시 후 다시 시도해 주세요.")
    try:
        yield
    finally:
        lock.release()


def init_session_state() -> None:
    """화면이 다시 실행되어도 값이 남도록 session_state 에 기본값을 만든다.

    Streamlit 은 버튼을 누르거나 메뉴를 바꿀 때마다 파일 전체를 처음부터 다시 실행한다.
    그래서 계속 기억해야 하는 값은 st.session_state 에 넣어 둔다.
    """
    st.session_state.setdefault("ref_embs", None)         # 등록된 제외 인물 벡터 (없으면 None)
    st.session_state.setdefault("candidates", None)       # 등록 화면의 얼굴 후보 목록
    st.session_state.setdefault("candidates_for", None)   # 그 후보를 찾은 파일 이름


def get_workdir() -> Path:
    """이 접속(세션) 전용 임시 폴더를 돌려준다. 처음 부를 때 한 번만 만든다."""
    if "workdir" not in st.session_state:
        st.session_state["workdir"] = tempfile.mkdtemp(prefix="triple_y_")
    return Path(st.session_state["workdir"])


def save_upload_to_temp(uploaded: UploadedFile, stem: str) -> str:
    """업로드된 파일을 임시 폴더에 저장하고 경로를 돌려준다.

    영상·인물 등록 모듈은 파일 "경로"를 받기 때문에 저장이 필요하다.
    같은 stem 을 쓰면 같은 이름으로 덮어써서 임시 파일이 계속 쌓이지 않는다.
    """
    suffix = Path(uploaded.name).suffix.lower()
    path = get_workdir() / f"{stem}{suffix}"
    path.write_bytes(uploaded.getvalue())
    return str(path)


def decode_image_rgb(uploaded: UploadedFile) -> np.ndarray:
    """업로드된 이미지 파일을 RGB numpy 이미지로 바꾼다. 읽을 수 없으면 ValueError."""
    data = np.frombuffer(uploaded.getvalue(), np.uint8)
    img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("이미지를 읽을 수 없습니다. 파일이 깨졌는지 확인해 주세요.")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def fit_to_canvas(image_rgb: np.ndarray, size: tuple[int, int] = DISPLAY_SIZE) -> np.ndarray:
    """사진을 size(가로, 세로) 캔버스 한가운데에 비율을 유지한 채 맞춰 넣는다. (RGBA 이미지)

    사진은 자르거나 늘리지 않고, 남는 빈 칸은 투명하게 둔다.
    투명이라 라이트·다크 어느 테마에서도 페이지 배경색이 그대로 비쳐 보인다.
    화면 표시용 복사본을 만들 뿐이고, 모자이크 처리에는 원본을 그대로 쓴다.
    """
    canvas_w, canvas_h = size
    h, w = image_rgb.shape[:2]
    scale = min(canvas_w / w, canvas_h / h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR    # 줄일 때와 늘릴 때 방식이 다르다
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=interp)

    canvas = np.zeros((canvas_h, canvas_w, 4), np.uint8)    # 4번째 값이 투명도. 0 = 완전 투명
    x, y = (canvas_w - new_w) // 2, (canvas_h - new_h) // 2
    canvas[y:y + new_h, x:x + new_w, :3] = resized
    canvas[y:y + new_h, x:x + new_w, 3] = 255               # 사진이 있는 곳만 불투명
    return canvas


def show_module_missing(err: ImportError) -> None:
    """아직 만들어지지 않은 팀원 모듈을 불렀을 때 보여 주는 안내."""
    st.warning(f"아직 연결되지 않은 모듈입니다: `{err.name}` (담당 팀원이 만들면 자동으로 동작합니다)")


def inject_css() -> None:
    """위의 APP_CSS 를 화면에 적용한다. 페이지를 그리기 전에 한 번 불러야 한다."""
    st.markdown(APP_CSS, unsafe_allow_html=True)


def render_landing(pill: str, title_html: str, desc: str, compact: bool = False) -> None:
    """첫 화면(랜딩): 화면 높이를 꽉 채우는 제목과 "스크롤하세요" 안내를 보여 준다.

    compact=True 이면 제목 글씨를 작게 한다. (인물 등록 화면용)
    이 아래에 이어지는 내용은 사용자가 스크롤을 내려야 보인다.
    HTML 안에 빈 줄이 있으면 마크다운이 HTML 을 끊어 버리므로 한 덩어리로 이어 쓴다.
    """
    st.markdown(
        f'<div class="landing{" compact" if compact else ""}">'
        '<div class="blob b1"></div><div class="blob b2"></div><div class="blob b3"></div>'
        f'<span class="pill">{pill}</span>'
        f'<div class="landing-title">{title_html}</div>'
        f'<p class="landing-desc">{desc}</p>'
        '<div class="scroll-hint"><span>스크롤하세요</span></div>'
        "</div>",
        unsafe_allow_html=True,
    )


def render_section_title(title: str, desc: str) -> None:
    """랜딩 아래 본문 맨 위의 가운데 정렬 구역 제목을 보여 준다."""
    st.markdown(
        f'<div class="section-head"><div class="section-title">{title}</div>'
        f'<p class="section-sub">{desc}</p></div>',
        unsafe_allow_html=True,
    )


def render_card_heading(title: str, desc: str = "") -> None:
    """카드 안쪽 맨 위의 작은 제목(과 한 줄 설명)을 보여 준다."""
    st.markdown(
        f'<div class="card-title">{title}</div><p class="card-desc">{desc}</p>',
        unsafe_allow_html=True,
    )


def render_ref_status() -> None:
    """제외 인물이 등록돼 있는지 회색 글씨로 안내한다. (등록은 [특정 인물 등록] 메뉴에서 한다)"""
    if st.session_state["ref_embs"] is None:
        st.caption("등록된 제외 인물이 없습니다. 모든 얼굴이 모자이크됩니다.")
    else:
        st.caption("제외 인물이 등록되어 있습니다. 해당 인물은 모자이크하지 않습니다.")


# ============================================================
# 3. 처리 함수 (명세 C장 ⑤) - 화면 코드(st.xxx)를 쓰지 않는다
# ============================================================

def on_register(file: UploadedFile) -> list[dict]:
    """업로드한 사진·영상에서 얼굴 후보 목록을 찾아 돌려준다.

    반환: [{"thumb": 대표 얼굴(BGR), "embs": (k, 512) 벡터, "frames": 등장 횟수, "score": 점수}, ...]
          얼굴이 없으면 [] (화면에 "얼굴을 찾지 못했습니다" 표시)
    후보 중 한 명을 고르는 일은 화면 쪽(page_register_person)에서 한다.
    """
    # ┌─────────────────────────────────────────────┐
    # │ [C파트 모듈] src/exclude.py : list_candidates │
    # └─────────────────────────────────────────────┘
    from src.exclude import list_candidates

    path = save_upload_to_temp(file, "register")
    return list_candidates(path)


def on_image(image_rgb: np.ndarray, ref_embs: Optional[np.ndarray]) -> tuple[np.ndarray, int]:
    """RGB 이미지의 얼굴을 모자이크한다. (결과 RGB 이미지, 모자이크한 얼굴 수)를 돌려준다.

    순서: RGB -> BGR 로 바꾸기 -> 얼굴 찾기 -> 제외 인물 빼기 -> 모자이크 -> BGR -> RGB 로 되돌리기
    """
    # ┌─────────────────────────────────────────────────────────┐
    # │ [A파트 모듈] src/detector.py : detect_faces              │
    # │ [B파트 모듈] src/mosaic.py   : apply_mosaic              │
    # │ [C파트 모듈] src/exclude.py  : filter_excluded           │
    # └─────────────────────────────────────────────────────────┘
    from src.detector import detect_faces
    from src.mosaic import apply_mosaic

    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)   # 입구: 웹은 RGB, 모듈은 BGR
    boxes = detect_faces(img_bgr)
    if ref_embs is not None:                               # 제외 인물이 등록돼 있을 때만
        from src.exclude import filter_excluded            # (등록 시에만 불러와 C파트 없이도 사진은 동작)

        boxes = filter_excluded(img_bgr, boxes, ref_embs)  # 그 사람 박스를 뺀다
    result_bgr = apply_mosaic(img_bgr, boxes)
    return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB), len(boxes)   # 출구: 다시 RGB


def check_video_limits(video_path: str) -> None:
    """영상이 길이·해상도 제한을 넘으면 ValueError 를 낸다."""
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise ValueError("영상을 열 수 없습니다. 파일이 깨졌는지 확인해 주세요.")
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        short_side = min(cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()

    seconds = frames / fps if fps > 0 else 0.0
    if seconds > MAX_VIDEO_SECONDS:
        raise ValueError(f"영상이 너무 깁니다 ({seconds:.0f}초). {MAX_VIDEO_SECONDS}초 이하만 가능합니다.")
    if short_side > MAX_VIDEO_SHORT_SIDE:
        raise ValueError(f"해상도가 너무 높습니다. {MAX_VIDEO_SHORT_SIDE}p 이하만 가능합니다.")


def on_video(
    video_path: str,
    ref_embs: Optional[np.ndarray],
    on_progress: Optional[Callable[[float], None]] = None,
) -> tuple[str, dict]:
    """영상의 얼굴을 모자이크해 (결과 영상 경로, 정보 dict)를 돌려준다.

    on_progress 는 진행률(0.0~1.0)을 받는 함수다. 정보 dict 는 process_video 가 만든
    {"faces": 등장 인원, "frames": 프레임 수, "fps": 처리 속도, "seconds": 걸린 시간}.
    """
    # ┌──────────────────────────────────────────────────────┐
    # │ [B파트 모듈] src/video.py : process_video              │
    # │ (안에서 A의 detect_faces, C의 filter_excluded 를 부른다) │
    # └──────────────────────────────────────────────────────┘
    from src.video import process_video

    check_video_limits(video_path)

    out_path = str(Path(video_path).with_name(Path(video_path).stem + "_out.mp4"))
    with processing_slot():
        info = process_video(video_path, out_path, ref_embs=ref_embs, on_progress=on_progress)
    return out_path, info


def on_stream(url: str, ref_embs: Optional[np.ndarray]) -> Iterator[tuple[np.ndarray, dict]]:
    """실시간 주소(RTSP·유튜브 라이브 등)를 모자이크하며 (RGB 프레임, 정보 dict)를 계속 내보낸다.

    정보 dict: {"faces_now": 현재 인원, "faces_total": 누적 인원, "fps": 처리 속도}
    """
    # ┌────────────────────────────────────────────┐
    # │ [B파트 모듈] src/stream.py : process_stream  │
    # └────────────────────────────────────────────┘
    from src.stream import process_stream

    with processing_slot():
        for frame_bgr, info in process_stream(url, ref_embs=ref_embs):
            yield cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), info


# ============================================================
# 4. 메인 페이지
# ============================================================

def page_main_demo() -> None:
    """[메인] 랜딩 화면 아래에서 사진·영상(·실시간) 처리를 한다. 종류는 전환 버튼으로 고른다."""
    # 첫 화면: 화면을 꽉 채우는 랜딩. 아래로 스크롤하면 업로드·결과 카드가 나온다.
    render_landing(
        "Triple Y · Auto Face Mosaic",
        "얼굴은 가리고,<br><mark>순간</mark>은 남기다",
        "YOLO가 사진과 영상 속 얼굴을 알아서 찾아 모자이크해요. "
        "지우고 싶은 얼굴만 가리고, 남기고 싶은 사람은 그대로 둘 수 있어요.",
    )
    render_section_title(
        "지금 바로 모자이크해 보세요",
        "사진(jpg, png)이나 동영상(mp4)을 올리면 얼굴을 자동으로 찾아 모자이크합니다. "
        "등록한 인물은 제외할 수 있어요.",
    )

    render_file_tab()


def get_modes() -> list[str]:
    """처리 종류 버튼에 보여 줄 목록을 돌려준다. 실시간은 SHOW_STREAM_MODE 가 True 일 때만 넣는다."""
    return [MODE_IMAGE, MODE_VIDEO] + ([MODE_STREAM] if SHOW_STREAM_MODE else [])


def render_file_tab() -> None:
    """처리 종류 전환 버튼 -> 파일 업로드 -> 원본·결과 비교 카드 순서로 보여 준다."""
    # 카드 2 (중앙): 종류 고르기 + 업로드. 종류를 바꾸면 그 종류에 맞는 업로드 칸이 나온다.
    with st.container(key="card_upload"):
        render_card_heading("파일 업로드", "처리할 종류를 고르고 파일을 올려 주세요.")
        with st.container(key="mode_picker"):
            mode = st.radio("처리 종류", get_modes(), key="mode", horizontal=True, label_visibility="collapsed")

        if mode == MODE_STREAM:
            render_ref_status()
            render_stream()
            return

        is_image = mode == MODE_IMAGE
        uploaded = st.file_uploader(
            "사진을 선택하세요" if is_image else "영상을 선택하세요",
            type=IMAGE_TYPES if is_image else VIDEO_TYPES,
            key=f"uploader_{mode}",    # 종류마다 다른 이름표를 써서, 종류를 바꾸면 이전 파일이 남지 않게 한다
            help="이미지: jpg, png" if is_image else "영상: mp4 (1분·720p 이하)",
        )
        render_ref_status()    # 업로드 칸 바로 아래, 아래 안내 문구의 바로 위
        if uploaded is None:
            st.caption("파일을 올리면 아래에 원본과 모자이크 결과가 나란히 표시됩니다.")
    if uploaded is None:
        return

    # 카드 3 (하단): 원본과 결과 비교.
    with st.container(key="card_result"):
        render_card_heading("원본과 결과 비교", "왼쪽이 원본, 오른쪽이 모자이크 결과입니다.")
        if is_image:
            render_image(uploaded)
        else:
            render_video(uploaded)


def render_image(uploaded: UploadedFile) -> None:
    """업로드한 이미지를 보여 주고, 버튼을 누르면 on_image 로 모자이크한다."""
    try:
        image_rgb = decode_image_rgb(uploaded)
    except ValueError as err:
        st.error(str(err))
        return

    # 원본(왼쪽)과 결과(오른쪽)를 나란히 놓는다. 결과 칸은 버튼을 누른 뒤에 채워진다.
    col_orig, col_result = st.columns(2)
    with col_orig:
        st.image(fit_to_canvas(image_rgb), caption="원본", use_container_width=True)

    if st.button("모자이크 적용", key="run_image"):
        try:
            with st.spinner("처리 중..."):
                result_rgb, n_faces = on_image(image_rgb, st.session_state["ref_embs"])
        except ImportError as err:      # 팀원 모듈이 아직 없을 때
            show_module_missing(err)
        else:
            with col_result:
                st.image(
                    fit_to_canvas(result_rgb),
                    caption=f"모자이크 결과 (얼굴 {n_faces}개)",
                    use_container_width=True,
                )


def render_video(uploaded: UploadedFile) -> None:
    """업로드한 영상을 재생해 보여 주고, 버튼을 누르면 on_video 로 모자이크한다."""
    # 원본(왼쪽)과 결과(오른쪽)를 나란히 놓는다. 결과 칸은 버튼을 누른 뒤에 채워진다.
    col_orig, col_result = st.columns(2)
    with col_orig:
        st.video(uploaded)    # 재생 버튼이 달린 플레이어
        st.caption("원본")

    if not st.button("모자이크 적용", key="run_video"):
        return

    in_path = save_upload_to_temp(uploaded, "input")
    bar = st.progress(0.0, text="처리 중...")
    try:
        out_path, info = on_video(in_path, st.session_state["ref_embs"], on_progress=bar.progress)
    except ImportError as err:
        show_module_missing(err)
    except (ValueError, BusyError) as err:    # 길이·해상도 초과, 다른 처리 중
        st.error(str(err))
    else:
        with col_result:
            st.video(out_path)
            st.caption("모자이크 결과")
        st.caption(
            f"등장 인원 {info['faces']}명 · 프레임 {info['frames']}개 · "
            f"{info['fps']:.1f} FPS · {info['seconds']:.1f}초"
        )
    finally:
        bar.empty()    # 성공·실패와 상관없이 진행률 막대를 지운다


def render_stream() -> None:
    """실시간 주소를 입력받아 on_stream 의 프레임을 차례로 화면에 보여 준다."""
    url = st.text_input("실시간 주소", placeholder="rtsp://... 또는 유튜브 라이브 주소")
    st.caption("멈추려면 다른 버튼을 누르거나 화면을 새로고침하세요.")

    if not (st.button("실시간 시작", key="run_stream") and url):
        return

    frame_box = st.empty()    # 같은 자리에 프레임을 계속 덮어 그린다
    info_box = st.empty()
    frames = on_stream(url, st.session_state["ref_embs"])
    try:
        for frame_rgb, info in frames:
            frame_box.image(frame_rgb, use_container_width=True)
            info_box.caption(
                f"현재 얼굴 {info['faces_now']}개 · 누적 {info['faces_total']}명 · {info['fps']:.1f} FPS"
            )
    except ImportError as err:
        show_module_missing(err)
    except BusyError as err:
        st.error(str(err))
    finally:
        frames.close()    # 화면이 멈출 때 자물쇠가 확실히 풀리게 한다


# ============================================================
# 5. 특정 인물 등록 페이지
# ============================================================

def page_register_person() -> None:
    """[특정 인물 등록] 모자이크에서 제외할 사람의 사진·영상을 올려 얼굴 후보를 고른다.

    (테스트에는 출처를 기록한 공개된 유명인 사진·영상만 쓴다. 일반인 사진은 쓰지 않는다.)
    """
    # 구역 3개를 위에서 아래로 이어 그린다: 소개 화면 -> 업로드 -> (있으면) 얼굴 후보
    render_landing(
        "Exclude Person",
        "특정 <mark>인물</mark> 등록",
        "모자이크하지 않을 사람의 정면 사진(또는 짧은 영상)을 올려 주세요. "
        "등록한 사람은 모자이크에서 제외됩니다.",
        compact=True,
    )

    with st.container(key="card_register_upload"):
        render_card_heading("인물 등록", "사진이나 영상을 올리고 얼굴 후보를 찾아 보세요.")
        uploaded = st.file_uploader(
            "등록할 사진 또는 영상",
            type=IMAGE_TYPES + VIDEO_TYPES,
            key="register_uploader",    # [메인]의 file_uploader 와 구분하는 이름표
        )

        if uploaded is None:
            st.caption("파일을 올리면 미리보기가 표시됩니다.")
        else:
            if uploaded.type.startswith("video/"):
                st.video(uploaded)
            else:
                st.image(uploaded, caption="등록할 사진", width=300)

            # 다른 파일을 올렸다면 이전 파일의 후보는 지운다.
            if st.session_state["candidates_for"] != uploaded.name:
                st.session_state["candidates"] = None

            if st.button("얼굴 후보 찾기", key="find_candidates"):
                try:
                    with st.spinner("얼굴을 찾는 중..."):
                        st.session_state["candidates"] = on_register(uploaded)
                        st.session_state["candidates_for"] = uploaded.name
                except ImportError as err:
                    show_module_missing(err)

    # 얼굴 후보 고르기와 등록 상태. 보여 줄 내용이 있을 때만 구역을 만든다.
    if st.session_state["candidates"] is not None or st.session_state["ref_embs"] is not None:
        with st.container(key="card_register_candidates"):
            render_candidates()

            # 등록 해제 버튼: 등록된 인물이 있을 때만 보여 준다.
            if st.session_state["ref_embs"] is not None:
                st.success("현재 등록된 제외 인물이 있습니다.")
                if st.button("등록 해제", key="clear_ref"):
                    st.session_state["ref_embs"] = None
                    st.rerun()    # 화면을 다시 그려 상태를 바로 반영


def render_candidates() -> None:
    """찾은 얼굴 후보를 썸네일로 보여 주고, 고른 사람의 벡터를 session_state 에 저장한다."""
    candidates = st.session_state["candidates"]
    if candidates is None:
        return                                  # 아직 찾기 전
    if not candidates:                          # 빈 리스트 = 얼굴을 못 찾음
        st.error("얼굴을 찾지 못했습니다. 정면 얼굴이 잘 보이는 파일로 다시 시도해 주세요.")
        return

    render_card_heading("얼굴 후보", "후보 1번이 가장 크게·오래 나온 사람입니다.")
    cols = st.columns(4)
    for i, cand in enumerate(candidates):
        with cols[i % 4]:
            thumb_rgb = cv2.cvtColor(cand["thumb"], cv2.COLOR_BGR2RGB)
            st.image(thumb_rgb, caption=f"후보 {i + 1} (등장 {cand['frames']}회)")
            if st.button("이 사람으로 등록", key=f"pick_{i}"):
                st.session_state["ref_embs"] = cand["embs"]   # [메인]에서 꺼내 쓴다
                st.success(f"후보 {i + 1}번을 제외 인물로 등록했습니다.")


# ============================================================
# 6. 시작점: 하단 바 메뉴로 페이지 나누기
# ============================================================

def scroll_to_top_if_page_changed(menu: str) -> None:
    """페이지가 바뀐 직후에만, 새 페이지가 맨 위부터 보이도록 스크롤을 올린다.

    Streamlit 은 내용이 바뀌어도 스크롤 위치를 그대로 두기 때문에, 이전 페이지에서 내려가 있던 만큼
    새 페이지가 중간부터 보이게 된다. 눈에 안 보이는 작은 스크립트로 한 번 올려 준다.
    """
    previous = st.session_state.get("last_menu")
    st.session_state["last_menu"] = menu
    if previous is None or previous == menu:    # 첫 접속이거나 같은 페이지에서 다시 실행된 경우
        return
    # 새 내용이 그려지는 시간을 고려해 여러 번 올린다. (스크롤 요소는 위쪽 문서(parent)에 있다)
    components.html(
        "<script>"
        "const el = window.parent.document.querySelector('[data-testid=\"stMain\"]');"
        "[0, 150, 500].forEach((ms) => setTimeout(() => el && el.scrollTo({ top: 0, behavior: 'instant' }), ms));"
        "</script>",
        height=0,
    )


def render_bottom_bar() -> None:
    """화면 아래쪽에 떠 있는 메뉴 바를 그린다. 누르면 session_state["menu"] 가 바뀌고 페이지가 전환된다."""
    with st.container(key="bottom_bar"):
        st.radio(
            "메뉴",
            [MENU_MAIN, MENU_REGISTER],
            key="menu",
            horizontal=True,
            label_visibility="collapsed",
        )


def main() -> None:
    """페이지 설정 -> 선택한 페이지 그리기 -> 하단 바 그리기 순서로 앱을 구성한다."""
    # set_page_config 는 다른 st 명령보다 먼저 한 번만 불러야 한다.
    st.set_page_config(
        page_title="Triple Y Mini",
        page_icon=str(ROOT / "app" / "static" / "yyy_favicon.png"),   # 브라우저 탭 아이콘
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    init_session_state()
    inject_css()

    # 하단 바는 화면 맨 아래에 고정되어 있어서 코드에서는 페이지 뒤에 그린다.
    # 바를 누를 때마다 스크립트가 다시 실행되므로, 그때 바뀐 값을 여기서 읽어 페이지를 고른다.
    menu = st.session_state.get("menu", MENU_MAIN)
    if menu == MENU_MAIN:
        page_main_demo()
    else:
        page_register_person()
    render_bottom_bar()
    scroll_to_top_if_page_changed(menu)


if __name__ == "__main__":
    main()
