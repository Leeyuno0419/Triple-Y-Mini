"""얼굴 자동 모자이크 웹 데모 (C 나예린 담당 영역).

실행 방법 (프로젝트 루트에서):
    streamlit run app/app.py

이 파일은 두 부분으로 나뉜다.
  1) 처리 함수  : on_register / on_image / on_video / on_stream
                  (역할별_구현명세.md C장 ⑤ 의 이름·입출력을 그대로 따른다)
  2) 화면 함수  : page_main_demo / render_register_screen / render_bottom_bar
                  (버튼·업로드 칸을 만들고, 위 처리 함수를 불러서 결과를 보여 준다.
                   화면은 세로로 이어진 한 페이지다: 메인 첫 화면 -> 인물 등록 안내 -> 업로드·결과·성능 지표.
                   스크롤을 한 번 내릴 때마다 다음 화면이 나오고, 하단 바 메뉴는 해당 화면으로 스크롤해 준다)

팀원 모듈은 처리 함수 안에서 "필요할 때" 불러온다. 아직 없는 모듈이어도 앱은 뜨고,
그 기능을 누르면 "모듈 대기 중" 안내만 보인다.

  [A 이유노]  src/detector.py : detect_faces()        얼굴 박스 찾기
  [B 조영준]  src/mosaic.py   : apply_mosaic()        이미지 모자이크
              src/video.py    : process_video()       영상 모자이크
              src/stream.py   : process_stream()      실시간 모자이크
  [C 나예린]  src/exclude.py  : list_candidates() / filter_excluded()  특정 인물 제외
"""
from __future__ import annotations

import base64
import html
import importlib.util
import inspect
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterator, Optional

try:    # Windows: InsightFace 가 쓰는 onnx 는 torch(YOLO)보다 먼저 불러야 한다. 거꾸로면 'DLL load failed' 로 얼굴 후보 찾기가 안 된다
    import onnx  # noqa: F401
except ImportError:    # 설치돼 있지 않으면 인물 등록을 쓸 때 안내가 나온다
    pass

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

from src.config import MAX_VIDEO_SECONDS, MAX_VIDEO_SHORT_SIDE, PAD    # 영상 제한(1분·720p)과 블러 여백은 공통 설정 파일에서 읽는다

# ============================================================
# 1. 기본 설정
# ============================================================

# 하단 바 메뉴 이름. 메뉴를 바꾸고 싶으면 여기만 고치면 된다.
MENU_MAIN = "메인"
MENU_REGISTER = "인물 등록"

# 업로드를 허용하는 파일 확장자 (점 없이 소문자로 쓴다)
IMAGE_TYPES = ["jpg", "jpeg", "png"]
VIDEO_TYPES = ["mp4"]

# [메인]의 처리 종류 버튼: 사진 / 영상 (/ 실시간 주소)
MODE_IMAGE = "사진"
MODE_VIDEO = "영상"
MODE_STREAM = "실시간 주소"

def module_exists(name: str) -> bool:
    """팀원 모듈(src.stream, src.video 등)이 있는지 본다. 이미 불러온 모듈(시험용 가짜 모듈 포함)도 있는 것으로 친다."""
    if name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


# [실시간 주소] 버튼을 보일지 여부. 실시간 기능(B파트 src/stream.py)이 저장소에 올라오면 자동으로 켜진다.
# (직접 켜거나 끄고 싶으면 아래 줄을 True / False 로 바꾼다. 파일은 있는데 불러오다 실패하면 on_stream 이 안내를 보여 준다)
SHOW_STREAM_MODE = module_exists("src.stream")

# 화면에 보여 줄 이미지의 크기 (가로, 세로 픽셀). 모든 사진을 이 크기의 캔버스에 맞춰 보여 준다.
DISPLAY_SIZE = (1280, 720)

# 영상 업로드 제한(명세 C장 ⑤: 1분·720p)은 src/config.py 의 MAX_VIDEO_SECONDS · MAX_VIDEO_SHORT_SIDE 를 쓴다.
# "720p"는 짧은 변이 720 이하로 해석했다. check_video_limits 가 이 값을 넘는 영상을 처리 전에 거절한다.

# 사진 여러 장을 한 번에 올릴 수 있는 최대 개수 (화면을 다시 그릴 때마다 모두 읽기 때문에 상한을 둔다)
# TODO: 위와 같이 팀에 알린 뒤 config.py 로 옮긴다.
MAX_IMAGE_FILES = 20

# 가리는 방식 (사진만. 영상은 B파트 video.py 가 모자이크만 지원한다)
EFFECT_MOSAIC = "모자이크"
EFFECT_BLUR = "블러"
EFFECTS = [EFFECT_MOSAIC, EFFECT_BLUR]
BLUR_SIGMA_RATIO = 0.2     # 블러 세기: 가림 영역 긴 변의 몇 배를 가우시안 시그마로 쓸지 (클수록 흐려진다)
BLUR_EDGE_RATIO = 0.06     # 블러 영역 가장자리를 부드럽게 섞는 폭 (짧은 변 대비 비율)

# 얼굴 신뢰도(conf)가 이 값보다 낮으면 "얼굴이 아닐 수 있다"는 경고를 보여 준다. (탐지 임계값 CONF 0.25 보다 높게)
# TODO: 위와 같이 팀에 알린 뒤 config.py 로 옮긴다.
LOW_CONF_WARN = 0.5

# 사진에서 YOLO 가 놓치는 작고 흐린 얼굴을 InsightFace 탐지기로 보완한다 (insightface 가 설치돼 있을 때만).
# 두 탐지기의 박스가 EXTRA_DETECTOR_IOU 이상 겹치면 같은 얼굴로 보고 하나만 쓴다.
# TODO: 위와 같이 팀에 알린 뒤 config.py 로 옮긴다.
USE_EXTRA_DETECTOR = True
EXTRA_DETECTOR_IOU = 0.25

# 얼굴 후보 · 등록된 인물 사진의 크기(픽셀 너비)와 한 줄에 놓는 후보 수
CANDIDATE_THUMB_WIDTH = 72
REGISTERED_THUMB_WIDTH = 72
CANDIDATES_PER_ROW = 5

# 얼굴 선택 컴포넌트에 보내는 미리보기 사진의 가로 최대 크기 (너무 크면 화면을 다시 그릴 때마다 느려진다)
PREVIEW_MAX_WIDTH = 1280

# 대기열: 대기 순번을 새로 확인하는 간격(초). 성능 지표: 화면에 보관할 최대 기록 수 / 표에 보일 개수.
WAIT_POLL_SECONDS = 1.0
METRICS_KEEP = 200
METRICS_SHOW = 10

# 화면 넘기기: 터치로 이만큼(픽셀) 밀면 다음 화면으로 넘어가고, 한 번 넘긴 뒤 이 시간(밀리초) 동안은 다시 넘기지 않는다.
PAGE_SWIPE_THRESHOLD = 50
PAGE_LOCK_MS = 800

# 상단 바와 브라우저 탭에 보이는 제목. 문구를 바꾸고 싶으면 여기만 고치면 된다.
APP_TITLE = "TRIPLE Y MINI"
APP_SUBTITLE = "얼굴 자동 모자이크"

# 처리 단계 표시줄에 보여 줄 단계 이름. 번호(0부터)는 on_image 의 on_step 으로 넘어온다.
# 영상은 process_video 가 프레임마다 탐지와 모자이크를 한꺼번에 하고 진행률만 알려 줘서, 둘을 한 단계로 묶었다.
STEPS_IMAGE = ["파일 확인", "얼굴 감지", "모자이크", "완료"]
STEPS_VIDEO = ["프레임 추출", "얼굴 감지·모자이크", "소리 합치기", "완료"]


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
/* 화면별 색 톤: 맨 첫 화면(메인)만 민트·하늘색이고, 인물 등록 화면과 '지금 바로 모자이크해 보세요' 화면은 같은 연한 분홍·연보라다.
   <html data-tone="main|register|work"> 는 PAGING_JS 가 스크롤 위치를 보고 붙인다. 색은 서서히 바뀌게 @property 로 등록한다. */
@property --tone-a { syntax: '<color>'; inherits: true; initial-value: #CDEFE2; }
@property --tone-b { syntax: '<color>'; inherits: true; initial-value: #DCF2EC; }
@property --tone-c { syntax: '<color>'; inherits: true; initial-value: #E0F1F4; }
@property --tone-d { syntax: '<color>'; inherits: true; initial-value: #D9ECFB; }
@property --wash-a { syntax: '<color>'; inherits: true; initial-value: rgba(205, 239, 226, 0.7); }
@property --wash-b { syntax: '<color>'; inherits: true; initial-value: rgba(217, 236, 251, 0.9); }
@property --wash-c { syntax: '<color>'; inherits: true; initial-value: rgba(234, 244, 253, 0.9); }
/* 상단 바는 네 색 그라데이션으로 그린다. 첫 화면과 인물 등록 화면은 이전의 세 색 그라데이션(민트 -> 연한 청록 -> 하늘색 /
   분홍 -> 연보라 -> 연한 하늘 보라)과 같은 모양이 되도록 중간 색을 이어 붙였고, 업로드 화면만 네 색을 섞은 새 그라데이션이다. */
.stApp {
    --tone-a: #CDEFE2; --tone-b: #DCF2EC; --tone-c: #E0F1F4; --tone-d: #D9ECFB;    /* 첫 화면(메인): 민트 -> 하늘색 */
    --wash-a: rgba(205, 239, 226, 0.7); --wash-b: rgba(217, 236, 251, 0.9); --wash-c: rgba(234, 244, 253, 0.9);   /* 배경 번짐 */
    --bar-text: "__BAR_MAIN__";
    transition: --tone-a .7s, --tone-b .7s, --tone-c .7s, --tone-d .7s, --wash-a .7s, --wash-b .7s, --wash-c .7s;
    background:
        radial-gradient(900px 520px at 6% 0%, var(--wash-a), transparent 70%),
        radial-gradient(900px 560px at 96% 100%, var(--wash-b), transparent 70%),
        radial-gradient(700px 420px at 100% 30%, var(--wash-c), transparent 70%),
        var(--bg);
    background-attachment: fixed;
}
html[data-tone="register"] .stApp {
    --tone-a: #F8DCE6; --tone-b: #F5E3F0; --tone-c: #EEE6F6; --tone-d: #E3E6FB;    /* 인물 등록 화면: 분홍 -> 연보라 */
    --wash-a: rgba(250, 221, 232, 0.75); --wash-b: rgba(226, 226, 250, 0.9); --wash-c: rgba(240, 236, 252, 0.9);
}
html[data-tone="work"] .stApp {
    --tone-a: #F8DCE6; --tone-b: #E6DDF6; --tone-c: #D9ECFB; --tone-d: #CDEFE2;    /* 업로드 화면: 분홍 -> 연보라 -> 하늘색 -> 민트 */
    --wash-a: rgba(250, 221, 232, 0.75); --wash-b: rgba(226, 226, 250, 0.9); --wash-c: rgba(240, 236, 252, 0.9);
}
html[data-tone="register"] .stApp { --bar-text: "__BAR_REGISTER__"; }    /* 상단 바 문구는 인물 등록 화면에서만 바뀐다 */
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
    content: var(--bar-text);
    position: fixed; top: 0; left: 0; right: 0; height: 2.6rem; z-index: 1000;
    display: flex; align-items: center; justify-content: center;
    background: linear-gradient(90deg, var(--tone-a) 0%, var(--tone-b) 34%, var(--tone-c) 67%, var(--tone-d) 100%);
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

/* 인물 등록 화면: 첫 화면(.landing)과 같은 모양의 소개 화면. 화면 높이를 채우고, 이 화면은 맨 위가 아니므로 위로 당기지 않는다.
   스크롤 힌트는 스크롤해도 사라지지 않으며 내용 맨 아래에 놓는다. */
[class*="st-key-screen_register"] {
    position: relative; min-height: 100vh; box-sizing: border-box; padding: 4rem 1rem 5.5rem; gap: .6rem;   /* 위·아래 막대에 안 가릴 만큼만 띄워서, 낮은 화면에서도 한 화면에 들어오게 한다 */
    display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center;
}
[class*="st-key-screen_register"] .blob { position: absolute; border-radius: 50%; opacity: .75; pointer-events: none; animation: float 7s ease-in-out infinite; }
[class*="st-key-screen_register"] .blob.b1 { width: 220px; height: 220px; left: -6%; top: 16%; background: #F8DCE6; }
[class*="st-key-screen_register"] .blob.b2 { width: 150px; height: 150px; right: -4%; top: 26%; background: #E3E6FB; animation-delay: -2.5s; }
[class*="st-key-screen_register"] .landing-title mark { background: linear-gradient(transparent 62%, #F8DCE6 62%); color: inherit; padding: 0 .08em; }
[class*="st-key-screen_register"] .blob.b3 { width: 100px; height: 100px; left: 4%; bottom: 24%; background: var(--line); animation-delay: -4.5s; }
[class*="st-key-screen_register"] .pill { position: relative; }
[class*="st-key-screen_register"] .landing-desc { margin: 0 auto; }
[class*="st-key-screen_register"] .landing-tip { margin: .7rem auto 0; }   /* 안쪽 상자 안이라 가운데 맞춤을 직접 한다 */
/* 장식 원과 스크롤 힌트는 화면 전체(screen_register)를 기준으로 놓는다. Streamlit 이 감싸는 상자(position: relative)가 기준이 되지 않게 푼다 */
[class*="st-key-screen_register"] [data-testid="stElementContainer"]:has(.blob) { position: static; }
[class*="st-key-screen_register"] .landing-title { font-size: clamp(1.9rem, 4.6vw, 2.8rem); margin: 1rem 0; }
[class*="st-key-screen_register"] .scroll-hint { position: static; margin-top: .6rem; animation: bounce 1.8s ease-in-out infinite; animation-timeline: auto; animation-range: normal; }
@media (prefers-reduced-motion: reduce) { [class*="st-key-screen_register"] .blob, [class*="st-key-screen_register"] .scroll-hint { animation: none; } }
.landing-title {
    position: relative; margin: 1.3rem 0 1.2rem; color: var(--ink);
    font-size: clamp(2.6rem, 8vw, 5.4rem); font-weight: 800; line-height: 1.14; letter-spacing: -.03em;
}
.landing-title mark { background: linear-gradient(transparent 62%, var(--mint) 62%); color: inherit; padding: 0 .08em; }
.landing.compact .landing-title { font-size: clamp(1.9rem, 4.6vw, 2.8rem); }
.landing-tip { position: relative; max-width: 38rem; margin: .8rem 0 0; color: var(--muted); font-size: .92rem; line-height: 1.6; word-break: keep-all; text-wrap: balance; opacity: .9; }
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
[data-testid="stDownloadButton"] button,
[data-testid="stFileUploaderDropzone"] button {
    background: var(--ink); color: #fff !important; border: none; border-radius: 999px;
    padding: .55rem 1.5rem; font-weight: 600; transition: background .15s;
}
.stButton > button *,
[data-testid="stDownloadButton"] button *,
[data-testid="stFileUploaderDropzone"] button * { color: #fff !important; }
.stButton > button:hover,
[data-testid="stDownloadButton"] button:hover,
[data-testid="stFileUploaderDropzone"] button:hover { background: var(--ink-soft); border: none; }
.stButton > button:focus:not(:active),
[data-testid="stDownloadButton"] button:focus:not(:active) { border: none; box-shadow: 0 0 0 3px rgba(159, 217, 194, 0.55); }
.stButton > button:disabled,
[data-testid="stDownloadButton"] button:disabled { opacity: .35; cursor: not-allowed; }   /* 누를 수 없는 버튼은 흐리게 */

/* 비교 카드 머리줄: 제목은 왼쪽, 버튼(결과 저장 · 적용)은 오른쪽 끝에 붙인다 */
[class*="st-key-result_head"] .card-desc { margin-bottom: 0; }
/* 원본·결과 바로 아래 한 줄: 왼쪽 칸에 [적용], 오른쪽 칸에 [저장]. 두 버튼은 각자 칸(= 원본·결과 사진 아래)의 가운데에 놓는다 */
[class*="st-key-apply_row"] .stButton,
[class*="st-key-apply_row"] [data-testid="stDownloadButton"] { display: flex; justify-content: center; }
[class*="st-key-apply_row"] { margin-top: .8rem; }
/* 사진이 여러 장일 때 사진마다 이름표 아래에 놓는 저장 버튼과 인물 카드의 [유지]/[초기화] 는 가운데에 놓는다 */
[class*="st-key-save_row_"] [data-testid="stDownloadButton"],
[class*="st-key-keep_ask"] .stButton { display: flex; justify-content: center; }
/* 인물 등록 카드 제목줄: 제목은 왼쪽, [모자이크 제외할 얼굴 선택] 버튼은 오른쪽 끝. 좁아지면 버튼이 제목 아래 줄로 내려간다 */
[class*="st-key-person_head"] {
    display: flex !important; flex-direction: row !important; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: .5rem 1rem;
}
[class*="st-key-person_head"] > div,
[class*="st-key-person_head"] [data-testid="stMarkdown"],
[class*="st-key-person_head"] .stButton { width: auto !important; flex: 0 0 auto; height: auto; }
[class*="st-key-person_head"] [data-testid="stMarkdown"],
[class*="st-key-person_head"] [data-testid="stMarkdown"] > div { margin: 0 !important; }
/* 얼굴 선택 · 적용 · 결과 저장 버튼은 조금 작게(컴팩트) */
[class*="st-key-apply_row"] .stButton > button,
[class*="st-key-keep_ask"] .stButton > button,
[class*="st-key-find_candidates"] .stButton > button,
[class*="st-key-apply_row"] [data-testid="stDownloadButton"] button,
[class*="st-key-save_row_"] [data-testid="stDownloadButton"] button { padding: .3rem 1.1rem; min-height: 2rem; }
[class*="st-key-apply_row"] .stButton > button p,
[class*="st-key-keep_ask"] .stButton > button p,
[class*="st-key-find_candidates"] .stButton > button p,
[class*="st-key-apply_row"] [data-testid="stDownloadButton"] button p,
[class*="st-key-save_row_"] [data-testid="stDownloadButton"] button p { font-size: .85rem; white-space: nowrap; }
[class*="st-key-card_person"] { padding: 1rem 1.4rem 1.2rem; }
/* 성능 지표 카드: 제목 · 설명 · 안내 글 사이와 카드 안쪽 여백을 넉넉하게 */
[class*="st-key-card_metrics"] { padding: 2.6rem 2.8rem 2.8rem; gap: 1.6rem; }
[class*="st-key-card_metrics"] .card-title { margin-bottom: .6rem; }
[class*="st-key-card_metrics"] .card-desc { margin: .6rem 0 .4rem; line-height: 1.9; }
[class*="st-key-card_metrics"] .stat-row { margin-bottom: 1.6rem; }
/* 새 파일을 올렸을 때 '등록한 인물을 유지할까요?' 묻는 상자 */
[class*="st-key-keep_ask"] { background: var(--sky-soft); border-radius: 14px; padding: .7rem 1rem .5rem; margin: .4rem 0 .8rem; text-align: center; }
[class*="st-key-keep_ask"] .keep-text { margin: 0 0 .4rem; color: var(--ink); font-size: .9rem; font-weight: 600; word-break: keep-all; }
[class*="st-key-keep_ask"] [data-testid="stHorizontalBlock"] { justify-content: center; gap: .6rem; }
/* 일반 버튼의 글씨도 Browse files 버튼처럼 굵게 (같은 알약 버튼으로 보이게) */
.stButton > button p,
[data-testid="stDownloadButton"] button p { font-weight: 600; }

/* 얼굴 후보의 '등록' 버튼: 평소에는 연한 민트 태그(단계 표시줄의 완료 태그와 같은 톤), 등록되면 민트 → 하늘색 선택 색으로 확 바뀌는 켜짐 상태 */
[class*="st-key-pick_off_"] .stButton > button,
[class*="st-key-pick_on_"] .stButton > button { padding: .15rem .9rem; min-height: 0; font-size: .82rem; border: 1px solid transparent; }
[class*="st-key-pick_off_"] .stButton > button { background: var(--mint-soft); border-color: var(--mint-line); }
[class*="st-key-pick_off_"] .stButton > button * { color: var(--ink) !important; }
[class*="st-key-pick_off_"] .stButton > button:hover { background: var(--mint); border: 1px solid var(--mint-line); }
[class*="st-key-pick_on_"] .stButton > button { background: var(--select); }
[class*="st-key-pick_on_"] .stButton > button * { color: var(--select-ink) !important; font-weight: 700; }
[class*="st-key-pick_on_"] .stButton > button:hover { background: var(--select); border: 1px solid transparent; filter: brightness(.96); }
[class*="st-key-pick_off_"] .stButton, [class*="st-key-pick_on_"] .stButton { margin-top: .2rem; }

/* 인물 등록 카드: 등록된 인물 줄(작은 사진 + 태그 + 해제 버튼)과 맨 아래의 '등록 인물 제외' 스위치 */
/* 등록된 사람들 줄: 태그 · 작은 사진들 · 등록 해제 버튼이 한 줄에 가지런히 놓이고, 좁아지면 줄만 바뀐다 */
[class*="st-key-reg_row"] {
    display: flex !important; flex-direction: row !important; flex-wrap: wrap; align-items: center; gap: .7rem 1.6rem;
    margin-top: .8rem; padding-top: .9rem; border-top: 1px solid var(--line);
}
[class*="st-key-reg_row"] > div { width: auto !important; flex: 0 0 auto; }
/* 작은 사진 묶음은 줄에 남는 폭을 모두 쓰고(사진이 몇 장이든 가로로 놓임), 모자라면 사진만 다음 줄로 넘어간다 */
[class*="st-key-reg_row"] > div:has([class*="st-key-reg_thumbs"]) { flex: 1 1 14rem; min-width: 0; }
[class*="st-key-reg_row"] > div:has([class*="st-key-reg_thumbs"]) > div { width: 100% !important; }
[class*="st-key-reg_thumbs"] { display: flex !important; flex-direction: row !important; flex-wrap: wrap; gap: .5rem; width: 100%; }
[class*="st-key-reg_thumbs"] > div { width: auto !important; flex: 0 0 auto; }
/* 후보 격자: 칸 너비가 일정하고(104px 이상) 줄마다 칸이 맞게 늘어서며, 창 너비에 따라 한 줄에 놓이는 개수만 바뀐다 */
[class*="st-key-cgrid"] { display: grid !important; grid-template-columns: repeat(auto-fill, minmax(104px, 1fr)); gap: .9rem .6rem; margin-top: .6rem; }
[class*="st-key-cand_"] { gap: .25rem; align-items: flex-start; min-width: 0; }
[class*="st-key-pick_off_"] .stButton > button p, [class*="st-key-pick_on_"] .stButton > button p { white-space: nowrap; }
[class*="st-key-clear_row"] .stButton > button { padding: .2rem .9rem; min-height: 0; margin-top: .3rem; }
[class*="st-key-clear_row"] .stButton > button p { font-size: .8rem; }
/* 원본·결과 이름표 줄: 두 이름표를 같은 높이에 가운데 정렬로 놓는다 */
[class*="st-key-cap_row"] [data-testid="stCaptionContainer"] { text-align: center; }
.reg-tag {
    display: inline-block; background: #fff; color: var(--muted); border: 1px solid var(--line);
    font-size: .8rem; font-weight: 600; padding: .25rem .9rem; border-radius: 999px;
}
.reg-tag.on { background: var(--select); color: var(--select-ink); border-color: transparent; font-weight: 700; }
/* 비교 카드 안의 파일 선택 칸: 안내 글·점선 테두리를 걷어 내고 버튼만 남긴다 (버튼 모양은 위와 같다) */
[class*="st-key-result_uploader"] [data-testid="stFileUploaderDropzone"] { background: none; border: none; padding: 0; justify-content: center; }
[class*="st-key-result_uploader"] [data-testid="stFileUploaderDropzoneInstructions"],
[class*="st-key-result_uploader"] [data-testid="stWidgetLabel"] { display: none; }
[class*="st-key-result_uploader"] { display: flex; flex-direction: column; align-items: center; }   /* 올린 뒤의 작은 Browse files 버튼은 사진/영상 전환 버튼 바로 아래 가운데 */

/* 처리 단계 표시줄: 지난 단계는 민트, 지금 단계는 선택 색(민트 → 하늘색), 남은 단계는 흰색 */
.steps { display: flex; flex-wrap: wrap; gap: .6rem 1.1rem; margin: .5rem 0 1.4rem; }    /* 뱃지끼리는 넉넉히, 아래 원본·결과 사진과는 떨어뜨린다 */
.steps:has(+ .steps-note) { margin-bottom: .6rem; }
.step {
    background: #fff; color: var(--muted); border: 1px solid var(--line);
    font-size: .8rem; font-weight: 600; padding: .25rem .9rem; border-radius: 999px;
}
.step.done { background: var(--mint); color: var(--ink); border-color: var(--mint); }
.step.now { background: var(--select); color: var(--select-ink); border-color: transparent; font-weight: 700; }
.steps-note { color: var(--muted); font-size: .85rem; margin: 0 0 1.4rem; }

/* 결과 파일 정보 · 개인정보 안내: 작은 회색 글씨 */
.result-info { color: var(--muted); font-size: .85rem; line-height: 1.7; margin-top: .8rem; word-break: keep-all; }
.result-info code { background: var(--line); color: var(--ink); padding: .05rem .4rem; border-radius: 6px; font-size: .82em; }
.privacy-note {
    max-width: 40rem; margin: 3rem auto 0; text-align: center;
    color: var(--muted); font-size: .82rem; line-height: 1.8; word-break: keep-all; text-wrap: balance;
}

/* 성능 지표: 요약 수치 4칸 + 최근 기록 표 (연한 하늘색 칸, 얇은 회색 선) */
.stat-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: .8rem; margin: .2rem 0 1.1rem; }
.stat { background: var(--sky-soft); border-radius: 14px; padding: .8rem 1rem; text-align: center; }
.stat-num { font-size: 1.5rem; font-weight: 800; color: var(--ink); line-height: 1.3; }
.stat-label { font-size: .8rem; color: var(--muted); }
.perf-table { width: 100%; border-collapse: collapse; font-size: .88rem; }
.perf-table th { color: var(--muted); font-weight: 600; text-align: left; border-bottom: 1px solid var(--line); padding: .4rem .5rem; }
.perf-table td { border-bottom: 1px solid var(--line); padding: .4rem .5rem; word-break: break-all; }
@media (max-width: 520px) { .stat-row { grid-template-columns: repeat(2, 1fr); } }

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
   [왼쪽: 메인] - [가운데: YYY 로고] - [오른쪽: 인물 등록]
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

/* 하단 메뉴의 선택 색은 지금 보고 있는 화면을 따라간다: 메인 화면이면 [메인], 인물 등록 화면이면 [인물 등록].
   (실제 라디오 값은 바뀌지 않고, PAGING_JS 가 정하는 <html data-tone> 으로만 색을 옮긴다) */
[class*="st-key-bottom_bar"] label[data-baseweb="radio"]::before { transition: opacity .3s; }
html:is([data-tone="register"], [data-tone="work"]) [class*="st-key-bottom_bar"] label[data-baseweb="radio"]:nth-of-type(1)::before { opacity: 0; }
html:is([data-tone="register"], [data-tone="work"]) [class*="st-key-bottom_bar"] label[data-baseweb="radio"]:nth-of-type(1) { color: var(--ink); font-weight: 400; }
html:is([data-tone="register"], [data-tone="work"]) [class*="st-key-bottom_bar"] label[data-baseweb="radio"]:nth-of-type(2)::before { opacity: 1; }
html:is([data-tone="register"], [data-tone="work"]) [class*="st-key-bottom_bar"] label[data-baseweb="radio"]:nth-of-type(2) { color: var(--select-ink); font-weight: 700; }
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

class TicketQueue:
    """모델을 쓰는 처리(사진 · 영상)를 번호표 순서대로 한 건씩만 하게 하는 대기열.

    추적 상태가 모델 하나에 묶여 있어서, 두 사람이 동시에 처리하면 추적 ID가 섞인다.
    그래서 동시에 하나만 처리하고 나머지는 번호표를 들고 차례를 기다린다.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._waiting: list[int] = []           # 기다리는 번호표 (앞이 먼저)
        self._active: Optional[int] = None      # 지금 처리 중인 번호표
        self._next = 1

    def join(self) -> int:
        """번호표를 뽑고 줄 맨 뒤에 선다."""
        with self._lock:
            ticket, self._next = self._next, self._next + 1
            self._waiting.append(ticket)
            return ticket

    def try_start(self, ticket: int) -> bool:
        """내 차례(처리 중인 사람이 없고 줄 맨 앞)이면 처리를 시작한 것으로 하고 True 를 돌려준다."""
        with self._lock:
            if self._active is None and self._waiting and self._waiting[0] == ticket:
                self._waiting.pop(0)
                self._active = ticket
                return True
            return False

    def rank(self, ticket: int) -> int:
        """대기 순번 (1 = 지금 처리 중인 작업 다음). 줄에 없으면 0."""
        with self._lock:
            return self._waiting.index(ticket) + 1 if ticket in self._waiting else 0

    def leave(self, ticket: int) -> None:
        """처리를 마쳤거나 취소했을 때 자리를 반납한다. 여러 번 불러도 안전하다."""
        with self._lock:
            if self._active == ticket:
                self._active = None
            if ticket in self._waiting:
                self._waiting.remove(ticket)


@st.cache_resource
def get_queue() -> TicketQueue:
    """접속한 모든 사람이 함께 쓰는 대기열 하나를 돌려준다. (cache_resource 덕분에 서버에 하나만 만들어진다)"""
    return TicketQueue()


@contextmanager
def processing_slot(on_wait: Optional[Callable[[int], None]] = None, wait: bool = True) -> Iterator[None]:
    """처리 자리를 하나 잡는다. 차례가 올 때까지 기다리며, 기다리는 동안 on_wait(대기 순번)을 부른다.

    wait=False 이면 기다리지 않고, 자리가 없을 때 바로 BusyError 를 낸다. (끝이 없는 실시간 처리용)
    `with processing_slot():` 블록이 끝나면 오류가 나도, 사용자가 취소해도 자리를 반드시 반납한다.
    on_wait 안에서 st.xxx 를 부르면, 취소 버튼을 눌렀을 때 Streamlit 이 이 대기를 끊어 준다.
    """
    queue = get_queue()
    ticket = queue.join()
    try:
        while not queue.try_start(ticket):
            if not wait:
                raise BusyError("다른 영상을 처리 중입니다. 잠시 후 다시 시도해 주세요.")
            if on_wait:
                on_wait(queue.rank(ticket))
            time.sleep(WAIT_POLL_SECONDS)
        yield
    finally:
        queue.leave(ticket)


def init_session_state() -> None:
    """화면이 다시 실행되어도 값이 남도록 session_state 에 기본값을 만든다.

    Streamlit 은 버튼을 누르거나 메뉴를 바꿀 때마다 파일 전체를 처음부터 다시 실행한다.
    그래서 계속 기억해야 하는 값은 st.session_state 에 넣어 둔다.
    """
    st.session_state.setdefault("ref_embs", None)         # 등록된 제외 인물 벡터 (없으면 None)
    st.session_state.setdefault("candidates", None)       # 등록 화면의 얼굴 후보 목록
    st.session_state.setdefault("candidates_for", None)   # 그 후보를 찾은 파일 (file_id)
    st.session_state.setdefault("keep_question", False)   # 새 파일을 올렸을 때 '등록한 인물을 유지할까요?' 를 묻는 중인지
    st.session_state.setdefault("ref_people", {})         # 등록된 사람들 {(후보를 찾은 파일 file_id, 후보 번호): {"embs": 얼굴 벡터, "thumb": 대표 얼굴 사진(RGB)}}. 여러 명 등록 가능
    st.session_state.setdefault("image_results", {})      # 사진 처리 결과 {결과 키: 탐지 박스·편집 내용·파일 정보}. 저장 버튼을 눌러 화면이 다시 그려져도 남긴다
    st.session_state.setdefault("video_result", None)     # 영상 처리 결과 {"key", "path", "info", "audio", "codec", "effect"}
    st.session_state.setdefault("metrics", [])            # 성능 지표 기록 (이 접속에서 처리한 것)
    st.session_state.setdefault("run_counter", 0)         # 처리를 시작할 때마다 1씩 올려, 얼굴 선택 칸을 새로 만든다


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


def square_thumb(image_rgb: np.ndarray, side: int) -> np.ndarray:
    """얼굴 사진을 비율을 유지한 채 side × side 정사각형 안에 가운데로 넣는다. 후보마다 사진 높이가 달라 버튼 줄이 어긋나는 것을 막는다."""
    h, w = image_rgb.shape[:2]
    scale = side / max(h, w)
    resized = cv2.resize(image_rgb, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    canvas = np.full((side, side, 3), 238, np.uint8)    # 연한 회색 바탕
    y0, x0 = (side - resized.shape[0]) // 2, (side - resized.shape[1]) // 2
    canvas[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
    return canvas


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
    """아직 없는 팀원 모듈(src.*)이나 설치되지 않은 패키지를 불렀을 때 보여 주는 안내."""
    name = err.name or "?"
    if "DLL load failed" in (err.msg or ""):    # torch 를 먼저 불러온 뒤 onnx 를 불러서 생긴 문제. 프로그램을 새로 시작하면 해결된다
        st.warning("InsightFace(onnx)를 불러오지 못했습니다(DLL 오류). 앱을 완전히 종료했다가 다시 실행해 주세요.")
    elif name.split(".")[0] == "src":
        st.warning(f"아직 연결되지 않은 모듈입니다: `{name}` (담당 팀원이 만들면 자동으로 동작합니다)")
    else:
        st.warning(f"필요한 패키지가 설치돼 있지 않습니다: `{name}`. 터미널에서 `pip install -r requirements.txt` 를 실행한 뒤 다시 시도해 주세요.")


def inject_css() -> None:
    """위의 APP_CSS 를 화면에 적용한다. 페이지를 그리기 전에 한 번 불러야 한다."""
    css = APP_CSS.replace("__BAR_MAIN__", f"{APP_TITLE}  ·  {APP_SUBTITLE}").replace("__BAR_REGISTER__", f"{APP_TITLE}  ·  {MENU_REGISTER}")
    st.markdown(css, unsafe_allow_html=True)


def render_landing(pill: str, title_html: str, desc: str, compact: bool = False) -> None:
    """첫 화면(랜딩): 화면 높이를 꽉 채우는 제목과 "스크롤하세요" 안내를 보여 준다.

    compact=True 이면 제목 글씨를 작게 한다.
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


def as_file_list(value: object) -> list[UploadedFile]:
    """file_uploader 의 값(없음 · 파일 하나 · 파일 여러 개)을 항상 리스트로 바꿔 준다."""
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


def active_ref() -> Optional[np.ndarray]:
    """이번 처리에 쓸 제외 인물 벡터(등록한 모든 사람의 벡터를 한데 모은 것). 등록한 사람이 없으면 None."""
    return st.session_state["ref_embs"]


def ref_tag() -> int:
    """제외 인물 설정이 바뀌었는지 알아보는 표식. 등록이 없으면 0, 있으면 그 벡터 객체의 id. (등록·해제할 때마다 새 객체가 만들어진다)"""
    ref = active_ref()
    return 0 if ref is None else id(ref)


def render_steps(box: object, names: list[str], current: int, note: str = "") -> None:
    """처리 단계 표시줄을 box(st.empty) 자리에 그린다. current 앞 단계는 완료, current 는 진행 중.

    current 가 단계 수와 같으면 모두 완료된 것이다. note 는 표시줄 아래의 작은 안내(파일 이름 등)다.
    """
    items = []
    for i, name in enumerate(names):
        state = "done" if i < current else ("now" if i == current else "")
        mark = "✓ " if i < current else ""
        items.append(f'<span class="step {state}">{mark}{i + 1}. {html.escape(name)}</span>')
    note_html = f'<p class="steps-note">{html.escape(note)}</p>' if note else ""
    box.markdown(f'<div class="steps">{"".join(items)}</div>{note_html}', unsafe_allow_html=True)


def render_result_info(lines: list[str], folder: Path) -> None:
    """결과 파일의 이름·크기와 서버 임시 저장 위치를 작은 회색 글씨로 보여 준다.

    lines 는 이미 HTML 로 안전하게 만든(html.escape 한) 줄들이다.
    브라우저로 내려받은 파일이 어디에 놓이는지는 서버가 알 수 없어서 안내 문구로 대신한다.
    """
    body = "<br>".join(lines)
    st.markdown(
        f'<div class="result-info">{body}<br>'
        f"서버 임시 위치 <code>{html.escape(str(folder))}</code><br>"
        "‘결과 저장’을 누르면 브라우저의 다운로드 폴더에 저장됩니다.</div>",
        unsafe_allow_html=True,
    )


def format_size(num_bytes: int) -> str:
    """바이트 수를 KB·MB 글자로 바꾼다."""
    if num_bytes >= 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.1f} MB"
    return f"{max(1, round(num_bytes / 1024))} KB"


def apply_blur(img: np.ndarray, boxes: list[dict], pad: float = PAD) -> np.ndarray:
    """BGR 이미지의 얼굴 박스에 부드러운 블러를 씌운 새 이미지를 돌려준다. 입력 이미지는 바꾸지 않는다.

    apply_mosaic 과 같은 방식으로 박스에 pad 비율만큼 여백을 더하고 이미지 범위로 자른다.
    영역은 타원이고, 가장자리는 서서히 섞어서 경계선이 티 나지 않게 한다.
    (B파트 mosaic.py 는 건드리지 않으려고 화면 쪽에 둔 함수다. 영상은 아직 지원하지 않는다.)
    """
    out = img.copy()
    img_h, img_w = img.shape[:2]
    for box in boxes:
        x1, y1, x2, y2 = box["xyxy"]
        px, py = (x2 - x1) * pad, (y2 - y1) * pad
        x1, y1 = max(math.floor(x1 - px), 0), max(math.floor(y1 - py), 0)
        x2, y2 = min(math.ceil(x2 + px), img_w), min(math.ceil(y2 + py), img_h)
        if x2 <= x1 or y2 <= y1:
            continue

        roi = out[y1:y2, x1:x2]    # out 의 뷰라서 쓰면 out 에 반영된다
        h, w = roi.shape[:2]
        blurred = cv2.GaussianBlur(roi, (0, 0), max(w, h) * BLUR_SIGMA_RATIO)

        mask = np.zeros((h, w), np.float32)
        cv2.ellipse(mask, (w // 2, h // 2), (w // 2, h // 2), 0, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), max(1.0, min(w, h) * BLUR_EDGE_RATIO))[..., None]
        roi[:] = (roi * (1.0 - mask) + blurred * mask).astype(np.uint8)
    return out


def mark_cancelled() -> None:
    """'처리 취소' 버튼의 콜백. 다음 화면에 취소 안내를 한 번 보여 주려고 표시만 해 둔다."""
    st.session_state["job_cancelled"] = True


class JobUI:
    """처리하는 동안의 화면 자리: 단계 표시줄 · 취소 버튼 · 진행률 막대·오류가 들어가는 영역을 한데 묶는다."""

    def __init__(self, area: object, names: list[str]) -> None:
        self.area = area                       # 영역 전체 (st.container). with ui.area: 로 안에 그린다
        self.names = names
        with area:
            self._steps_box = st.empty()
            self._cancel_box = st.empty()

    def steps(self, current: int, note: str = "") -> None:
        """단계 표시줄을 갱신한다. current 가 단계 수와 같으면 모두 완료."""
        render_steps(self._steps_box, self.names, current, note)

    def waiting(self, rank: int) -> None:
        """대기 중 화면: 단계는 아직 시작 전이고, 몇 번째로 기다리는지 보여 준다."""
        render_steps(self._steps_box, self.names, -1, f"{rank}번째 대기 중입니다. 앞의 작업이 끝나면 자동으로 시작합니다.")

    def show_cancel(self) -> None:
        """'처리 취소' 버튼을 보인다. 누르면 Streamlit 이 지금 실행 중인 처리를 끊고 대기열 자리도 반납된다."""
        self._cancel_box.button("처리 취소", key="cancel_job", on_click=mark_cancelled)

    def clear(self) -> None:
        """단계 표시줄과 취소 버튼을 지운다. (실패했거나 취소됐을 때)"""
        self._steps_box.empty()
        self._cancel_box.empty()

    def finish(self, note: str) -> None:
        """모든 단계를 완료로 표시하고 취소 버튼을 지운다."""
        self._cancel_box.empty()
        self.steps(len(self.names), note)


def record_metric(kind: str, name: str, seconds: float, faces: int, frames: int = 1, fps: Optional[float] = None) -> None:
    """성능 지표 기록 한 줄을 남긴다. fps 를 모르면 frames / seconds 로 계산한다. (사진 한 장은 frames=1)"""
    rows = st.session_state["metrics"]
    rows.append({
        "at": time.strftime("%H:%M:%S"),
        "kind": kind,
        "name": name,
        "seconds": float(seconds),
        "faces": int(faces),
        "fps": float(fps) if fps is not None else (frames / seconds if seconds > 0 else 0.0),
    })
    del rows[:-METRICS_KEEP]


def find_ffmpeg() -> Optional[str]:
    """ffmpeg 실행 파일 경로를 돌려준다. 시스템 PATH 에 없으면 imageio-ffmpeg 패키지(설치돼 있을 때)에 들어 있는 것을 쓴다. 없으면 None."""
    found = shutil.which("ffmpeg")
    if found is not None:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:    # 패키지가 없거나 실행 파일을 못 찾으면 ffmpeg 없음으로 본다
        return None


def has_audio_stream(path: str) -> Optional[bool]:
    """영상 파일에 소리(오디오 트랙)가 있는지 알아낸다. ffprobe 를 먼저 쓰고, 없으면 ffmpeg 출력에서 찾는다. 둘 다 없거나 읽지 못하면 None."""
    ffprobe, ffmpeg = shutil.which("ffprobe"), find_ffmpeg()
    try:
        if ffprobe is not None:
            done = subprocess.run(
                [ffprobe, "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", path],
                capture_output=True, text=True, timeout=15,
            )
            return bool(done.stdout.strip()) if done.returncode == 0 else None
        if ffmpeg is not None:
            done = subprocess.run([ffmpeg, "-hide_banner", "-i", path], capture_output=True, text=True, timeout=15)
            if "Invalid data" in done.stderr or "No such file" in done.stderr:    # ffmpeg -i 는 출력 없이 항상 실패 코드를 내므로 글로 판단한다
                return None
            return "Audio:" in done.stderr
    except (OSError, subprocess.SubprocessError):
        return None
    return None


def merge_audio(in_path: str, out_path: str) -> bool:
    """처리된 영상(out_path)에 원본(in_path)의 소리를 합쳐 out_path 를 바꿔 쓴다. 성공하면 True.

    화면(영상)은 다시 인코딩하지 않고 그대로 복사하고, 소리는 AAC 로 넣는다. 영상이 더 짧으면 거기에 맞춘다.
    ffmpeg 가 없거나 실패하면 결과 영상은 그대로 두고 False 를 돌려준다.
    """
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        return False
    tmp = str(Path(out_path).with_name(Path(out_path).stem + "_audio.mp4"))
    try:
        done = subprocess.run(
            [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", out_path, "-i", in_path,
             "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-shortest", tmp],
            capture_output=True, text=True, timeout=300,
        )
        if done.returncode != 0 or not Path(tmp).exists() or Path(tmp).stat().st_size == 0:
            return False
        os.replace(tmp, out_path)
        return True
    except (OSError, subprocess.SubprocessError):
        return False
    finally:
        if Path(tmp).exists():
            Path(tmp).unlink()


_H264_NAMES = {"h264", "avc1", "avc", "x264"}    # 브라우저(st.video)가 재생하는 H.264 의 이름들 (ffprobe 는 h264, OpenCV FOURCC 는 avc1/h264)


def video_codec(path: str) -> Optional[str]:
    """영상의 비디오 코덱 이름(소문자)을 알아낸다. ffprobe 가 있으면 그것으로, 없으면 OpenCV 의 FOURCC 로. 못 읽으면 None."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is not None:
        try:
            done = subprocess.run(
                [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name", "-of", "csv=p=0", path],
                capture_output=True, text=True, timeout=15,
            )
            if done.returncode == 0 and done.stdout.strip():
                return done.stdout.strip().lower()
        except (OSError, subprocess.SubprocessError):
            pass
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return None
        code = int(cap.get(cv2.CAP_PROP_FOURCC))
    finally:
        cap.release()
    if code <= 0:
        return None
    return "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00 ").lower() or None


def transcode_h264(path: str) -> bool:
    """영상(path)을 H.264(+ 브라우저 호환 yuv420p, 빠른 시작) 로 다시 인코딩해 같은 이름으로 바꿔 쓴다. 소리는 그대로 복사한다. 성공하면 True.

    ffmpeg 가 없거나 실패하면 원래 영상은 그대로 두고 False 를 돌려준다.
    """
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        return False
    tmp = str(Path(path).with_name(Path(path).stem + "_h264.mp4"))
    try:
        done = subprocess.run(
            [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", path, "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", "-c:a", "copy", tmp],
            capture_output=True, text=True, timeout=600,
        )
        if done.returncode != 0 or not Path(tmp).exists() or Path(tmp).stat().st_size == 0:
            return False
        os.replace(tmp, path)
        return True
    except (OSError, subprocess.SubprocessError):
        return False
    finally:
        if Path(tmp).exists():
            Path(tmp).unlink()


def ensure_h264(path: str) -> tuple[str, bool]:
    """결과 영상이 st.video 로 재생되는 H.264 인지 확인하고, 아니면 ffmpeg 로 다시 인코딩한다. (안내 문구, 경고 여부)

    영상 처리 모듈(B)이 명세대로 H.264 로 저장하면 아무 안내도 없다(빈 문구). OpenCV 의 기본 mp4v 로 저장한 영상은
    브라우저에서 재생되지 않으므로, ffmpeg 가 있으면 여기서 H.264 로 바꾸고 없으면 경고한다.
    """
    codec = video_codec(path)
    if codec is None or codec in _H264_NAMES:
        return "", False
    if transcode_h264(path):
        return f"브라우저에서 재생되도록 영상을 H.264 로 다시 만들었습니다 ({codec} → h264).", False
    return (f"결과 영상이 H.264 가 아니라서({codec}) 브라우저에서 재생되지 않을 수 있습니다. "
            "ffmpeg 를 설치하거나 영상 처리 모듈(B)이 H.264 로 저장하게 해 주세요."), True


def ensure_audio(in_path: str, out_path: str) -> tuple[str, bool]:
    """결과 영상에 원본 소리가 들어 있게 하고, (안내 문구, 경고 여부)를 돌려준다.

    영상 처리 모듈이 이미 소리를 합쳤으면 그대로 두고, 원본에 소리가 있는데 결과에서 빠졌다면 여기서 합쳐 넣는다.
    """
    before, after = has_audio_stream(in_path), has_audio_stream(out_path)
    if before is None or after is None:
        return "소리를 확인하지 못했습니다 (ffmpeg 가 설치돼 있지 않거나 파일을 읽지 못했습니다).", False
    if not before:
        return "원본에 소리가 없습니다.", False
    if after:
        return "원본의 소리가 결과 영상에 그대로 유지되었습니다.", False
    if merge_audio(in_path, out_path) and has_audio_stream(out_path):
        return "처리 중 빠진 원본 소리를 결과 영상에 다시 합쳤습니다.", False
    return "원본에는 있던 소리가 결과 영상에서 빠졌고 합치지도 못했습니다. ffmpeg 설치와 영상 처리 모듈(소리 병합)을 확인해 주세요.", True


# ------------------------------------------------------------
# 얼굴 선택 컴포넌트 (app/components/face_picker) 와 좌표 변환
# ------------------------------------------------------------

_face_picker = components.declare_component(
    "face_picker", path=str(Path(__file__).resolve().parent / "components" / "face_picker")
)


def canvas_layout(width: int, height: int, size: tuple[int, int] = DISPLAY_SIZE) -> dict:
    """원본 (width × height) 이 fit_to_canvas 의 캔버스 안 어디에 놓이는지 계산한다. fit_to_canvas 와 같은 식이다.

    돌려주는 값: 캔버스 크기 cw·ch, 사진이 놓이는 왼쪽 위 x·y 와 크기 w·h (캔버스 픽셀).
    """
    cw, ch = size
    scale = min(cw / width, ch / height)
    new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))
    return {"cw": cw, "ch": ch, "x": (cw - new_w) // 2, "y": (ch - new_h) // 2, "w": new_w, "h": new_h}


def to_canvas_box(xyxy: tuple[int, int, int, int], src_w: int, src_h: int) -> tuple[float, float, float, float]:
    """원본 픽셀 박스를 캔버스 좌표로 바꾼다."""
    lay = canvas_layout(src_w, src_h)
    sx, sy = lay["w"] / src_w, lay["h"] / src_h
    x1, y1, x2, y2 = xyxy
    return lay["x"] + x1 * sx, lay["y"] + y1 * sy, lay["x"] + x2 * sx, lay["y"] + y2 * sy


def from_canvas_box(box: list[int], src_w: int, src_h: int) -> tuple[int, int, int, int]:
    """캔버스 좌표 박스를 원본 픽셀 박스로 되돌린다. 원본 범위로 자르고 정수로 만든다."""
    lay = canvas_layout(src_w, src_h)
    sx, sy = src_w / lay["w"], src_h / lay["h"]
    x1, y1, x2, y2 = box[:4]
    px1, px2 = sorted([round((x1 - lay["x"]) * sx), round((x2 - lay["x"]) * sx)])
    py1, py2 = sorted([round((y1 - lay["y"]) * sy), round((y2 - lay["y"]) * sy)])
    return (max(0, min(px1, src_w)), max(0, min(py1, src_h)), max(0, min(px2, src_w)), max(0, min(py2, src_h)))


def to_data_url(image_rgb: np.ndarray, max_side: int) -> str:
    """RGB 이미지를 긴 변 max_side 이하의 JPEG data URL 로 바꾼다. (컴포넌트에 사진을 보낼 때 쓴다)"""
    h, w = image_rgb.shape[:2]
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        image_rgb = cv2.resize(image_rgb, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise ValueError("미리보기 이미지를 만들 수 없습니다.")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


@st.cache_data(show_spinner=False, max_entries=64)
def preview_data_url(file_id: str, _image_rgb: np.ndarray) -> str:
    """업로드한 사진의 미리보기 data URL. 같은 파일은 한 번만 만든다. (_ 로 시작하는 인자는 캐시 열쇠에서 뺀다)"""
    return to_data_url(_image_rgb, PREVIEW_MAX_WIDTH)


def face_picker_boxes(
    key: str,
    src: str,
    image_shape: tuple[int, int],
    detected: list[dict],
    excluded: list[int],
    manual: list[list[int]],
    reset_excluded: list[int],
    token: str,
) -> Optional[dict]:
    """사진·영상 프레임 위에서 얼굴을 클릭해 제외하고, 직접 영역을 그리는 컴포넌트를 보여 준다.

    src 는 미리보기 data URL, image_shape 는 원본 (높이, 너비), detected 는 탐지 박스 목록(원본 픽셀),
    excluded 는 처음부터 제외로 표시할 detected 번호, manual 은 이미 있는 직접 영역(원본 픽셀),
    reset_excluded 는 '초기화'를 눌렀을 때 돌아갈 제외 목록, token 은 이 값이 바뀌면 컴포넌트를 처음부터 다시 그린다.
    반환: 사용자가 만지기 전에는 None, 만진 뒤에는 {"excluded": [번호...], "manual": [[x1,y1,x2,y2]...]} (캔버스 좌표).
    """
    h, w = image_shape
    lay = canvas_layout(w, h)
    boxes = []
    for i, b in enumerate(detected):
        x1, y1, x2, y2 = to_canvas_box(b["xyxy"], w, h)
        boxes.append({"id": i, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": b["conf"], "auto": b.get("auto_excluded", False)})
    manual_canvas = [list(to_canvas_box(tuple(m), w, h)) for m in manual]
    return _face_picker(
        mode="boxes", cw=lay["cw"], ch=lay["ch"], img={"src": src, "x": lay["x"], "y": lay["y"], "w": lay["w"], "h": lay["h"]},
        boxes=boxes, excluded=excluded, manual=manual_canvas, lowConf=LOW_CONF_WARN, resetExcluded=reset_excluded,
        token=token, key=key, default=None,
    )


def parse_edit(value: object, fallback: dict, valid_ids: set, src_size: tuple[int, int]) -> dict:
    """컴포넌트가 돌려준 값을 {"excluded": [...], "manual": [[x1,y1,x2,y2], ...]} (원본 픽셀)로 정리한다.

    컴포넌트의 직접 영역은 캔버스 좌표라서 src_size(원본 너비, 높이)로 원본 픽셀로 되돌린다.
    값이 없거나 이상하면 fallback 을 그대로 쓴다. valid_ids 에 없는 번호는 버리고, 너무 작은 영역도 버린다.
    """
    if not isinstance(value, dict):
        return fallback
    try:
        excluded = sorted({int(i) for i in value.get("excluded", []) if int(i) in valid_ids})
        manual = []
        for m in value.get("manual", []):
            if len(m) < 4:
                continue
            x1, y1, x2, y2 = from_canvas_box([float(v) for v in m[:4]], *src_size)
            if x2 - x1 >= 2 and y2 - y1 >= 2:
                manual.append([x1, y1, x2, y2])
    except (TypeError, ValueError):
        return fallback
    return {"excluded": excluded, "manual": manual}


# ============================================================
# 3. 처리 함수 (명세 C장 ⑤) - 화면 코드(st.xxx)를 쓰지 않는다
# ============================================================

def on_register(files: list[UploadedFile]) -> list[dict]:
    """업로드한 사진(여러 장 가능)·영상에서 얼굴 후보 목록을 찾아 돌려준다.

    사진을 여러 장 넘기면 한꺼번에 살펴서 같은 사람은 하나의 후보로 묶는다. (정면 1장 + 좌우 측면 각 1장처럼
    한 사람의 사진을 여러 장 올려도 후보 하나에 그 사람의 여러 각도 얼굴 벡터가 모인다)

    반환: [{"thumb": 대표 얼굴(BGR), "embs": (k, 512) 벡터, "frames": 등장 횟수, "score": 점수}, ...]
          얼굴이 없으면 [] (화면에 "얼굴을 찾지 못했습니다" 표시)
    후보 중 한 명을 고르는 일은 화면 쪽(render_register_screen)에서 한다.
    """
    # ┌─────────────────────────────────────────────┐
    # │ [C파트 모듈] src/exclude.py : list_candidates │
    # └─────────────────────────────────────────────┘
    from src.exclude import list_candidates

    paths = [save_upload_to_temp(f, f"register_{i}") for i, f in enumerate(files)]
    return list_candidates(paths[0] if len(paths) == 1 else paths)


def box_iou(a: tuple, b: tuple) -> float:
    """두 박스 (x1, y1, x2, y2)의 겹침 비율(IoU, 0~1)."""
    ix, iy = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def merge_boxes(primary: list[dict], extra: list[dict], iou_threshold: float = EXTRA_DETECTOR_IOU) -> list[dict]:
    """primary 박스에, 어떤 primary 박스와도 겹치지 않는 extra 박스만 더해서 돌려준다. (같은 얼굴을 두 번 가리지 않는다)"""
    def same_face(a: tuple, b: tuple) -> bool:    # 많이 겹치거나, 한쪽 박스의 중심이 다른 쪽 안에 있으면 같은 얼굴이다
        def center_in(inner: tuple, outer: tuple) -> bool:
            cx, cy = (inner[0] + inner[2]) / 2, (inner[1] + inner[3]) / 2
            return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]
        return box_iou(a, b) >= iou_threshold or center_in(a, b) or center_in(b, a)

    merged = list(primary)
    for box in extra:
        if not any(same_face(tuple(box["xyxy"]), tuple(p["xyxy"])) for p in merged):
            merged.append(box)
    return merged


def find_extra_boxes(img_bgr: np.ndarray) -> list[dict]:
    """InsightFace 탐지기로 얼굴 박스를 더 찾는다. insightface 가 없거나 오류가 나면 [] (YOLO 결과만 쓴다)."""
    # ┌──────────────────────────────────────────────────────────────┐
    # │ [C파트 모듈] src/exclude.py : detect_faces_insight             │
    # └──────────────────────────────────────────────────────────────┘
    try:
        from src.exclude import detect_faces_insight

        return detect_faces_insight(img_bgr)
    except ImportError:
        return []
    except Exception:    # 보조 탐지가 실패해도 기본 탐지로 계속 진행한다
        return []


def on_detect(image_rgb: np.ndarray, ref_embs: Optional[np.ndarray]) -> list[dict]:
    """RGB 이미지에서 얼굴을 모두 찾고, 등록된 제외 인물이면 표시를 붙여 돌려준다.

    반환: [{"xyxy": (x1, y1, x2, y2), "conf": 신뢰도, "track_id": None, "auto_excluded": 등록 인물 여부}, ...]
          얼굴이 없으면 []. 등록 인물도 목록에는 남겨 두고, 화면에서 '제외된 얼굴' 로 보여 주려는 것이다.
    """
    # ┌─────────────────────────────────────────────────────────┐
    # │ [A파트 모듈] src/detector.py : detect_faces              │
    # │ [C파트 모듈] src/exclude.py  : filter_excluded           │
    # └─────────────────────────────────────────────────────────┘
    from src.detector import detect_faces

    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)   # 입구: 웹은 RGB, 모듈은 BGR
    boxes = detect_faces(img_bgr)
    if USE_EXTRA_DETECTOR:                                 # 놓친 얼굴이 없도록 InsightFace 탐지기로 보완 (프라이버시상 놓치는 쪽이 더 위험하다)
        boxes = merge_boxes(boxes, find_extra_boxes(img_bgr))
    kept: Optional[set] = None
    if ref_embs is not None:                               # 제외 인물이 등록돼 있을 때만
        from src.exclude import filter_excluded            # (등록 시에만 불러와 C파트 없이도 사진은 동작)

        kept = {tuple(b["xyxy"]) for b in filter_excluded(img_bgr, boxes, ref_embs)}   # 남은 것 = 모자이크할 얼굴
    return [
        {
            "xyxy": tuple(int(v) for v in b["xyxy"]),
            "conf": float(b["conf"]),
            "track_id": b.get("track_id"),
            "auto_excluded": kept is not None and tuple(b["xyxy"]) not in kept,
        }
        for b in boxes
    ]


def on_mask(image_rgb: np.ndarray, boxes: list[dict], effect: str = EFFECT_MOSAIC) -> np.ndarray:
    """RGB 이미지의 boxes 에 모자이크(또는 블러)를 씌운 새 RGB 이미지를 돌려준다. 입력 이미지는 바꾸지 않는다.

    boxes 는 [{"xyxy", "conf", "track_id"}, ...]. 얼굴을 다시 찾지 않으므로 빠르다.
    """
    # ┌─────────────────────────────────────────────────────────┐
    # │ [B파트 모듈] src/mosaic.py   : apply_mosaic              │
    # └─────────────────────────────────────────────────────────┘
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if effect == EFFECT_BLUR:
        result_bgr = apply_blur(img_bgr, boxes)
    else:
        from src.mosaic import apply_mosaic                # 블러는 B파트 없이도 동작

        result_bgr = apply_mosaic(img_bgr, boxes)
    return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)


def on_image(
    image_rgb: np.ndarray,
    ref_embs: Optional[np.ndarray],
    effect: str = EFFECT_MOSAIC,
    on_step: Optional[Callable[[int], None]] = None,
) -> tuple[np.ndarray, int]:
    """RGB 이미지의 얼굴을 모자이크(또는 블러)한다. (결과 RGB 이미지, 가린 얼굴 수)를 돌려준다.

    on_detect -> on_mask 를 이어 부르는 한 번에 처리용 함수다. (명세 C장 ⑤ 의 기본 사용법 그대로)
    on_step 은 단계 번호(1=얼굴 찾기, 2=얼굴 가리기)를 받는 함수로, 단계 표시줄을 갱신할 때 쓴다.
    """
    if on_step:
        on_step(1)
    detected = on_detect(image_rgb, ref_embs)
    boxes = [b for b in detected if not b["auto_excluded"]]
    if on_step:
        on_step(2)
    return on_mask(image_rgb, boxes, effect), len(boxes)


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


def process_video_demo(
    in_path: str,
    out_path: str,
    ref_embs: Optional[np.ndarray] = None,
    on_progress: Optional[Callable[[float], None]] = None,
    effect: str = EFFECT_MOSAIC,
) -> dict:
    """웹 데모용 영상 처리: 프레임마다 얼굴을 찾고(추적), 등록 인물을 빼고, 모자이크(또는 블러)해 out_path 에 저장한다.

    B 의 src/video.py(process_video)가 아직 없을 때 on_video 가 대신 쓴다. 사진과 같은 부품을 쓴다:
    A 의 detect_faces(track=True) -> C 의 filter_excluded(track 별 판정 캐시) -> B 의 apply_mosaic / 블러.
    탐지가 한두 프레임 끊겨도 가려진 얼굴이 노출되지 않도록 HOLD_FRAMES 동안 마지막 박스를 유지한다.
    반환: {"faces": 가린 인원(고유 track_id 수), "frames": 처리 프레임 수, "fps": 처리 속도, "seconds": 걸린 시간}
    저장은 OpenCV(mp4v)라서 브라우저 재생용 H.264 변환(ensure_h264)은 부르는 쪽이 한다.
    """
    from src.config import HOLD_FRAMES
    from src.detector import detect_faces, reset_tracker

    filter_excluded = None
    if ref_embs is not None:
        from src.exclude import filter_excluded    # 등록 인물이 있을 때만 불러온다

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise ValueError("영상을 열 수 없습니다. 파일이 깨졌는지 확인해 주세요.")
    total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer: Optional[cv2.VideoWriter] = None
    cache: dict = {}                                  # filter_excluded 의 track 별 판정 캐시 (영상마다 새로)
    held: dict = {}                                   # 끊긴 track 의 마지막 박스 {track_id: (박스, 마지막으로 본 프레임)}
    covered: set = set()                              # 가린 적 있는 track_id
    started = time.perf_counter()
    index = 0
    reset_tracker()                                   # 새 영상이니 추적 번호를 1부터 다시 매긴다
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if writer is None:                        # 첫 프레임을 읽어 보고 실제 크기로 저장 파일을 연다
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                if not writer.isOpened():
                    raise RuntimeError("결과 영상을 저장할 수 없습니다.")
            detected = detect_faces(frame, track=True)
            boxes = filter_excluded(frame, detected, ref_embs, cache=cache) if filter_excluded else list(detected)
            present = {b.get("track_id") for b in detected}
            for b in boxes:
                tid = b.get("track_id")
                if tid is not None:
                    held[tid] = (b, index)
                    covered.add(tid)
            for tid in [t for t, (_, last) in held.items() if index - last > HOLD_FRAMES]:
                del held[tid]
            boxes = boxes + [b for tid, (b, last) in held.items() if tid not in present and last < index]
            if effect == EFFECT_BLUR:
                result = apply_blur(frame, boxes)
            else:
                from src.mosaic import apply_mosaic

                result = apply_mosaic(frame, boxes)
            writer.write(result)
            index += 1
            if on_progress:
                on_progress(min(1.0, index / total))
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        reset_tracker()                               # 다음 영상 처리에 이 추적 상태가 섞이지 않게
    if index == 0:
        raise ValueError("영상에서 프레임을 읽을 수 없습니다.")
    seconds = time.perf_counter() - started
    return {"faces": len(covered), "frames": index, "fps": index / seconds if seconds > 0 else 0.0, "seconds": seconds}


def on_video(
    video_path: str,
    ref_embs: Optional[np.ndarray],
    on_progress: Optional[Callable[[float], None]] = None,
    effect: str = EFFECT_MOSAIC,
    on_wait: Optional[Callable[[int], None]] = None,
) -> tuple[str, dict]:
    """영상의 얼굴을 모자이크(또는 블러)해 (결과 영상 경로, 정보 dict)를 돌려준다.

    on_progress 는 진행률(0.0~1.0)을 받는 함수다. 정보 dict 는 process_video 가 만든
    {"faces": 등장 인원, "frames": 프레임 수, "fps": 처리 속도, "seconds": 걸린 시간}.
    on_wait 는 대기열에서 기다리는 동안 대기 순번을 받는 함수다.
    B 의 src/video.py 가 있으면 그것을 쓰고, 아직 없으면 같은 부품으로 만든 process_video_demo 로 처리한다.
    """
    # ┌──────────────────────────────────────────────────────┐
    # │ [B파트 모듈] src/video.py : process_video              │
    # │ (안에서 A의 detect_faces, C의 filter_excluded 를 부른다) │
    # └──────────────────────────────────────────────────────┘
    check_video_limits(video_path)

    extra_kwargs: dict = {}
    if module_exists("src.video"):
        from src.video import process_video

        if "effect" in inspect.signature(process_video).parameters:
            extra_kwargs["effect"] = effect
        elif effect != EFFECT_MOSAIC:    # 조용히 모자이크로 바꾸지 않고 알려 준다
            raise ValueError("영상 처리 모듈(process_video)이 아직 블러를 지원하지 않습니다. 모자이크를 고르거나, 조영준(B)에게 effect 인자 추가를 요청해 주세요.")
    else:
        process_video = process_video_demo
        extra_kwargs["effect"] = effect

    out_path = str(Path(video_path).with_name(Path(video_path).stem + "_out.mp4"))
    with processing_slot(on_wait):
        info = process_video(video_path, out_path, ref_embs=ref_embs, on_progress=on_progress, **extra_kwargs)
    return out_path, info


def on_stream(url: str, ref_embs: Optional[np.ndarray]) -> Iterator[tuple[np.ndarray, dict]]:
    """실시간 주소(RTSP·유튜브 라이브 등)를 모자이크하며 (RGB 프레임, 정보 dict)를 계속 내보낸다.

    정보 dict: {"faces_now": 현재 인원, "faces_total": 누적 인원, "fps": 처리 속도}
    끝이 없는 처리라 대기열에서 기다리지 않고, 자리가 없으면 바로 BusyError 를 낸다.
    """
    # ┌────────────────────────────────────────────┐
    # │ [B파트 모듈] src/stream.py : process_stream  │
    # └────────────────────────────────────────────┘
    from src.stream import process_stream

    with processing_slot(wait=False):
        for frame_bgr, info in process_stream(url, ref_embs=ref_embs):
            yield cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), info


# ============================================================
# 4. 메인 페이지
# ============================================================

def page_main_demo() -> None:
    """세로로 이어진 한 페이지: 메인 첫 화면 -> 인물 등록 -> 사진·영상 처리·성능 지표.

    처음에는 맨 위의 첫 화면만 보인다. 스크롤을 한 번 내리면 인물 등록 화면(인물 등록 + '등록 인물 제외'
    스위치)이, 한 번 더 내리면 "지금 바로 모자이크해 보세요" 구역(업로드 · 결과 · 성능 지표)이 나온다.
    (화면을 한 장씩 넘기는 동작은 PAGING_JS 가 한다)
    """
    # 첫 화면: 화면을 꽉 채우는 랜딩.
    render_landing(
        "Triple Y · Auto Face Mosaic",
        "얼굴은 가리고,<br><mark>순간</mark>은 남기다",
        "YOLO가 사진과 영상 속 얼굴을 알아서 찾아 모자이크해요. "
        "지우고 싶은 얼굴만 가리고, 남기고 싶은 사람은 그대로 둘 수 있어요.",
    )
    # 두 번째 화면: 인물 등록 (인물 등록 UI + '등록 인물 제외' 스위치)
    render_register_screen()
    # 세 번째 화면: 업로드 · 결과 · 성능 지표
    render_section_title(
        "지금 바로 모자이크해 보세요",
        "사진(jpg, png)이나 동영상(mp4)을 올리면 얼굴을 자동으로 찾아 모자이크합니다. "
        "등록한 인물은 제외할 수 있어요.",
    )

    render_file_tab()
    render_metrics_card()
    render_privacy_note()


# 화면 넘기기 스크립트: Streamlit 위쪽 문서(parent)에 한 번만 심는다. (components.html 의 iframe 은 화면을 다시 그릴 때
# 사라지므로, iframe 안에 두면 이벤트가 끊긴다.)
# 세로로 이어진 세 화면 [메인 첫 화면] -> [인물 등록] -> [업로드 구역] 을 스크롤 한 번에 한 화면씩 부드럽게 넘긴다.
# 업로드 구역에 들어간 뒤에는 평소처럼 스크롤되고, 그 구역의 맨 위에서 위로 올리면 다시 한 화면씩 돌아간다.
# 하단 바의 메뉴(메인 · 인물 등록)는 페이지를 바꾸지 않고 해당 화면으로 스크롤해 준다.
PAGING_JS = """
(function () {
  if (window.__tyPaging) return;
  window.__tyPaging = true;
  var LOCK_MS = __LOCK__, SWIPE = __SWIPE__, TOL = 12, GAP_MS = 140;
  var locked = false, lastWheel = 0, startY = null;

  function main() { return document.querySelector('[data-testid="stMain"]'); }
  function relTop(m, el) { return el.getBoundingClientRect().top - m.getBoundingClientRect().top + m.scrollTop; }

  // 세 화면의 위치. B 는 등록 화면의 맨 위, E 는 등록 화면의 맨 아래가 화면 아래쪽에 닿는 위치다.
  // (얼굴 후보가 나와서 등록 화면이 한 화면보다 길어지면 E > B 이고, 그 사이는 평소처럼 스크롤한다)
  //   [0] 메인 첫 화면 · [1] 등록 화면 맨 위 · [2] 등록 화면 맨 아래 · [3] 업로드 구역(제목이 상단 바 바로 밑에 오도록)
  function anchors(m) {
    var second = document.querySelector('[class*="st-key-screen_register"]'), head = document.querySelector('.section-head');
    if (!second || !head) return null;
    var b = relTop(m, second), e = Math.max(b, b + second.offsetHeight - m.clientHeight);
    return [0, b, e, Math.max(e, relTop(m, head) - 56)];
  }

  // dir > 0 아래로, dir < 0 위로 한 화면 넘길 때의 목적지. 평소처럼 스크롤해야 하면 null.
  function target(m, dir) {
    var a = anchors(m);
    if (!a) return null;
    var st = m.scrollTop, B = a[1], E = a[2], C = a[3];
    if (dir > 0) {
      if (st < B - TOL) return B;                              // 첫 화면 -> 등록 화면
      if (st < E - TOL) return null;                           // 긴 등록 화면 안: 평소 스크롤
      if (st < C - TOL) return C;                              // 등록 화면 끝 -> 업로드 구역
      return null;                                             // 업로드 구역 안: 평소 스크롤
    }
    if (st > C + TOL || st <= TOL) return null;                // 업로드 구역 안이거나 맨 위
    if (st > E + TOL) return E;                                // 업로드 구역 맨 위 -> 등록 화면 끝
    if (st > B + TOL) return null;                             // 긴 등록 화면 안: 평소 스크롤
    return 0;                                                  // 등록 화면 맨 위 -> 첫 화면
  }

  function page(m, to) {
    locked = true;
    m.scrollTo({ top: to, behavior: 'smooth' });
    setTimeout(function () { locked = false; }, LOCK_MS);
  }

  // 지금 화면 가운데에 걸쳐 있는 화면에 따라 <html data-tone="main|register|work"> 를 바꾼다. (첫 화면 / 인물 등록 / 업로드 구역)
  // CSS 가 상단 바 · 배경 · 하단 메뉴의 색을 이 값으로 바꾼다: 첫 화면은 민트/하늘색, 인물 등록과 업로드 구역은 같은 분홍/연보라.
  function updateTone() {
    var m = main(), second = document.querySelector('[class*="st-key-screen_register"]');
    if (!m || !second) return;
    var mid = m.scrollTop + m.clientHeight / 2, top = relTop(m, second);
    var tone = mid < top ? 'main' : (mid < top + second.offsetHeight ? 'register' : 'work');
    if (document.documentElement.getAttribute('data-tone') !== tone) document.documentElement.setAttribute('data-tone', tone);
  }
  document.addEventListener('scroll', function () { requestAnimationFrame(updateTone); }, true);
  setInterval(updateTone, 500);    // 화면이 다시 그려진 뒤에도 맞춘다
  updateTone();

  // 마우스 휠 · 트랙패드. 새 스크롤 동작(앞 동작과 GAP_MS 이상 떨어진 것)만 화면을 넘기고, 같은 동작의 관성은 무시한다.
  document.addEventListener('wheel', function (e) {
    var m = main();
    if (!m || e.ctrlKey || e.deltaY === 0) return;
    var now = Date.now(), gap = now - lastWheel;
    lastWheel = now;
    var to = target(m, e.deltaY > 0 ? 1 : -1);
    if (to === null) return;
    e.preventDefault();
    if (!locked && gap > GAP_MS) page(m, to);
  }, { passive: false });

  // 터치: 손가락을 위로 SWIPE 픽셀 이상 밀면 다음 화면, 아래로 밀면 이전 화면
  document.addEventListener('touchstart', function (e) { startY = e.touches[0].clientY; }, { passive: true });
  document.addEventListener('touchmove', function (e) {
    var m = main();
    if (!m || startY === null) return;
    var y = e.touches[0].clientY, dy = startY - y;
    var to = target(m, dy > 0 ? 1 : -1);
    if (to === null) { startY = y; return; }                   // 평소 스크롤 구간이면 기준점만 따라간다
    e.preventDefault();
    if (!locked && Math.abs(dy) > SWIPE) { page(m, to); startY = null; }
  }, { passive: false });

  // 하단 바 메뉴: 라디오를 바꾸지 않고(= 화면을 다시 그리지 않고) 해당 화면으로 스크롤한다.
  document.addEventListener('click', function (e) {
    var label = e.target.closest && e.target.closest('[class*="st-key-bottom_bar"] label[data-baseweb="radio"]');
    if (!label) return;
    e.preventDefault(); e.stopPropagation();
    var m = main(), a = m && anchors(m);
    if (!a) return;
    var idx = Array.prototype.indexOf.call(label.parentNode.querySelectorAll('label[data-baseweb="radio"]'), label);
    page(m, idx === 0 ? a[0] : a[1]);                           // 메인 -> 첫 화면, 인물 등록 -> 등록 화면 맨 위
  }, true);
})();
"""


def request_scroll(target: str) -> None:
    """이번 화면을 다 그린 뒤 결과('result') 또는 인물 등록 카드('person')의 아래쪽이 보이도록 부드럽게 내려가 달라고 표시한다."""
    st.session_state["scroll_to"] = (target, time.time_ns())    # 같은 요청이 연달아도 스크립트가 다시 실행되도록 번호를 붙인다


def inject_scroll_script() -> None:
    """request_scroll 로 표시된 곳으로 내려가는 스크립트를 한 번 심는다. 카드 아래쪽이 이미 보이면 아무것도 안 한다."""
    request = st.session_state.pop("scroll_to", None)
    if not request:
        return
    target, nonce = request
    card = {"result": "st-key-card_result", "person": "st-key-card_person"}.get(target)
    if card is None:
        return
    components.html(
        f"<script>/* {nonce} */"
        "const doc = window.parent.document;"
        "setTimeout(() => {"
        f"  const el = doc.querySelector('[class*=\"{card}\"]'), main = doc.querySelector('[data-testid=\"stMain\"]');"
        "  if (!el || !main) return;"
        "  const over = el.getBoundingClientRect().bottom - (main.getBoundingClientRect().bottom - 96);"    # 하단 바에 가리지 않게 96px 위
        "  if (over > 0) main.scrollBy({ top: over, behavior: 'smooth' });"
        "}, 700);"    # 새 내용이 그려질 시간을 준다
        "</script>",
        height=0,
    )


def inject_paging_script() -> None:
    """화면 넘기기 스크립트를 화면에 심는다. (이미 심었으면 아무 일도 안 한다)"""
    code = PAGING_JS.replace("__LOCK__", str(PAGE_LOCK_MS)).replace("__SWIPE__", str(PAGE_SWIPE_THRESHOLD))
    components.html(
        "<script>"
        "const doc = window.parent.document;"
        "if (!doc.getElementById('ty-paging')) {"
        "  const s = doc.createElement('script'); s.id = 'ty-paging';"
        f"  s.textContent = {json.dumps(code)};"
        "  doc.head.appendChild(s);"
        "}"
        "</script>",
        height=0,
    )


def render_metrics_card() -> None:
    """[메인] 아래쪽 '성능 지표' 카드: 이 접속에서 처리한 기록의 요약 수치와 최근 기록 표를 보여 준다."""
    with st.container(key="card_metrics"):
        rows = st.session_state["metrics"]
        desc = "이 접속에서 처리한 사진·영상의 속도와 감지된 얼굴 수입니다."
        if not rows:    # 기록이 없어도 지표 칸은 보여 주고, 안내는 설명 바로 아래에 붙인다
            desc += "<br>아직 처리 기록이 없습니다. 사진이나 영상을 처리하면 여기에 쌓입니다."
        render_card_heading("성능 지표", desc)

        if rows:
            total_sec = sum(r["seconds"] for r in rows)
            avg_fps = sum(r["fps"] for r in rows) / len(rows)
            stats = [
                (f"{len(rows)}회", "처리 횟수"),
                (f"{total_sec / len(rows):.2f}초", "평균 소요 시간"),
                (f"{avg_fps:.1f}", "평균 FPS (사진은 장/초)"),
                (f"{sum(r['faces'] for r in rows)}개", "감지된 얼굴 합계"),
            ]
        else:
            stats = [("0회", "처리 횟수"), ("-", "평균 소요 시간"), ("-", "평균 FPS (사진은 장/초)"), ("0개", "감지된 얼굴 합계")]
        stat_html = "".join(
            f'<div class="stat"><div class="stat-num">{num}</div><div class="stat-label">{label}</div></div>'
            for num, label in stats
        )
        body_html = "".join(
            f"<tr><td>{html.escape(r['at'])}</td><td>{html.escape(r['kind'])}</td><td>{html.escape(r['name'])}</td>"
            f"<td>{r['seconds']:.2f}</td><td>{r['faces']}</td><td>{r['fps']:.1f}</td></tr>"
            for r in reversed(rows[-METRICS_SHOW:])
        ) or "<tr>" + "<td>-</td>" * 6 + "</tr>"
        st.markdown(
            f'<div class="stat-row">{stat_html}</div>'
            '<table class="perf-table"><thead><tr><th>시각</th><th>종류</th><th>파일</th>'
            "<th>소요(초)</th><th>얼굴</th><th>FPS</th></tr></thead>"
            f"<tbody>{body_html}</tbody></table>",
            unsafe_allow_html=True,
        )


def render_privacy_note() -> None:
    """[메인] 맨 아래에 개인정보 보호 안내 문구를 작은 회색 글씨로 보여 준다."""
    st.markdown(
        '<p class="privacy-note">'
        "업로드한 사진·영상은 모자이크 처리를 위해 서버의 임시 폴더에 저장되며, 이 서비스 밖으로 전송되지 않습니다. "
        "등록한 인물의 얼굴 정보는 접속하는 동안에만 메모리에 보관됩니다. "
        "다른 사람의 얼굴이 담긴 파일은 당사자의 동의를 받은 뒤 사용해 주세요."
        "</p>",
        unsafe_allow_html=True,
    )


def get_modes() -> list[str]:
    """처리 종류 버튼에 보여 줄 목록을 돌려준다. 실시간은 SHOW_STREAM_MODE 가 True 일 때만 넣는다."""
    return [MODE_IMAGE, MODE_VIDEO] + ([MODE_STREAM] if SHOW_STREAM_MODE else [])


def render_file_tab() -> None:
    """처리 종류 전환 버튼(+ 바로 아래 파일 선택) -> (파일을 올린 뒤에만) 인물 등록 카드 -> 원본·결과 비교 카드 순서로 보여 준다.

    파일 선택 칸은 항상 업로드 카드의 사진/영상 전환 버튼 바로 아래에 있다. 파일을 올리기 전에는 점선 상자로,
    올린 뒤에는 가운데의 작은 Browse files 버튼(과 파일 이름)으로 줄어든다.
    """
    mode = st.session_state.get("mode", MODE_IMAGE)    # 아래 라디오가 만들어지기 전에도 현재 값을 알아야 한다
    files = as_file_list(st.session_state.get(f"uploader_{mode}"))
    has_files = bool(files)
    signature = tuple(f.file_id for f in files)
    if has_files and signature != st.session_state.get("files_signature"):    # 새 파일을 올렸다
        reset_file_state()    # 이전 파일의 후보 · 처리 결과는 비운다
        st.session_state["keep_question"] = bool(st.session_state["ref_people"])    # 등록해 둔 인물이 있으면 유지할지 묻는다
        st.session_state["files_signature"] = signature
        request_scroll("result")    # 결과 카드 아래쪽으로 내려간다

    # 카드 2 (중앙): 종류 고르기 + 업로드. 종류를 바꾸면 그 종류에 맞는 업로드 칸이 나온다.
    with st.container(key="card_upload"):
        render_card_heading("파일 업로드", "처리할 종류를 고르고 파일을 올려 주세요.")
        with st.container(key="mode_picker"):
            mode = st.radio("처리 종류", get_modes(), key="mode", horizontal=True, label_visibility="collapsed")

        if mode == MODE_STREAM:
            render_stream()
            return

        with st.container(key="result_uploader" if has_files else "upload_full"):    # 올린 뒤에는 CSS 가 작은 버튼으로 줄인다
            render_uploader(mode)
        if not has_files:
            st.caption("파일을 올리면 아래에 원본과 모자이크 결과가 나란히 표시됩니다.")
    if not has_files:    # 파일을 고르기 전에는 '인물 등록' 박스를 아예 보이지 않는다 (후보를 찾을 파일이 없으니 쓸 수 없다)
        return
    render_person_card(mode, files)    # 업로드 박스 바로 아래: 인물 관련 UI 는 이 카드 하나에 모여 있다

    # 카드 3 (하단): 원본과 결과 비교.
    with st.container(key="card_result"):
        render_result_card(mode, files)


def render_uploader(mode: str) -> None:
    """종류(사진·영상)에 맞는 파일 선택 칸을 그린다. 사진은 여러 장을 한 번에 올릴 수 있다.

    올린 파일은 반환값 대신 st.session_state["uploader_종류"] 에서 꺼내 쓴다.
    """
    is_image = mode == MODE_IMAGE
    st.file_uploader(
        "사진을 선택하세요" if is_image else "영상을 선택하세요",
        type=IMAGE_TYPES if is_image else VIDEO_TYPES,
        accept_multiple_files=is_image,
        key=f"uploader_{mode}",    # 종류마다 다른 이름표를 써서, 종류를 바꾸면 이전 파일이 남지 않게 한다
        help=f"이미지: jpg, png (최대 {MAX_IMAGE_FILES}장)" if is_image else "영상: mp4 (1분·720p 이하)",
    )


def image_result_key(uploaded: UploadedFile) -> str:
    """사진 결과를 찾는 열쇠. 파일이나 제외 인물 설정이 바뀌면 다른 결과로 본다. (가리는 방식은 바로 바꿀 수 있어 뺀다)"""
    return f"{uploaded.file_id}|{ref_tag()}"


def video_result_key(uploaded: UploadedFile) -> str:
    """영상 결과를 찾는 열쇠. 파일이나 제외 인물 설정, 가리는 방식이 바뀌면 다른 결과로 본다."""
    return f"{uploaded.file_id}|{ref_tag()}|{st.session_state.get('effect', EFFECT_MOSAIC)}"


def picker_key(entry: dict) -> str:
    """사진 한 장의 얼굴 선택 컴포넌트 이름표. 새로 처리할 때마다(run) 바뀌어서 컴포넌트가 새로 만들어진다."""
    return f"picker_{entry['run']}_{entry['fid']}"


def build_results_zip(rows: list[dict]) -> bytes:
    """결과 사진 파일들을 ZIP 하나로 묶은 바이트를 돌려준다. (PNG 는 이미 압축돼 있어 다시 압축하지 않는다)"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for row in rows:
            zf.write(row["path"], row["name"])
    return buf.getvalue()


@contextmanager
def job_errors(ui: JobUI) -> Iterator[None]:
    """처리 중 흔한 오류를 화면 안내로 바꿔 준다. 모듈·패키지가 없을 때 · 길이/해상도 초과 · 다른 처리 중 · 파일·모델 오류."""
    try:
        yield
    except ImportError as err:
        ui.clear()
        show_module_missing(err)
    except (ValueError, BusyError) as err:
        ui.clear()
        st.error(str(err))
    except (OSError, RuntimeError) as err:    # 모델 내려받기 실패, 파일 읽기 실패, 인식 엔진 오류 등
        ui.clear()
        st.error(f"처리 중 오류가 났습니다: {err}")


def render_result_card(mode: str, files: list[UploadedFile]) -> None:
    """비교 카드 전체: 머리줄(제목 · 적용) -> 파일 선택·옵션 -> 처리 상태 -> 원본·결과(결과 바로 아래에 저장 버튼).

    화면 위치는 아래 네 상자를 먼저 만들어 정해 두고, 실제 내용은 처리가 끝난 뒤에 채운다.
    그래야 '적용' 을 누른 바로 그 화면에서 결과와 저장 버튼이 나타난다.
    """
    is_image = mode == MODE_IMAGE
    if is_image and len(files) > MAX_IMAGE_FILES:
        st.warning(f"사진은 한 번에 {MAX_IMAGE_FILES}장까지 처리합니다. 앞의 {MAX_IMAGE_FILES}장만 사용합니다.")
        files = files[:MAX_IMAGE_FILES]
    effect = st.session_state.get("effect", EFFECT_MOSAIC)
    apply_key = "run_image" if is_image else "run_video"
    apply_clicked = bool(st.session_state.get(apply_key, False))    # 버튼은 아래에서 만들지만 눌렸는지는 지금 알 수 있다

    head = st.container(key="result_head")
    options = st.container()
    job_area = st.container()
    body = st.container()

    if st.session_state.pop("job_cancelled", False):
        with job_area:
            st.info("처리를 취소했습니다. 다시 시작하려면 적용 버튼을 눌러 주세요.")
    ui = JobUI(job_area, STEPS_IMAGE if is_image else STEPS_VIDEO)

    entries: list[tuple[UploadedFile, np.ndarray]] = []
    if is_image:
        with body:
            entries = decode_entries(files)

    # 1) 적용 버튼을 눌렀다면 처리한다. (대기열 순서를 기다리고, 취소 버튼도 여기서 보인다)
    if apply_clicked:
        with ui.area, job_errors(ui):
            if is_image:
                if entries:
                    run_image_batch(entries, effect, ui)
            else:
                run_video(files[0], ui)
        request_scroll("result")    # 처리가 끝났으니 결과(원본·결과 · 저장 버튼)가 보이는 아래쪽으로 내려간다

    # 2) 얼굴 선택 컴포넌트에서 고친 내용이나 가리는 방식이 바뀌었다면 결과를 다시 만든다. (얼굴을 다시 찾지는 않는다)
    if is_image:
        with ui.area, job_errors(ui):
            sync_image_edits(entries, effect)

    # 3) 머리줄: 제목. (파일 선택은 위 업로드 카드, 적용·저장 버튼은 원본·결과 바로 아래에 있다)
    with head:
        render_card_heading("원본과 결과 비교", f"왼쪽이 원본, 오른쪽이 {effect} 결과입니다.")

    # 4) 가리는 방식
    with options:
        with st.container(key="mode_picker_effect"):    # 사진/영상 전환 버튼과 같은 알약 모양
            st.radio("가리는 방식", EFFECTS, key="effect", horizontal=True, label_visibility="collapsed")

    # 5) 원본과 결과 (아래에 적용 버튼)
    with body:
        if is_image:
            render_image_rows(entries, effect, apply_key)
        else:
            render_video_rows(files[0], ui, effect, apply_key)


# ------------------------------------------------------------
# 사진
# ------------------------------------------------------------

def decode_entries(files: list[UploadedFile]) -> list[tuple[UploadedFile, np.ndarray]]:
    """올린 사진들을 RGB 이미지로 읽는다. 읽을 수 없는 파일은 오류 안내를 보이고 뺀다."""
    entries: list[tuple[UploadedFile, np.ndarray]] = []
    for uploaded in files:
        try:
            entries.append((uploaded, decode_image_rgb(uploaded)))
        except ValueError as err:
            st.error(f"{uploaded.name}: {err}")
    return entries


def default_edit(detected: list[dict]) -> dict:
    """처음 상태의 편집 내용: 등록 인물로 판정된 얼굴만 제외하고, 직접 영역은 없다."""
    return {"excluded": [i for i, box in enumerate(detected) if box["auto_excluded"]], "manual": []}


def resolve_boxes(detected: list[dict], edit: dict, width: int, height: int) -> list[dict]:
    """탐지 박스 중 제외하지 않은 것 + 직접 지정한 영역을 합쳐, 실제로 가릴 박스 목록(원본 픽셀)을 만든다."""
    skipped = set(edit["excluded"])
    boxes = [{"xyxy": b["xyxy"], "conf": b["conf"], "track_id": None} for i, b in enumerate(detected) if i not in skipped]
    for x1, y1, x2, y2 in edit["manual"]:
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
        if x2 - x1 >= 2 and y2 - y1 >= 2:
            boxes.append({"xyxy": (int(x1), int(y1), int(x2), int(y2)), "conf": 1.0, "track_id": None})
    return boxes


def write_result(result_rgb: np.ndarray, base: str, effect: str) -> dict:
    """결과 사진을 이 접속의 임시 폴더(results)에 PNG 로 저장하고 파일 정보 dict 를 돌려준다."""
    out_dir = get_workdir() / "results"
    out_dir.mkdir(exist_ok=True)
    name = f"{base}_{'blur' if effect == EFFECT_BLUR else 'mosaic'}.png"
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("결과 이미지를 저장할 수 없습니다.")
    path = out_dir / name
    path.write_bytes(encoded.tobytes())
    h, w = result_rgb.shape[:2]
    return {"name": name, "path": str(path), "size": (w, h), "bytes": path.stat().st_size}


def load_result_rgb(path: str) -> Optional[np.ndarray]:
    """저장해 둔 결과 PNG 를 RGB 이미지로 읽는다. 파일이 없으면 None."""
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    return None if img_bgr is None else cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def run_image_batch(entries: list[tuple[UploadedFile, np.ndarray]], effect: str, ui: JobUI) -> None:
    """올린 사진들을 대기열 순서로 처리한다. 얼굴을 찾아 가리고, 결과와 탐지 박스를 session_state 에 모은다."""
    ref_embs = active_ref()
    results: dict = {}
    used_bases: set = set()
    st.session_state["run_counter"] += 1
    run_id = st.session_state["run_counter"]
    total = len(entries)

    ui.steps(-1, "차례를 확인하는 중...")
    ui.show_cancel()
    with processing_slot(ui.waiting):
        for i, (uploaded, image_rgb) in enumerate(entries):
            note = f"{i + 1} / {total}번째 사진 · {uploaded.name}" if total > 1 else uploaded.name
            started = time.perf_counter()
            h, w = image_rgb.shape[:2]

            ui.steps(1, note)                                     # 파일 확인은 사진을 읽을 때 끝났다
            detected = on_detect(image_rgb, ref_embs)
            edit = default_edit(detected)
            boxes = resolve_boxes(detected, edit, w, h)
            ui.steps(2, note)
            result_rgb = on_mask(image_rgb, boxes, effect)
            ui.steps(3, note)

            stem = Path(uploaded.name).stem
            base, count = stem, 1
            while base in used_bases:                             # 같은 이름의 사진이 여러 장이어도 덮어쓰지 않는다
                count += 1
                base = f"{stem}_{count}"
            used_bases.add(base)

            entry = {
                "fid": uploaded.file_id, "run": run_id, "base": base, "effect": effect,
                "detected": detected, "init_edit": edit, "edit": edit, "faces": len(boxes), "src_size": (w, h),
            }
            entry.update(write_result(result_rgb, base, effect))
            results[image_result_key(uploaded)] = entry
            record_metric("사진", uploaded.name, time.perf_counter() - started, len(boxes))
    ui.finish(f"{total}장 처리 완료")
    st.session_state["image_results"] = results    # 이전 결과는 버리고 이번 결과만 남긴다


def sync_image_edits(entries: list[tuple[UploadedFile, np.ndarray]], effect: str) -> None:
    """얼굴 선택 컴포넌트의 현재 값과 가리는 방식을 결과에 반영한다. 바뀐 사진만 다시 가린다."""
    results = st.session_state["image_results"]
    for uploaded, image_rgb in entries:
        entry = results.get(image_result_key(uploaded))
        if entry is None:
            continue
        edit = parse_edit(st.session_state.get(picker_key(entry)), entry["edit"],
                          set(range(len(entry["detected"]))), entry["src_size"])
        if edit == entry["edit"] and effect == entry["effect"]:
            continue
        h, w = image_rgb.shape[:2]
        boxes = resolve_boxes(entry["detected"], edit, w, h)
        entry.update(write_result(on_mask(image_rgb, boxes, effect), entry["base"], effect))
        entry.update(edit=edit, effect=effect, faces=len(boxes))


def render_apply_row(apply_key: str, effect: str, render_save: Optional[Callable[[], None]] = None) -> None:
    """원본·결과 바로 아래 한 줄: 왼쪽 칸(원본 아래)에 '적용' 버튼, 오른쪽 칸(결과 아래)에 저장 버튼을 놓는다.

    저장 버튼은 render_save 가 그린다(결과가 없으면 그리지 않아 오른쪽 칸이 빈다). 누른 값은 render_result_card 가 session_state 로 미리 읽는다.
    """
    with st.container(key="apply_row"):
        col_apply, col_save = st.columns(2)
        with col_apply:
            st.button(f"{effect} 적용", key=apply_key)
        with col_save:
            if render_save is not None:
                render_save()


def render_image_rows(entries: list[tuple[UploadedFile, np.ndarray]], effect: str, apply_key: str) -> None:
    """사진마다 원본(왼쪽)과 결과(오른쪽)를 나란히 놓는다. 처리한 뒤에는 원본 쪽이 얼굴을 고르는 화면이 된다."""
    results = st.session_state["image_results"]
    info_lines: list[str] = []
    saved_rows: list[dict] = []
    result_dir: Optional[Path] = None
    for idx, (uploaded, image_rgb) in enumerate(entries):
        entry = results.get(image_result_key(uploaded))
        h, w = image_rgb.shape[:2]
        col_orig, col_result = st.columns(2)
        with col_orig:
            if entry is None:
                st.image(fit_to_canvas(image_rgb), use_container_width=True)
            else:
                # 컴포넌트에 넘기는 값은 처음 상태(init_edit)로 고정한다. 값이 바뀌면 Streamlit 이 새 컴포넌트로 보기 때문이다.
                face_picker_boxes(
                    key=picker_key(entry), src=preview_data_url(uploaded.file_id, image_rgb), image_shape=(h, w),
                    detected=entry["detected"], excluded=entry["init_edit"]["excluded"], manual=[],
                    reset_excluded=entry["init_edit"]["excluded"], token=picker_key(entry),
                )
        result_rgb = load_result_rgb(entry["path"]) if entry else None
        has_result = entry is not None and result_rgb is not None
        if has_result:
            with col_result:
                st.image(fit_to_canvas(result_rgb), use_container_width=True)
        # 이름표는 두 사진 아래에 같은 높이로 한 줄 (왼쪽 사진 아래에는 초기화 버튼이 더 있어서 사진 안에 두면 높이가 어긋난다)
        with st.container(key=f"cap_row_{idx}"):
            cap_orig, cap_result = st.columns(2)
            with cap_orig:
                st.caption(f"원본 · {uploaded.name}" if len(entries) > 1 else "원본")
            with cap_result:
                if has_result:
                    st.caption(f"{entry['effect']} 결과 (얼굴 {entry['faces']}개)")
        if not has_result:
            continue
        if len(entries) > 1:    # 여러 장이면 사진마다 이름표 아래에 저장 버튼을 둔다 (한 장이면 아래 버튼 줄의 오른쪽에 둔다)
            with st.container(key=f"save_row_{len(saved_rows)}"):
                st.download_button("결과 이미지 저장", data=Path(entry["path"]).read_bytes(), file_name=entry["name"],
                                   mime="image/png", key=f"save_{entry['run']}_{entry['fid']}")
        saved_rows.append(entry)

        rw, rh = entry["size"]
        info_lines.append(
            f"<code>{html.escape(entry['name'])}</code> · {rw}×{rh} · {format_size(entry['bytes'])} · 가린 얼굴 {entry['faces']}개"
        )
        result_dir = Path(entry["path"]).parent

    def render_save() -> None:    # 버튼 줄의 오른쪽: 한 장이면 그 결과 저장, 여러 장이면 한꺼번에 받는 ZIP
        if len(entries) == 1 and len(saved_rows) == 1:
            only = saved_rows[0]
            st.download_button("결과 이미지 저장", data=Path(only["path"]).read_bytes(), file_name=only["name"],
                               mime="image/png", key=f"save_{only['run']}_{only['fid']}")
        elif len(saved_rows) > 1:
            st.download_button("전체 저장 (ZIP)", data=build_results_zip(saved_rows), file_name="triple_y_results.zip",
                               mime="application/zip", key="save_zip")

    render_apply_row(apply_key, effect, render_save)
    if info_lines and result_dir is not None:
        render_result_info(info_lines, result_dir)


# ------------------------------------------------------------
# 영상
# ------------------------------------------------------------

def run_video(uploaded: UploadedFile, ui: JobUI) -> None:
    """영상을 on_video 로 처리하면서 진행률 막대와 단계 표시줄을 갱신하고, 결과를 session_state 에 남긴다.

    process_video 는 진행률만 알려 주므로, 진행률이 1.0 이 되면 소리 합치기 단계로 보고, 처리가 끝난 뒤
    ensure_audio 로 원본 소리가 결과에 들어 있는지 확인하고 빠졌으면 합쳐 넣는다.
    """
    note = uploaded.name
    in_path = save_upload_to_temp(uploaded, "input")
    effect = st.session_state.get("effect", EFFECT_MOSAIC)
    ui.steps(-1, "차례를 확인하는 중...")
    ui.show_cancel()
    bar = st.progress(0.0, text="대기 중...")
    stage = {"now": -1}
    started = time.perf_counter()

    def on_progress(fraction: float) -> None:
        fraction = min(max(fraction, 0.0), 1.0)
        bar.progress(fraction, text=f"처리 중... {fraction:.0%}")
        new_stage = 2 if fraction >= 1.0 else 1
        if new_stage != stage["now"]:    # 프레임마다 표시줄을 다시 그리면 느려지므로 단계가 바뀔 때만
            stage["now"] = new_stage
            ui.steps(new_stage, note)

    try:
        out_path, info = on_video(in_path, active_ref(), on_progress=on_progress, effect=effect, on_wait=ui.waiting)
    finally:
        bar.empty()    # 성공·실패·취소와 상관없이 진행률 막대를 지운다
    elapsed = time.perf_counter() - started
    ui.steps(2, note)                                  # 소리 합치기: 원본 소리가 빠졌다면 여기서 넣는다
    codec = ensure_h264(out_path)                      # 먼저 H.264 로 맞춘다 (소리 합치기는 영상을 그대로 복사하기 때문)
    audio = ensure_audio(in_path, out_path)
    st.session_state["video_result"] = {
        "key": video_result_key(uploaded), "path": out_path, "info": info, "audio": audio, "codec": codec, "effect": effect,
    }
    record_metric("영상", uploaded.name, float(info.get("seconds", elapsed)), int(info.get("faces", 0)),
                  frames=int(info.get("frames", 1)), fps=info.get("fps"))
    ui.finish("처리 완료")


def render_video_rows(uploaded: UploadedFile, ui: JobUI, effect: str, apply_key: str) -> None:
    """업로드한 영상을 재생해 보여 주고 처리 결과를 보여 준다."""
    # 원본(왼쪽)과 결과(오른쪽)를 나란히 놓는다. 결과 칸은 처리가 끝난 뒤에 채워진다.
    res = st.session_state["video_result"]
    has_result = res is not None and res["key"] == video_result_key(uploaded) and Path(res["path"]).exists()
    col_orig, col_result = st.columns(2)
    with col_orig:
        st.video(uploaded)    # 재생 버튼이 달린 플레이어
    with col_result:
        if has_result:
            st.video(res["path"])
    with st.container(key="cap_row_video"):    # 두 이름표를 같은 높이에 한 줄로
        cap_orig, cap_result = st.columns(2)
        with cap_orig:
            st.caption("원본")
        with cap_result:
            if has_result:
                st.caption(f"{res.get('effect', EFFECT_MOSAIC)} 결과")

    def render_save() -> None:    # 버튼 줄의 오른쪽 칸 (처리한 결과가 있을 때만)
        if has_result:
            st.download_button("결과 영상 저장", data=Path(res["path"]).read_bytes(), file_name=f"{Path(uploaded.name).stem}_{'blur' if res.get('effect') == EFFECT_BLUR else 'mosaic'}.mp4",
                               mime="video/mp4", key="save_video")

    render_apply_row(apply_key, effect, render_save)

    if has_result:
        info, (audio_text, audio_warn) = res["info"], res["audio"]
        st.caption(
            f"등장 인원 {info['faces']}명 · 프레임 {info['frames']}개 · "
            f"{info['fps']:.1f} FPS · {info['seconds']:.1f}초"
        )
        codec_text, codec_warn = res.get("codec", ("", False))
        if codec_warn:
            st.warning(codec_text)
        if audio_warn:
            st.warning(audio_text)
        out = Path(res["path"])
        lines = [f"<code>{html.escape(out.name)}</code> · {format_size(out.stat().st_size)}"]
        if codec_text and not codec_warn:
            lines.append(html.escape(codec_text))
        if not audio_warn:
            lines.append(html.escape(audio_text))
        render_result_info(lines, out.parent)


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
# 5. 인물 등록 (후보 찾기 · 여러 명 등록 · 해제)
# ============================================================

def sync_ref_embs() -> None:
    """등록한 사람들(ref_people)의 얼굴 벡터를 한데 모아 처리에 쓸 ref_embs 로 만든다. 등록이 없으면 None.

    등록·해제할 때마다 새 배열이 만들어져서, 앞서 처리한 결과가 '이전 설정의 결과'로 구분된다(ref_tag).
    """
    people = st.session_state["ref_people"]
    st.session_state["ref_embs"] = np.vstack([p["embs"] for p in people.values()]) if people else None


def toggle_candidate(cand_key: tuple, embs: np.ndarray, thumb_bgr: np.ndarray) -> None:
    """후보의 '등록' 태그 버튼 콜백. 꺼져 있으면 이 사람을 등록하고(켜짐), 이미 등록된 사람이면 그 사람만 해제한다(꺼짐).

    사람마다 따로 켜고 끌 수 있어서 여러 명을 등록할 수 있다. 등록한 사람은 모두 모자이크에서 빠진다.
    """
    people = st.session_state["ref_people"]
    if cand_key in people:
        del people[cand_key]
    else:
        people[cand_key] = {"embs": embs, "thumb": cv2.cvtColor(thumb_bgr, cv2.COLOR_BGR2RGB)}
    st.session_state["keep_question"] = False    # 직접 등록·해제하고 있으니 더 묻지 않는다
    sync_ref_embs()


def clear_registration() -> None:
    """'등록 해제' 버튼의 콜백. 등록한 사람을 모두 지운다."""
    st.session_state["ref_people"] = {}
    st.session_state["keep_question"] = False    # 지울 인물이 없으면 물을 것도 없다
    sync_ref_embs()


def reset_file_state() -> None:
    """새 파일을 올렸을 때 이전 파일에 묶인 상태를 비운다: 얼굴 후보 · 이전 처리 결과. (등록한 인물과 처리 기록은 남긴다)"""
    st.session_state["candidates"] = None
    st.session_state["candidates_for"] = None
    st.session_state["image_results"] = {}
    st.session_state["video_result"] = None


def answer_keep_people() -> None:
    """새 파일 질문의 [유지] 콜백. 등록해 둔 인물을 그대로 두고 질문을 닫는다."""
    st.session_state["keep_question"] = False


def answer_reset_people() -> None:
    """새 파일 질문의 [초기화] 콜백. 등록해 둔 인물을 모두 지우고 질문을 닫는다."""
    clear_registration()


def render_register_screen() -> None:
    """두 번째 화면 '인물 등록': 소개 글만 있는 화면. (색 톤이 분홍·연보라로 바뀌어 지금 이 화면에 있다는 걸 알려 준다)

    파일 업로드 박스도 카드도 두지 않는다. 인물 등록은 아래 업로드 구역의 '인물 등록' 카드 하나에 모두 모여 있다
    (render_person_card). 파일을 올린 바로 그 자리에서 이어서 등록하게 하려는 것이다.
    """
    with st.container(key="screen_register"):
        st.markdown(    # 장식 원 (CSS 가 화면 전체를 기준으로 놓는다)
            '<div class="blob b1"></div><div class="blob b2"></div><div class="blob b3"></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<span class="pill">Exclude Person</span>'
            '<div class="landing-title"><mark>인물</mark> 등록</div>'
            '<p class="landing-desc">영상이나 사진에서 모자이크 처리에서 제외할 인물을 등록하세요. '
            "파일을 업로드한 뒤, 감지된 얼굴 후보 중에서 제외할 사람을 선택할 수 있습니다.</p>"
            '<p class="landing-tip">정면 1장 + 좌우 측면 각 1장, 또는 고개를 천천히 돌리는 5초 영상으로 등록하면 더 정확해요.</p>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="scroll-hint"><span>스크롤하세요</span></div>', unsafe_allow_html=True)


def render_person_card(mode: str, files: list[UploadedFile]) -> None:
    """'인물 등록' 카드 하나에 인물 관련 UI 를 모두 모은다. 위에서부터: 제목줄(제목 + 오른쪽 끝 [모자이크 제외할 얼굴 선택]) -> 후보 격자 -> 등록된 사람들.

    [모자이크 제외할 얼굴 선택]은 카드 제목줄의 오른쪽 끝에 있다. 후보 사진은 CSS 격자(cgrid)로 놓아서 창이 좁아져도
    줄이 흐트러지지 않고, 등록된 사람들은 맨 아래의 한 줄(reg_row)에 모인다. 후보마다 '등록' 태그를 따로 켜고 끌 수 있어
    여러 명을 등록할 수 있고, 등록한 사람은 모두 모자이크에서 빠진다(별도의 '제외' 스위치는 없다).
    새 파일을 올렸을 때 등록해 둔 인물이 있으면 맨 위에서 [유지] / [초기화] 를 묻는다. 영상이면 [모자이크 제외할 얼굴 선택]가 영상 속
    얼굴 목록도 함께 만든다. 사진을 여러 장 올렸다면 '모든 사진'(같은 사람의 정면·측면 사진을 한 사람으로 묶는다)이나 한 장을
    고른다. (후보를 찾는 일은 C파트 src/exclude.py 의 list_candidates 가 한다)
    """
    with st.container(key="card_person"):
        with st.container(key="person_head"):    # 제목은 왼쪽, 버튼은 오른쪽 끝
            st.markdown('<div class="card-title">인물 등록</div>', unsafe_allow_html=True)
            find_clicked = st.button("모자이크 제외할 얼굴 선택", key="find_candidates")
        people = st.session_state["ref_people"]
        if people and st.session_state.get("keep_question"):    # 새 파일을 올렸고 등록해 둔 인물이 있으면 유지할지 묻는다
            with st.container(key="keep_ask"):
                st.markdown(f'<p class="keep-text">새 파일을 올렸어요. 등록해 둔 인물 {len(people)}명을 유지할까요?</p>', unsafe_allow_html=True)
                _, col_keep, col_reset, _ = st.columns([2.2, 1, 1, 2.2])    # 두 버튼을 가운데에 나란히
                with col_keep:
                    st.button("유지", key="keep_people", on_click=answer_keep_people)
                with col_reset:
                    st.button("초기화", key="reset_people", on_click=answer_reset_people)

        sources = list(files)    # 후보를 찾을 파일들
        if mode == MODE_IMAGE and len(files) > 1:
            names = [f"모든 사진 ({len(files)}장)"] + [f"{i + 1}. {f.name}" for i, f in enumerate(files)]
            picked = st.selectbox("후보를 찾을 사진", names, key="pick_source")
            if picked != names[0]:
                sources = [files[names.index(picked) - 1]]
        source_id = "|".join(f.file_id for f in sources)
        if st.session_state["candidates_for"] != source_id:    # 다른 파일로 바뀌면 이전 파일의 후보는 지운다
            st.session_state["candidates"] = None

        if find_clicked:
            try:
                with st.spinner("얼굴을 찾는 중..."):
                    st.session_state["candidates"] = on_register(sources)
                    st.session_state["candidates_for"] = source_id
            except ImportError as err:
                show_module_missing(err)
            except (ValueError, OSError, RuntimeError) as err:    # 파일을 읽을 수 없음 · 모델 내려받기 실패 · 인식 엔진 오류
                st.error(f"얼굴 후보를 찾지 못했습니다: {err}")
            request_scroll("person")    # 후보가 보이도록 이 카드의 아래쪽으로 내려간다
        render_candidates(source_id)

        if people:    # 등록된 사람들은 이 카드 맨 아래에만 나오고, 모두 해제하는 버튼도 여기 하나뿐이다
            with st.container(key="reg_row"):
                st.markdown(f'<span class="reg-tag on">✓ 등록됨({len(people)}명)</span>', unsafe_allow_html=True)
                with st.container(key="reg_thumbs"):
                    for person in people.values():
                        st.image(square_thumb(person["thumb"], REGISTERED_THUMB_WIDTH * 2), width=REGISTERED_THUMB_WIDTH)
                with st.container(key="clear_row"):
                    st.button("등록 해제", key="clear_ref", on_click=clear_registration)


def render_candidates(source_id: str) -> None:
    """찾은 얼굴 후보를 작은 사진으로 격자에 늘어놓고, 사진마다 '등록' 태그 버튼을 둔다. 등록된 후보는 '✓ 등록됨'(켜짐)으로 색이 확 바뀐다."""
    candidates = st.session_state["candidates"]
    if candidates is None:
        return                                  # 아직 찾기 전
    if not candidates:                          # 빈 리스트 = 얼굴을 못 찾음
        st.caption("찾은 얼굴이 없어요.")
        return

    people = st.session_state["ref_people"]
    with st.container(key="cgrid"):             # CSS 격자: 한 칸에 후보 하나 (사진 · 이름표 · 등록 태그). 창이 좁아지면 줄만 바뀐다
        for i, cand in enumerate(candidates):
            cand_key = (source_id, i)
            registered = cand_key in people
            with st.container(key=f"cand_{i}"):
                st.image(square_thumb(cv2.cvtColor(cand["thumb"], cv2.COLOR_BGR2RGB), CANDIDATE_THUMB_WIDTH * 2), width=CANDIDATE_THUMB_WIDTH)
                st.caption(f"후보 {i + 1} · {cand['frames']}회")
                with st.container(key=f"pick_{'on' if registered else 'off'}_{i}"):    # 켜짐/꺼짐에 따라 CSS 가 다른 색을 입힌다
                    st.button("✓ 등록됨" if registered else "등록", key=f"pick_{i}",
                              on_click=toggle_candidate, args=(cand_key, cand["embs"], cand["thumb"]))


# ============================================================
# 6. 시작점
# ============================================================

def render_bottom_bar() -> None:
    """화면 아래쪽에 떠 있는 메뉴 바를 그린다. 모양은 라디오 그대로이고, 누르면 PAGING_JS 가 해당 화면으로 스크롤해 준다."""
    with st.container(key="bottom_bar"):
        st.radio(
            "메뉴",
            [MENU_MAIN, MENU_REGISTER],
            key="menu",
            horizontal=True,
            label_visibility="collapsed",
        )


def main() -> None:
    """페이지 설정 -> 한 페이지 그리기 -> 하단 바 그리기 순서로 앱을 구성한다."""
    # set_page_config 는 다른 st 명령보다 먼저 한 번만 불러야 한다.
    st.set_page_config(
        page_title=f"Triple Y Mini · {APP_SUBTITLE}",
        page_icon=str(ROOT / "app" / "static" / "yyy_favicon.png"),   # 브라우저 탭 아이콘
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    init_session_state()
    inject_css()

    # 하단 바는 화면 맨 아래에 고정되어 있어서 코드에서는 페이지 뒤에 그린다.
    page_main_demo()
    render_bottom_bar()
    inject_paging_script()
    inject_scroll_script()


if __name__ == "__main__":
    main()
