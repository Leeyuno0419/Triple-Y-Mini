"""유튜브 라이브용 최신 yt-dlp 환경 만들기 (프로젝트 루트의 .venv-ytdlp).

yt-dlp 는 2024.10.22 를 끝으로 Python 3.8 을 지원하지 않고, 이 버전은 유튜브가 요청을 거절한다
(ios 클라이언트 'Precondition check failed' -> HLS 조각 HTTP 403 -> 앱의 '스트림 연결에 실패했습니다').
팀의 .venv 는 3.8 그대로 두고, Python 3.9 이상으로 만든 별도 환경에 최신 yt-dlp 를 설치한다.
src/stream.py 는 이 환경이 있으면 그 yt-dlp 로 주소를 해석하고, 없으면 기본 yt-dlp 로 시도한다.

사용법 (프로젝트 루트에서):
  python tools\\setup_ytdlp.py                                   # 3.9 이상 Python 을 찾아 쓴다
  python tools\\setup_ytdlp.py --python C:\\path\\to\\python3.11.exe   # 직접 지정
  python tools\\setup_ytdlp.py --update                           # 이미 만든 환경의 yt-dlp 만 최신으로

다른 위치의 환경을 쓰려면 환경 변수 YTDLP_PYTHON 에 그 환경의 python 경로를 둔다.
JavaScript 런타임(Node.js 또는 Deno)이 있으면 유튜브의 서명 확인을 풀 수 있다. (Node.js 가 PATH 에 있으면 자동으로 쓴다)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / ".venv-ytdlp"
PYTHON = VENV / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def python_version(python: str) -> tuple[int, int]:
    """해당 Python 의 (major, minor). 실행할 수 없으면 (0, 0)."""
    try:
        out = subprocess.run([python, "-c", "import sys; print(sys.version_info[0], sys.version_info[1])"],
                             capture_output=True, text=True, timeout=20).stdout.split()
        return int(out[0]), int(out[1])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return 0, 0


def find_python() -> str | None:
    """3.9 이상인 Python 을 찾는다: 지금 실행 중인 것 -> conda 환경들 -> PATH."""
    candidates = [sys.executable]
    for base in (Path.home() / "anaconda3", Path.home() / "miniconda3", Path.home() / "miniforge3"):
        candidates += [str(p) for p in sorted((base / "envs").glob("*/python.exe"))] + [str(p) for p in sorted((base / "envs").glob("*/bin/python"))]
    candidates += ["python3.12", "python3.11", "python3.10", "python3.9"]
    for candidate in candidates:
        if python_version(candidate) >= (3, 9):
            return candidate
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="유튜브 라이브용 최신 yt-dlp 환경을 만든다.")
    parser.add_argument("--python", help="환경을 만들 때 쓸 Python 3.9 이상 실행 파일")
    parser.add_argument("--update", action="store_true", help="이미 만든 환경의 yt-dlp 만 최신으로 올린다")
    args = parser.parse_args()

    if not args.update or not PYTHON.is_file():
        base = args.python or find_python()
        if base is None or python_version(base) < (3, 9):
            print("Python 3.9 이상을 찾지 못했습니다. --python 으로 경로를 알려 주세요.")
            return 2
        print(f"{VENV} 를 만듭니다 (Python {'.'.join(map(str, python_version(base)))}: {base})")
        subprocess.run([base, "-m", "venv", str(VENV)], check=True)
    subprocess.run([str(PYTHON), "-m", "pip", "install", "-U", "pip", "yt-dlp[default]"], check=True)
    version = subprocess.run([str(PYTHON), "-m", "yt_dlp", "--version"], capture_output=True, text=True).stdout.strip()
    print(f"완료: yt-dlp {version} ({PYTHON})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
