# Triple Y Mini

YOLO를 활용한 얼굴 자동 모자이크 서비스 (3인, 2주).

사진·영상·실시간 영상 속 얼굴을 YOLO11로 자동 탐지해 모자이크하고, 등록한 특정 인물(본인 등)은 모자이크에서 제외하는 개인정보 보호 서비스입니다.

- 기획서: [기획서.md](기획서.md) ([PDF](기획서.pdf))
- 함수 입출력·데이터 타입 규칙: [역할별_구현명세.md](역할별_구현명세.md)

## 핵심 기술
YOLO11(ultralytics), ByteTrack, OpenCV, ffmpeg, yt-dlp, InsightFace, 웹 데모(Streamlit·Gradio·Vercel 중 선택)

## 폴더 구조
```
Triple Y Mini/
├─ src/          # 탐지·모자이크·영상·인물 제외 모듈
├─ app/          # 웹 데모
├─ data/
│  ├─ raw/        # 원본 데이터 (수정 금지, git 제외)
│  └─ processed/  # 전처리 결과 (git 제외)
├─ weights/      # 모델 가중치 (git 제외)
└─ tools/        # 보조 스크립트
```

## 시작하기
```powershell
git clone https://github.com/Leeyuno0419/Triple-Y-Mini.git
cd Triple-Y-Mini
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
- `requirements.txt`는 패키지를 추가할 때 버전과 함께 적는다.
- 영상 소리 병합에 ffmpeg가 필요하다. Colab·리눅스 서버는 기본 제공, Windows는 별도 설치 후 PATH에 등록한다.
- 비밀키·API 키는 프로젝트 루트의 `.env`에 둔다 (git 제외).
- 데이터와 가중치(`best.pt`)는 git에 올리지 않으므로 팀 공유 폴더(Drive·서버)에서 받는다.
- 기획서를 고친 뒤 PDF 재생성: `python tools\md_to_pdf.py 기획서.md 기획서.pdf` (reportlab, 맑은 고딕 글꼴 필요)

## 얼굴 탐지 모델 준비 (`detect_faces`)
지금은 사전학습 가중치 [yolov11n-face](https://github.com/akanametov/yolo-face) (WIDER FACE로 학습, GPL-3.0)를 `weights\best.pt`로 쓴다. 직접 학습을 하기로 해서 모델이 나오면 A가 `best.pt`만 교체한다.
```powershell
Invoke-WebRequest https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov11n-face.pt -OutFile weights\best.pt
```
Colab에서는 [tools/colab_detector.ipynb](tools/colab_detector.ipynb)를 열어 위에서부터 실행한다 (사진 탐지·영상 추적 확인 포함). 유튜브 주소와 구간을 넣어 탐지 결과 영상을 만들려면 [tools/youtube_detect.ipynb](tools/youtube_detect.ipynb)를 쓴다. 직접 입력할 때는:
```
!git clone https://github.com/Leeyuno0419/Triple-Y-Mini.git
%cd Triple-Y-Mini
!pip install -r requirements.txt
!wget -O weights/best.pt https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov11n-face.pt
```
사용 예 (프로젝트 루트에서 실행):
```python
import cv2
from src.detector import detect_faces, reset_tracker

img = cv2.imread("사진.jpg")               # BGR
boxes = detect_faces(img)                  # [{"xyxy": (x1, y1, x2, y2), "conf": 0.9, "track_id": None}, ...]
reset_tracker()                            # 영상마다 처음에 한 번
boxes = detect_faces(frame, track=True)    # 프레임마다, track_id가 붙음
```
- `src\` 안의 파일을 `python src\xxx.py`로 직접 실행하지 않는다. `src\types.py`가 표준 라이브러리 `types`와 이름이 겹친다.

## 협업 규칙
### 브랜치
| 브랜치 | 용도 | 누가 |
|---|---|---|
| `main` | 항상 돌아가는 상태만 | 직접 푸시 금지 |
| `feat/기능명` | 기능 개발 (문서·설정·리팩터링 포함) | 각자 |
| `fix/버그명` | 버그 수정 | 각자 |

- 이름은 소문자 영문과 하이픈만 쓴다. 예: `feat/detector`, `fix/mosaic-zero-box`
- 브랜치 하나에는 목적 하나만 담고, 끝나면 빨리 병합한다.

### 작업 흐름
```powershell
git switch main; git pull                  # 1. main 최신화
git switch -c feat/기능명                   # 2. 브랜치 생성
# ... 작업 후 커밋 ...
git push -u origin feat/기능명              # 3. 푸시
```
4. GitHub에서 PR을 만든다. 제목은 커밋 메시지와 같은 `접두사: 내용` 형식.
5. 팀원 1명 이상이 승인하면 `main`에 병합한다.
6. 병합이 끝난 브랜치는 삭제하고 `main`을 다시 pull한다.

### 커밋 메시지
`접두사: 내용` 형식. 접두사: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`.

### 코드 컨벤션
- PEP 8 준수, 함수·변수는 `snake_case`, 클래스는 `PascalCase`.
- 모든 함수에 타입힌트와 기능 설명 docstring을 쓴다.
- 숫자 설정(임계값 등)은 `src/config.py`에서만 바꾼다.

자세한 규칙은 [CLAUDE.md](CLAUDE.md)를 참고한다.

## 팀원
| 담당 | 이름 | 역할 |
|---|---|---|
| A | 이유노 | 모델·데이터·평가 |
| B | 조영준 | 모자이크·영상·실시간 |
| C | 나예린 | 웹·인물 제외 |
