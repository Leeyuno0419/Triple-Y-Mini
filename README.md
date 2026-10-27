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

## WIDER FACE 데이터 받기와 라벨 검수
| 데이터 | 주소 | 용도 |
|---|---|---|
| train 이미지 (1.4GB) | [Drive](https://drive.google.com/file/d/15hGDLhsx8bLgLcIRD5DhYt5iBxnjNF1M/view?usp=sharing) | 학습 · 검수 |
| val 이미지 (360MB) | [Drive](https://drive.google.com/file/d/1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q/view?usp=sharing) | 검증 · 검수 |
| test 이미지 (1.8GB) | [Drive](https://drive.google.com/file/d/1HIfDbVEWKmsYKJZm4lchTBDLW5N7dY5T/view?usp=sharing) | 정답 박스가 공개되지 않아 쓰지 않음 |
| 정답 박스 (annotations) | [wider_face_split.zip](http://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip) | train·val 정답 |

**검수 방법:** 위 [시작하기](#시작하기)로 clone과 `pip install`을 마친 뒤, 프로젝트 폴더에서 본인 이니셜로 실행한다.
```powershell
python tools\prepare_widerface.py cyj      # 이니셜: lyn(폴더 0~20), cyj(21~40), nyr(41~61)
```
1. train·val 이미지와 정답을 `data\raw\widerface\`에 받아 압축을 푼다 (이미 있으면 건너뜀, 디스크 약 4GB 필요).
2. train·val 정답을 YOLO 라벨로 바꾼다.
3. 본인이 맡은 범위에서 train·val 각 30장에 박스를 그려 `data\processed\widerface\review\<이니셜>\train`, `val`에 저장하고 그 폴더를 연다.

- 볼 것: ① 박스가 얼굴에서 밀리거나 크기가 엉뚱한 것 ② 박스가 빠진 얼굴 (WIDER FACE는 아주 작거나 심하게 가려진 얼굴 일부를 원래 라벨하지 않음) ③ 박스 범위(이마·턱·귀)가 팀 라벨링 기준과 같은지
- 이상한 이미지는 `train`/`val` 구분과 파일 이름, 무엇이 이상한지 팀에 공유한다.
- 다운로드가 실패하면 위 주소에서 브라우저로 받아 zip을 `data\raw\widerface\`에 넣고 다시 실행한다.
- 압축 풀기가 중간에 끊겼으면 `data\raw\widerface\` 안의 풀다 만 폴더(`WIDER_train` 등)를 지우고 다시 실행한다.
- 얼굴 데이터이므로 `data\`의 어떤 파일도 git에 올리지 않는다 (`.gitignore`로 제외됨).

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
