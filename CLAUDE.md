# Triple Y Mini 팀 프로젝트 지침

YOLO를 활용한 얼굴 자동 모자이크 서비스 (3인, 2주). 전역 지침(`~/.claude/CLAUDE.md`)에 더해 아래 규칙을 따른다.
기획은 `기획서.md`, 함수 입출력·데이터 타입 규칙은 `역할별_구현명세.md`가 기준이다.

## 폴더 구조
```
Triple Y Mini/
├─ src/
│  ├─ config.py, types.py        # 공통 (숫자 설정은 config.py에서만)
│  ├─ detector.py                # A 이유노
│  ├─ mosaic.py, video.py, stream.py  # B 조영준
│  └─ exclude.py                 # C 나예린
├─ app/          # 웹 데모 (C 나예린)
├─ data/
│  ├─ raw/        # 원본 데이터 (수정 금지, git 제외)
│  └─ processed/  # 전처리 결과 (git 제외)
├─ weights/      # 모델 가중치 best.pt 등 (git 제외)
├─ tools/        # 보조 스크립트 (md_to_pdf.py 등)
├─ 기획서.md, 기획서.pdf, 역할별_구현명세.md
├─ requirements.txt
└─ .env          # 비밀키 (git 제외)
```
- 원본 데이터(`data/raw/`)는 절대 덮어쓰거나 수정하지 않는다.
- 가중치(`weights/`, `*.pt`)와 학습 결과(`runs/`)는 git에 올리지 않고 Drive·서버로 공유한다.

## 모듈 인터페이스
- 함수 이름·입출력을 바꿀 때는 `역할별_구현명세.md`와 `기획서.md` 5장을 함께 고치고 팀에 공유한다.
- `img`는 BGR `uint8`, 박스 좌표는 픽셀 정수, 얼굴이 없으면 빈 리스트 `[]` (None 금지).
- 임계값 등 숫자는 `src/config.py`에서만 바꾸고 코드에 직접 쓰지 않는다.
- `기획서.md`를 고치면 `python tools\md_to_pdf.py 기획서.md 기획서.pdf`로 PDF도 다시 만든다.

## Git 협업 규칙
### 브랜치
| 브랜치 | 용도 | 누가 |
|---|---|---|
| `main` | 항상 돌아가는 상태만 | 직접 푸시 금지 |
| `feat/기능명` | 기능 개발 (문서·설정·리팩터링 포함) | 각자 |
| `fix/버그명` | 버그 수정 | 각자 |

- 이름은 소문자 영문과 하이픈만 쓴다. 예: `feat/detector`, `fix/mosaic-zero-box`
- 브랜치 하나에는 목적 하나만 담는다. 작업이 끝나면 빨리 병합하고 오래 끌지 않는다.

### 작업 흐름
1. `main` 최신화: `git switch main; git pull`
2. 브랜치 생성: `git switch -c feat/기능명`
3. 작업 후 커밋, 푸시: `git push -u origin feat/기능명`
4. GitHub에서 PR 생성. 제목은 커밋 메시지와 같은 `접두사: 내용` 형식.
5. 팀원 1명 이상 승인 후 `main`에 병합.
6. 병합이 끝난 브랜치는 삭제하고 `main`을 다시 pull한다.

### 충돌 줄이는 법
- 작업을 시작하기 전에 `main`을 pull하고, 긴 작업은 중간에 `main`을 반영한다.
- 담당 파일(위 폴더 구조)을 나눠 같은 파일을 동시에 수정하지 않는다. 공통 파일(`config.py`, `types.py`)은 고치기 전에 팀에 알린다.
- PR은 작게 자주 올린다.

### Claude 작업 시 규칙
- `main`에 직접 커밋·푸시하지 않는다. 항상 위 이름 규칙의 브랜치를 만들어 작업한다.
- 커밋·푸시·PR 생성은 사용자가 요청할 때만 한다.
- 커밋 메시지는 `접두사: 내용` 형식. 접두사: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`.

## 코드 컨벤션
- PEP 8 준수. 함수·변수는 `snake_case`, 클래스는 `PascalCase`.
- 모든 함수에 타입힌트를 쓰고, 기능 설명 docstring(한국어)을 붙인다.

## 데이터·AI 규칙
- 비밀키·API 키는 `.env`에만 두고 코드에 직접 쓰지 않는다.
- 팀원 얼굴 사진·영상(인물 제외 테스트용)은 본인 동의를 받고, git에 올리지 않는다.
- 패키지를 추가하면 `requirements.txt`에 반영하고 팀에 알린다. 설치·삭제는 승인 후 진행한다.
- 재현성: 랜덤 시드를 고정하고(`split_dataset` 기본 `seed=42`), 실험 파라미터와 결과를 기록한다.

## 역할
| 담당 | 이름 | 역할 | 담당 파일 |
|---|---|---|---|
| A | 이유노 | 모델·데이터·평가 | `src/detector.py`, 평가 스크립트 |
| B | 조영준 | 모자이크·영상·실시간 | `src/mosaic.py`, `src/video.py`, `src/stream.py` |
| C | 나예린 | 웹·인물 제외 | `src/exclude.py`, `app/` |
