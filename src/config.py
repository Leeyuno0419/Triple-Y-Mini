"""공통 설정. 숫자는 이 파일에서만 바꾼다 (역할별_구현명세.md 0장)."""

WEIGHTS = "weights/best.pt"   # A가 교체 (프로젝트 루트 기준 경로)
CONF = 0.25                   # 탐지 임계값 (누락을 줄이려고 낮게 시작)
PAD = 0.15                    # 박스 여백 비율
BLOCKS = 6                    # 얼굴 긴 변을 몇 칸으로 나눌지 (모자이크 강도)
HOLD_FRAMES = 5               # 탐지가 끊겨도 박스를 유지할 프레임 수
UNTRACKED_HOLD_FRAMES = 2      # track_id가 없는 얼굴 박스를 잠깐 유지할 프레임 수
UNTRACKED_MATCH_IOU = 0.3      # track_id가 없는 박스를 같은 얼굴로 연결하는 IoU 기준
SIM_THRESHOLD = 0.5           # 인물 제외 유사도 임계값 (C가 실험으로 조정)
MIN_VERIFY_SIZE = 40          # 짧은 변(px)이 이보다 작은 얼굴은 인물 제외 판정 없이 모자이크 (C가 조정)
RECHECK_FRAMES = 15           # 제외로 판정된 track을 다시 확인하는 간격 (프레임, C가 조정)
MIN_EVAL_SIZE = 15            # 짧은 변(px)이 이보다 작은 정답 얼굴은 누락률에서 따로 집계 (A)
TRACKER = {                   # ByteTrack 설정 (A). 홍대 30초 실험으로 track ID 162개 -> 125개
    "track_high_thresh": CONF,  # 1단계 매칭 점수
    "track_low_thresh": 0.1,    # 2단계 매칭 점수: 고개를 돌려 점수가 낮아진 얼굴도 기존 track에 이어 붙임
    "new_track_thresh": CONF,   # 새 track을 만드는 점수
    "track_buffer": 30,         # 사라진 track을 기억하는 프레임 수
    "match_thresh": 0.9,        # 같은 사람으로 이어 붙이는 기준 (기본 0.8, 높을수록 너그러움)
    "fuse_score": True,
}
