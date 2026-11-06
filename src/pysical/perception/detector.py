"""검출기 인터페이스 + 한↔영 프롬프트 매핑.

검출기의 **계약(인터페이스)** 을 먼저 정하고, 실제 모델(YOLOE 등)은 이 인터페이스를
구현하는 형태로 나중에 갈아끼운다. 이렇게 하면 위치추정 파이프라인은 "어떤 모델을
쓰는지" 몰라도 되고, mock 검출기로 전체 흐름을 미리 검증할 수 있다(Phase 0에서
mock 송신기로 했던 것과 같은 전략).

실제 YOLOE 연결은 Phase 1 후반에 `YoloeDetector(BaseDetector)` 로 추가한다.
그 자리를 여기 stub으로 표시해 둔다.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ..io.types import Frame
from ..utils.logging import get_logger
from .types import Detection

log = get_logger(__name__)


# 검출 모델은 영어로 학습됐으므로 한국어 프롬프트를 영어로 바꾼다(README §1).
# 대상 물체가 열 개 남짓이라 작은 매핑 테이블로 시작하고, 필요하면 나중에 LLM으로 확장.
from .ko_labels import EXPERIMENTAL, KO_EN_LABELS  # noqa: F401  (공개 API 유지)

# 음성 발화의 군더더기(명령어·조사성 단어). 대상이 아니므로 프롬프트에서 제거한다.
# 예: "사과 찾아줘" → "찾아줘" 제거 → ["apple"]만 남는다.
# 토큰 '전체'가 일치할 때만 버리므로 "안경"의 "안" 같은 부분 일치 사고는 없다.
_STOPWORDS = {
    "과", "와",
    "찾아줘", "찾아", "찾아라", "보여줘", "보여", "잡아줘", "잡아",
    "줘", "해줘", "please", "find", "show", "the", "a", "me",
    # 문장으로 말할 때 섞여 들어오는 서술어·대명사 조각. 대상어가 아니다.
    "잡고", "잡을", "보고", "볼래", "싶어", "싶다", "싶지", "싶은",
    "않고", "않아", "않을", "안", "말고", "말아",
    "싫어", "싫다", "좋아", "좋다",
    "하기", "하고", "해", "한번", "좀", "이제", "다시", "지금", "그리고",
    "나는", "내가", "저는", "제가", "우리", "너", "나",
    "것", "거", "게",
}


def normalize_prompt(prompt: str) -> list[str]:
    """자연어 프롬프트를 검출기가 쓸 영어 라벨 리스트로 정규화한다.

    예: "컵과 마우스" → ["cup", "mouse"]. 매핑에 없는 토큰은 그대로 소문자로 넘겨
    (이미 영어일 수 있으므로) open-vocab 검출기가 시도하게 둔다.
    """
    # 1) 접속사/구분자로 토큰을 나눈다. 주의: "과/와"를 무턱대고 치환하면 "사과"의
    #    "과"까지 잘려 "사"가 되는 버그가 생긴다(실측). 그래서 접속사는 **공백으로
    #    둘러싸인 독립 토큰**이거나 명시적 구분자(쉼표 등)일 때만 자른다.
    for sep in [",", "、", "그리고"]:
        prompt = prompt.replace(sep, " ")
    raw_tokens = [t.strip() for t in prompt.split() if t.strip()]
    # 독립 접속사 토큰("과"/"와")과 명령어(음성 발화의 군더더기)를 버린다.
    # 예: "사과 찾아줘" → "찾아줘"는 대상이 아니라 명령이므로 제거.
    tokens = [t for t in raw_tokens if t not in _STOPWORDS]

    labels: list[str] = []
    for tok in tokens:
        base = tok
        # 조사는 "떼었을 때 매핑에 있을 때만" 뗀다. 이렇게 하면 "사과"(조사 아님)는
        # 그대로 두고, "사과를"처럼 진짜 조사가 붙은 경우만 처리한다.
        # 접속 꼬리 "과/와"도 여기서 함께 시도(예: "사과와" → "사과").
        for josa in ["을", "를", "이", "가", "은", "는", "도", "과", "와"]:
            if base.endswith(josa) and base[:-1] in KO_EN_LABELS:
                base = base[:-1]
                break
        mapped = KO_EN_LABELS.get(base)
        if mapped is None:
            mapped = base.lower()
            if _is_hangul(mapped):
                # 한국어인데 사전에 없다. MobileCLIP 텍스트 인코더는 영어 전용이라
                # 이 라벨은 사실상 무작위 방향이 되어 아무것도 못 잡는다(실측:
                # "펜"↔"pen" 유사도 0.24 < "cup"↔"pen" 0.30). 조용히 실패하지
                # 않도록 경고를 남긴다.
                log.warning(
                    "사전에 없는 한국어 라벨 %r — 검출되지 않을 가능성이 높다. "
                    "영어로 말하거나 KO_EN_LABELS에 추가하세요.", mapped
                )
        elif mapped in EXPERIMENTAL:
            # LVIS 1203 밖이라 검출 성능이 보장되지 않는 라벨. 안 잡혔을 때
            # 사전 문제인지 모델 한계인지 구분하려고 남긴다.
            log.info("실험적 라벨 %r — LVIS 어휘 밖이라 검출이 불확실하다", mapped)
        labels.append(mapped)
    # 중복 제거(순서 유지)
    seen: set[str] = set()
    return [x for x in labels if not (x in seen or seen.add(x))]


class BaseDetector(Protocol):
    """검출기가 지켜야 할 인터페이스.

    구현체는 프레임과 대상 라벨을 받아 Detection 리스트를 낸다. prompt-then-detect
    모델은 set_prompt로 어휘를 고정한 뒤 detect를 반복 호출하는 형태가 될 것이다.
    """

    def set_prompt(self, labels: list[str]) -> None:
        """검출 대상 어휘를 설정한다(prompt-then-detect)."""
        ...

    def detect(self, frame: Frame) -> list[Detection]:
        """현재 프롬프트에 맞는 물체들을 검출한다."""
        ...


class MockDetector:
    """테스트용 검출기 — 화면 중앙에 고정 bbox를 하나 낸다.

    실제 모델 없이 위치추정 파이프라인·시각화·ROS 발행을 끝까지 돌려보기 위한 stub.
    YOLOE를 붙이기 전까지 이걸로 배관을 검증한다.
    """

    def __init__(self) -> None:
        self._labels = ["object"]

    def set_prompt(self, labels: list[str]) -> None:
        self._labels = labels or ["object"]

    def detect(self, frame: Frame) -> list[Detection]:
        if frame.rgb is None:
            return []
        h, w = frame.rgb.shape[:2]
        # 화면 중앙 하단에 물체가 있다고 가정한 더미 박스.
        cx, cy = w / 2, h * 0.6
        bbox = np.array([cx - w * 0.1, cy - h * 0.15, cx + w * 0.1, cy + h * 0.05])
        return [Detection(label=self._labels[0], score=0.99, bbox_xyxy=bbox)]


# 실제 open-vocabulary 검출기(YOLOE)는 yoloe_detector.py 의 YoloeDetector 로 구현했다.
# 무거운 ultralytics/torch import를 이 모듈에서 분리하기 위해 파일을 나눴다 —
# 그래야 여기(순수 파이썬)와 기하 코드는 무거운 의존성 없이 import·테스트된다.
#
# 하이브리드 방침(memory: pysical-3device-arch): 지금은 ultralytics(.pt)로 검증,
# 최종 배포는 A6000에서 ONNX export → 노트북 onnxruntime. 교체 지점은 YoloeDetector
# 내부뿐이고 이 BaseDetector 인터페이스는 그대로 유지된다.


def _is_hangul(text: str) -> bool:
    """한글 음절이 하나라도 있으면 True."""
    return any("\uac00" <= ch <= "\ud7a3" for ch in text)
