"""검출기 프롬프트 정규화 + mock 검출기 테스트."""

from __future__ import annotations

import numpy as np

from pysical.io.types import Frame
from pysical.perception.detector import MockDetector, normalize_prompt


def test_korean_single_word():
    assert normalize_prompt("사과") == ["apple"]


def test_korean_with_josa():
    """조사가 붙어도 매핑된다: '사과를 찾아줘'의 '사과를' → apple."""
    assert normalize_prompt("사과를") == ["apple"]


def test_korean_conjunction():
    """'컵과 마우스' → [cup, mouse]."""
    assert normalize_prompt("컵과 마우스") == ["cup", "mouse"]


def test_english_passthrough():
    """이미 영어면 소문자로 그대로 통과."""
    assert normalize_prompt("Bottle") == ["bottle"]


def test_dedup():
    """같은 영어 라벨로 매핑되는 두 단어는 하나로 합쳐진다.

    (머그컵은 이제 LVIS의 별도 클래스 'mug'로 매핑되므로 노트북/랩탑을 쓴다)
    """
    assert normalize_prompt("노트북 랩탑") == ["laptop computer"]


def test_apple_not_broken_by_conjunction():
    """회귀: '사과'가 접속사 '과'로 잘려 '사'가 되면 안 된다(실측 버그)."""
    assert normalize_prompt("사과") == ["apple"]
    assert normalize_prompt("사과 물병") == ["apple", "water bottle"]
    assert normalize_prompt("컵 마우스 사과") == ["cup", "mouse", "apple"]


def test_apple_with_conjunction_tail():
    """'사과와 컵' 같은 접속 꼬리도 올바르게 처리."""
    assert normalize_prompt("사과와 컵") == ["apple", "cup"]


def test_standalone_conjunction_dropped():
    """독립 접속사 토큰 '과'/'와'는 버려진다."""
    assert normalize_prompt("컵 과 마우스") == ["cup", "mouse"]


def test_command_words_removed():
    """음성 명령어(찾아줘/보여줘)는 제거되고 대상만 남는다."""
    assert normalize_prompt("사과 찾아줘") == ["apple"]
    assert normalize_prompt("컵 보여줘") == ["cup"]
    assert normalize_prompt("마우스 잡아줘") == ["mouse"]


def test_find_the_cup_english():
    """영어 명령형도 대상만 남긴다."""
    assert normalize_prompt("find the cup") == ["cup"]


def test_mock_detector_returns_box():
    det = MockDetector()
    det.set_prompt(["apple"])
    frame = Frame(timestamp=0.0, T_world_camera=np.eye(4), rgb=np.zeros((120, 160, 3), np.uint8))
    dets = det.detect(frame)
    assert len(dets) == 1
    assert dets[0].label == "apple"
    assert dets[0].bbox_xyxy.shape == (4,)


def _run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} 통과")
    return failed == 0


if __name__ == "__main__":
    import sys

    sys.exit(0 if _run_all() else 1)
