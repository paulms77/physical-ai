"""Focus-and-Refine 순수 로직 테스트 (모델 불필요 부분).

적응 배율 계산과 크롭/좌표 복원 로직을 검증한다. 실제 재검출(모델)은 통합
검증에서 확인했고(mouse 0.22→0.61), 여기서는 기하 로직을 못박는다.

실행:  .venv/bin/python tests/test_refine.py
"""

from __future__ import annotations

import numpy as np

from pysical.perception.refine import (
    MAX_UPSCALE,
    TARGET_OBJECT_PX,
    adaptive_upscale_factor,
    crop_with_padding,
)


def test_small_object_gets_upscaled():
    """작은 물체(짧은 변 40px)는 target(120px)까지 확대(≈3x)."""
    bbox = np.array([100, 100, 140, 160])  # 40x60 → 짧은 변 40
    f = adaptive_upscale_factor(bbox)
    assert np.isclose(f, min(TARGET_OBJECT_PX / 40, MAX_UPSCALE)), f


def test_large_object_not_upscaled():
    """이미 큰 물체(짧은 변 300px)는 확대하지 않는다(1.0)."""
    bbox = np.array([0, 0, 400, 300])
    assert adaptive_upscale_factor(bbox) == 1.0


def test_upscale_is_capped():
    """아주 작은 물체도 MAX_UPSCALE를 넘지 않는다."""
    bbox = np.array([0, 0, 5, 5])  # 5px → 24x 필요하지만 상한
    assert adaptive_upscale_factor(bbox) == MAX_UPSCALE


def test_crop_padding_and_offset():
    """크롭이 여백을 포함하고, 오프셋으로 원좌표 복원이 가능하다."""
    rgb = np.zeros((200, 200, 3), dtype=np.uint8)
    bbox = np.array([80, 80, 120, 120])  # 40x40
    crop, (ox, oy) = crop_with_padding(rgb, bbox, pad_ratio=0.5)
    # pad 0.5 * 40 = 20px 씩 확장 → 원점은 (60, 60)
    assert (ox, oy) == (60, 60)
    assert crop.shape[0] == 80 and crop.shape[1] == 80  # 40 + 20*2


def test_crop_clamps_to_image_bounds():
    """이미지 경계를 벗어나는 크롭은 잘려서 안전하다."""
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    bbox = np.array([90, 90, 110, 110])  # 일부가 밖
    crop, (ox, oy) = crop_with_padding(rgb, bbox)
    assert ox >= 0 and oy >= 0
    assert crop.shape[0] > 0 and crop.shape[1] > 0


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
