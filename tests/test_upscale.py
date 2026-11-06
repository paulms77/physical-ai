"""Upscaler(ESPCN super-resolution) 테스트.

ESPCN 모델이 있으면 SR로, 없으면 CUBIC 폴백으로 정확히 배율만큼 확대하는지 확인한다.
(SR의 화질/신뢰도 우위는 통합 실측으로 검증 — docs/REFINE_FINDINGS.md)

실행:  .venv/bin/python tests/test_upscale.py
"""

from __future__ import annotations

import numpy as np

from pysical.perception.upscale import Upscaler, get_upscaler


def test_upscale_changes_size_x3():
    """x3 배율이면 가로·세로가 약 3배가 된다(SR이든 폴백이든)."""
    up = Upscaler(scale=3)
    img = np.random.randint(0, 255, (20, 30, 3), dtype=np.uint8)
    out = up.upscale(img)  # factor 미지정 → 모델 배율
    # SR이면 정확히 3배, 폴백도 3배
    assert abs(out.shape[0] - 60) <= 3 and abs(out.shape[1] - 90) <= 3, out.shape


def test_upscale_custom_factor():
    """factor=2.0을 주면 결과가 약 2배 크기가 된다."""
    up = Upscaler(scale=3)
    img = np.random.randint(0, 255, (25, 25, 3), dtype=np.uint8)
    out = up.upscale(img, factor=2.0)
    assert abs(out.shape[0] - 50) <= 3, out.shape


def test_output_is_uint8_3ch():
    """출력이 원본과 같은 형식(uint8, 3채널)."""
    up = get_upscaler(scale=3)
    img = np.full((16, 16, 3), 100, dtype=np.uint8)
    out = up.upscale(img, factor=2.0)
    assert out.dtype == np.uint8 and out.shape[2] == 3


def test_singleton_reuse():
    """get_upscaler는 같은 scale이면 같은 인스턴스를 재사용한다."""
    a = get_upscaler(scale=3)
    b = get_upscaler(scale=3)
    assert a is b


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
