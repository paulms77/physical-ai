"""Auto-Focus 대상 선택 — 고정(lock-on)과 음성 전환.

포커스가 프레임마다 대상을 다시 고르면 컵이 둘일 때 화면이 깜빡인다. 한번 고른
대상을 유지하는 것이 이 모듈의 존재 이유다.
"""

from __future__ import annotations

import numpy as np

from pysical.io.types import Frame
from pysical.perception.types import Detection
from pysical.viz.focus_select import FocusSelector

W, H = 640, 480


def _frame() -> Frame:
    return Frame(
        timestamp=0.0,
        T_world_camera=np.eye(4),
        rgb=np.zeros((H, W, 3), dtype=np.uint8),
    )


def _det(label: str, cx: float, cy: float, half: float = 60.0, score: float = 0.8) -> Detection:
    return Detection(
        label=label,
        score=score,
        bbox_xyxy=np.array([cx - half, cy - half, cx + half, cy + half], dtype=float),
    )


def test_prefers_centered_over_higher_score():
    """사용자가 겨눈 것이 중앙에 온다 — 신뢰도보다 중앙 근접도를 우선한다."""
    centered = _det("cup", 320, 240, score=0.55)
    edge = _det("cup", 610, 40, score=0.95)
    assert FocusSelector().select([centered, edge], _frame()) is centered


def test_lock_survives_score_inversion():
    """신뢰도가 역전돼도 대상을 바꾸지 않는다(깜빡임 방지)."""
    sel = FocusSelector()
    left = _det("cup", 300, 240, score=0.60)
    right = _det("cup", 600, 100, score=0.90)
    assert sel.select([left, right], _frame()) is left

    left_low = _det("cup", 303, 241, score=0.30)
    right_high = _det("cup", 600, 100, score=0.99)
    assert sel.select([left_low, right_high], _frame()) is left_low


def test_lock_released_after_grace():
    """대상이 사라지면 유예 동안 기다렸다가 고정을 푼다."""
    sel = FocusSelector()
    sel.select([_det("cup", 320, 240)], _frame())
    assert sel.locked_label == "cup"

    for _ in range(sel._cfg.lost_grace + 2):
        assert sel.select([], _frame()) is None
    assert sel.locked_label is None


def test_next_cycles_within_same_label():
    sel = FocusSelector()
    a = _det("cup", 200, 240)
    b = _det("cup", 440, 240)
    first = sel.select([a, b], _frame())
    sel.request("next")
    assert sel.select([a, b], _frame()) is not first


def test_next_widens_when_label_has_single_instance():
    """컵 1개 + 펜 1개면 '다음'이 갈 곳이 없으므로 전체로 넓힌다."""
    sel = FocusSelector()
    cup = _det("cup", 300, 240)
    pen = _det("pen", 560, 250)
    assert sel.select([cup, pen], _frame()) is cup
    sel.request("next")
    assert sel.select([cup, pen], _frame()) is pen


def test_nearest_falls_back_to_largest_box_without_depth():
    sel = FocusSelector()
    small = _det("cup", 200, 240, half=25)
    big = _det("cup", 460, 240, half=95)
    sel.request("nearest")
    assert sel.select([small, big], _frame()) is big


def test_request_label_switches_focus_kind():
    """'포커스 펜' — 검출 어휘는 그대로 두고 포커스만 그 종류로 옮긴다."""
    sel = FocusSelector()
    cup = _det("cup", 300, 240)
    pen = _det("pen", 560, 250)
    assert sel.select([cup, pen], _frame()) is cup

    sel.request_label("pen")
    assert sel.select([cup, pen], _frame()) is pen
    assert sel.locked_label == "pen"


def test_request_label_waits_then_gives_up():
    """지정한 라벨이 아직 안 보이면 잠시 기다렸다가 포기한다."""
    sel = FocusSelector()
    cup = _det("cup", 300, 240)
    sel.request_label("pen", retries=3)
    for _ in range(3):
        sel.select([cup], _frame())
    assert sel._pending_label is None
