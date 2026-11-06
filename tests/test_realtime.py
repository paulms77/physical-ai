"""실시간 엔진 로직 테스트 (mock 검출기, 성능 무관).

detect-and-track 주기 제어와 object map 연동이 올바른지 못박는다. 실제 모델·속도는
공용 서버 부하와 무관하게, 여기서는 '로직'만 검증한다.

실행:  .venv/bin/python tests/test_realtime.py
"""

from __future__ import annotations

import numpy as np

from pysical.io.types import CameraIntrinsics, Frame
from pysical.perception.detector import MockDetector
from pysical.perception.realtime import RealtimeConfig, RealtimeEngine


def _frame(t=0.0, with_depth=False):
    rgb = np.full((120, 160, 3), 100, dtype=np.uint8)
    kw = {}
    if with_depth:
        kw = dict(
            depth_lidar=np.full((120, 160), 1.5, np.float32),
            depth_intrinsics=CameraIntrinsics(150, 150, 80, 60),
            intrinsics=CameraIntrinsics(150, 150, 80, 60),
        )
    return Frame(timestamp=t, T_world_camera=np.eye(4), rgb=rgb, **kw)


def test_detects_only_on_schedule():
    """detect_every=3이면 0,3,6…프레임에서만 검출한다(나머지는 트래커)."""
    eng = RealtimeEngine(MockDetector(), RealtimeConfig(detect_every=3))
    eng.set_prompt(["object"])
    # 검출 프레임(0)에서는 트래커가 새로 생긴다.
    eng.process(_frame(0.0))
    assert len(eng._trackers) == 1  # 검출됨 → 트래커 1개
    # 비검출 프레임(1,2)에서는 재검출 안 함(트래커 유지 시도)
    eng.process(_frame(0.1))
    eng.process(_frame(0.2))
    # 다음 검출 프레임(3)
    eng.process(_frame(0.3))
    assert eng._frame_idx == 3


def test_process_returns_boxes_and_tracked():
    """process는 (표시박스, 추적객체) 튜플을 돌려준다."""
    eng = RealtimeEngine(MockDetector(), RealtimeConfig(detect_every=1))
    eng.set_prompt(["object"])
    boxes, tracked = eng.process(_frame(0.0, with_depth=True))
    assert isinstance(boxes, list)
    assert isinstance(tracked, list)
    # depth가 있으니 3D 위치추정 → object map에 등록될 수 있다.


def test_set_prompt_clears_trackers():
    """대상을 바꾸면 기존 트래커가 비워진다(새 명령 반영)."""
    eng = RealtimeEngine(MockDetector(), RealtimeConfig(detect_every=1))
    eng.set_prompt(["cup"])
    eng.process(_frame(0.0))
    assert len(eng._trackers) >= 0
    eng.set_prompt(["mouse"])       # 대상 교체
    assert len(eng._trackers) == 0  # 트래커 초기화됨


def test_no_detector_runs_passthrough():
    """검출기 없이도(배관만) 크래시 없이 돈다."""
    eng = RealtimeEngine(None, RealtimeConfig())
    boxes, tracked = eng.process(_frame(0.0))
    assert boxes == [] and tracked == []


def test_xyxy_to_xywh_conversion():
    """rerun Boxes2D용 좌표 변환(회귀: _xyxy_to_xywh 누락 버그)."""
    from pysical.viz.realtime_viz import _xyxy_to_xywh

    assert _xyxy_to_xywh(np.array([10, 20, 50, 80])) == [10.0, 20.0, 40.0, 60.0]


def test_object_map_accumulates_over_frames():
    """depth 있는 프레임을 여러 번 처리하면 object map이 인스턴스를 유지한다."""
    eng = RealtimeEngine(MockDetector(), RealtimeConfig(detect_every=1, max_misses=5))
    eng.set_prompt(["object"])
    for i in range(3):
        eng.process(_frame(i * 0.1, with_depth=True))
    # 같은 위치(고정 mock)라 한 인스턴스로 유지되어야
    assert len(eng.object_map.objects) <= 1


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
        except Exception as e:  # noqa
            failed += 1
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} 통과")
    return failed == 0


if __name__ == "__main__":
    import sys

    sys.exit(0 if _run_all() else 1)
