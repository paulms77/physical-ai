"""3D object map 인스턴스 연관 테스트.

손 가림(유예)·재등장(같은 ID 유지)·다른 물체(다른 ID) 같은 실시간 핵심 동작을
못박는다. 이게 깨지면 마커가 깜빡이거나 물체가 뒤바뀐다.

실행:  .venv/bin/python tests/test_object_map.py
"""

from __future__ import annotations

import numpy as np

from pysical.tracking.object_map import ObjectMap


def test_new_object_gets_id():
    m = ObjectMap()
    tracked = m.update(0, [("cup", np.array([0.4, -0.6, -1.1]), 0.9)])
    assert len(tracked) == 1
    assert tracked[0].label == "cup"


def test_same_position_keeps_id():
    """수 cm 안에서 다시 관측되면 같은 ID를 유지한다."""
    m = ObjectMap(assoc_dist_m=0.08)
    m.update(0, [("cup", np.array([0.40, -0.60, -1.10]), 0.9)])
    id0 = m.objects[0].obj_id
    # 다음 프레임: 2cm 이동 → 같은 물체
    m.update(1, [("cup", np.array([0.41, -0.60, -1.11]), 0.9)])
    assert len(m.objects) == 1
    assert m.objects[0].obj_id == id0


def test_far_position_is_new_object():
    """멀리 떨어진 관측은 다른 인스턴스(새 ID)."""
    m = ObjectMap(assoc_dist_m=0.08)
    m.update(0, [("cup", np.array([0.4, -0.6, -1.1]), 0.9)])
    m.update(1, [("cup", np.array([0.9, -0.6, -1.1]), 0.9)])  # 50cm 떨어짐
    assert len(m.objects) == 2


def test_occlusion_grace_period():
    """손 가림: 몇 프레임 안 보여도 유예 안에서는 맵에 남는다."""
    m = ObjectMap(max_misses=3)
    m.update(0, [("cup", np.array([0.4, -0.6, -1.1]), 0.9)])
    # 3프레임 동안 관측 없음(가림) → 아직 살아있어야
    for f in range(1, 4):
        m.update(f, [])
    assert len(m.objects) == 1, "유예 기간 안에서는 유지되어야"
    # 유예 초과 → 제거
    m.update(5, [])
    assert len(m.objects) == 0


def test_reappear_after_occlusion_same_id():
    """가렸다 같은 위치에 다시 나타나면 같은 ID로 복귀한다."""
    m = ObjectMap(max_misses=5)
    m.update(0, [("cup", np.array([0.4, -0.6, -1.1]), 0.9)])
    id0 = m.objects[0].obj_id
    m.update(1, [])  # 가림
    m.update(2, [])  # 가림
    # 다시 등장(거의 같은 위치)
    m.update(3, [("cup", np.array([0.41, -0.60, -1.10]), 0.88)])
    assert len(m.objects) == 1
    assert m.objects[0].obj_id == id0  # 새 물체가 아니라 같은 물체


def test_different_labels_not_associated():
    """같은 위치라도 라벨이 다르면 다른 인스턴스."""
    m = ObjectMap(assoc_dist_m=0.5)
    m.update(0, [("cup", np.array([0.4, -0.6, -1.1]), 0.9)])
    m.update(1, [("mouse", np.array([0.4, -0.6, -1.1]), 0.9)])
    assert len(m.objects) == 2


def test_visible_vs_all():
    """visible_objects는 이번에 실제 보인 것만, objects는 유예 포함 전부."""
    m = ObjectMap(max_misses=5)
    m.update(0, [("cup", np.array([0.4, -0.6, -1.1]), 0.9)])
    m.update(1, [])  # 안 보임
    assert len(m.objects) == 1        # 유예로 남음
    assert len(m.visible_objects) == 0  # 이번엔 안 보임


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
