"""좌표변환 검증 테스트.

README §6의 축 규약이 코드에 정확히 반영됐는지 못박는다. 부호 하나만 틀려도
좌표가 통째로 뒤집히므로, 사람이 직관으로 확신할 수 있는 '알려진 방향' 몇 개를
직접 넣어 확인한다.

실행:  .venv/bin/python -m pytest tests/ -v
       (pytest 없이) .venv/bin/python tests/test_transforms.py
"""

from __future__ import annotations

import numpy as np

from pysical.geometry.transforms import (
    R_MAP_FROM_ARKIT,
    R_OPTICAL_FROM_ARKIT_CAM,
    arkit_camera_pose_to_ros,
    arkit_world_point_to_ros,
    is_right_handed,
)


def test_world_axis_mapping_known_directions():
    """ARKit world의 각 축이 ROS map의 어디로 가는지 직접 확인.

    규약: ARKit (x, y, z) → ROS (-z, -x, y)
      - ARKit +X (오른쪽)     → ROS (0, -1, 0) = 오른쪽(= -Y, ROS에서 오른쪽)
      - ARKit +Y (위)         → ROS (0, 0, 1)  = 위(+Z)  ✔ 중력축이 맞아떨어짐
      - ARKit +Z (뒤/카메라뒤)→ ROS (-1, 0, 0) = 뒤(-X)
    """
    x = arkit_world_point_to_ros(np.array([1.0, 0.0, 0.0]))
    y = arkit_world_point_to_ros(np.array([0.0, 1.0, 0.0]))
    z = arkit_world_point_to_ros(np.array([0.0, 0.0, 1.0]))

    assert np.allclose(x, [0.0, -1.0, 0.0]), x
    assert np.allclose(y, [0.0, 0.0, 1.0]), y   # ARKit '위' → ROS '위'
    assert np.allclose(z, [-1.0, 0.0, 0.0]), z


def test_world_mapping_batch():
    """(N, 3) 여러 점을 한 번에 변환해도 각 점이 같은 결과를 낸다."""
    pts = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    out = arkit_world_point_to_ros(pts)
    assert out.shape == (2, 3)
    assert np.allclose(out[0], [0.0, -1.0, 0.0])
    assert np.allclose(out[1], [0.0, 0.0, 1.0])


def test_rotations_are_right_handed():
    """두 축 변환 모두 det=+1 (오른손 좌표계 보존) — README §6의 요구."""
    assert is_right_handed(R_MAP_FROM_ARKIT)
    assert is_right_handed(R_OPTICAL_FROM_ARKIT_CAM)


def test_pose_identity_maps_to_pure_axis_swap():
    """카메라가 ARKit world 원점에 단위 포즈로 있을 때의 ROS 포즈.

    T_world_camera = I 이면 결과는 두 축 변환의 곱뿐이어야 한다.
    위치는 원점 그대로(0,0,0), 회전만 축 규약 차이만큼 남는다.
    """
    T = arkit_camera_pose_to_ros(np.eye(4))
    assert T.shape == (4, 4)
    assert np.allclose(T[:3, 3], [0.0, 0.0, 0.0])   # 위치는 원점
    assert is_right_handed(T)                        # 여전히 오른손


def test_pose_translation_is_carried_through():
    """카메라가 ARKit world에서 +Y(위)로 1 m 올라가면, ROS에서 +Z(위) 1 m여야 한다.

    이게 README §6이 말한 '눈으로 하는 검증'의 코드판이다:
    실제 위로 든 만큼 시각화의 카메라도 위로 가야 한다.
    """
    T = np.eye(4)
    T[:3, 3] = [0.0, 1.0, 0.0]        # ARKit world 기준 위로 1 m
    ros = arkit_camera_pose_to_ros(T)
    assert np.allclose(ros[:3, 3], [0.0, 0.0, 1.0]), ros[:3, 3]


def _run_all():
    """pytest 없이도 돌릴 수 있게 하는 간이 러너."""
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
