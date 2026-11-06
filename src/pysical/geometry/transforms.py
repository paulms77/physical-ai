"""ARKit ↔ ROS 좌표계 변환.

README §6에서 "이 프로젝트에서 가장 실수하기 쉬운 부분"으로 꼽은 곳이다.
ARKit이 준 숫자를 그대로 ROS에 던지면 반드시 틀린다. 두 시스템의 축 규약이
다르기 때문이다. 그 변환을 순수 함수로 못박아, 눈으로 검증하고 테스트로 고정한다.

축 규약 정리
------------
- **ARKit world**  : 오른손 좌표계, Y = 중력 반대(위).
- **ARKit camera** : OpenGL 관례. X=오른쪽, Y=위, Z=뒤. 즉 카메라가 보는 방향은 -Z.
- **ROS map**      : REP-103. X=앞, Y=왼쪽, Z=위.
- **ROS optical**  : X=오른쪽, Y=아래, Z=앞.

두 단계의 변환
--------------
1. 카메라 축 변환 : ARKit camera (x, y, z) → ROS optical (x, -y, -z)
2. 월드 축 변환   : ARKit world  (x, y, z) → ROS map     (-z, -x, y)

두 변환 모두 행렬식이 +1이라 오른손 좌표계가 보존된다.

주의
----
이 부호는 문서로만 믿지 말고 **실측으로 검증**해야 한다(README §6).
폰을 알고 있는 방향으로 1 m 밀었을 때 rerun/RViz의 카메라가 같은 방향으로
1 m 움직이는지 눈으로 확인하는 게 가장 확실하다.
"""

from __future__ import annotations

import numpy as np


def _homogeneous(rotation: np.ndarray) -> np.ndarray:
    """3x3 회전을 4x4 동차 변환 행렬로 감싼다 (평행이동 0)."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rotation
    return T


# --- 월드 축 변환: ARKit world → ROS map ---
# (x, y, z)_arkit → (-z, -x, y)_ros 를 만드는 회전 행렬.
#   ros_x = -arkit_z
#   ros_y = -arkit_x
#   ros_z =  arkit_y
R_MAP_FROM_ARKIT = np.array(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)
T_MAP_FROM_ARKIT = _homogeneous(R_MAP_FROM_ARKIT)


# --- 카메라 축 변환: ARKit camera → ROS optical ---
# (x, y, z)_arkit_cam → (x, -y, -z)_optical 를 만드는 회전 행렬.
R_OPTICAL_FROM_ARKIT_CAM = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)
T_OPTICAL_FROM_ARKIT_CAM = _homogeneous(R_OPTICAL_FROM_ARKIT_CAM)


def arkit_world_point_to_ros(point_arkit: np.ndarray) -> np.ndarray:
    """ARKit world 좌표계의 한 점을 ROS map 좌표계로 옮긴다.

    Parameters
    ----------
    point_arkit:
        (3,) 또는 (N, 3) 배열. ARKit world 기준 좌표.

    Returns
    -------
    np.ndarray
        같은 shape의 ROS map 기준 좌표.
    """
    p = np.asarray(point_arkit, dtype=np.float64)
    single = p.ndim == 1
    p = np.atleast_2d(p)                     # (N, 3)
    ros = p @ R_MAP_FROM_ARKIT.T             # 각 점에 회전 적용
    return ros[0] if single else ros


def arkit_camera_pose_to_ros(T_world_camera_arkit: np.ndarray) -> np.ndarray:
    """ARKit이 준 카메라 포즈(4x4)를 ROS 규약의 포즈로 변환한다.

    포즈 변환은 점 변환과 다르다. 포즈에는 **월드 축**과 **카메라 축** 두 규약이
    동시에 얽혀 있으므로, 양쪽에서 축을 갈아끼워야 한다::

        T_map_optical = T_map_from_arkit @ T_world_camera_arkit @ T_optical_from_arkit_cam^{-1}

    - 왼쪽 곱(``T_map_from_arkit``)  : 월드 프레임을 ARKit world → ROS map 으로.
    - 오른쪽 곱(역행렬)              : 카메라 프레임을 ARKit cam → ROS optical 으로.

    직교 회전이라 역행렬은 전치와 같다.

    Parameters
    ----------
    T_world_camera_arkit:
        (4, 4) ARKit 규약 카메라 포즈.

    Returns
    -------
    np.ndarray
        (4, 4) ROS 규약(map → optical) 카메라 포즈.
    """
    T = np.asarray(T_world_camera_arkit, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"포즈는 4x4여야 합니다. 받은 shape={T.shape}")
    return T_MAP_FROM_ARKIT @ T @ T_OPTICAL_FROM_ARKIT_CAM.T


def is_right_handed(rotation: np.ndarray) -> bool:
    """회전 행렬이 오른손 좌표계를 보존하는지(det ≈ +1) 확인한다."""
    return bool(np.isclose(np.linalg.det(np.asarray(rotation)[:3, :3]), 1.0))
