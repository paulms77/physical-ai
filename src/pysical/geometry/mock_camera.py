"""가상 카메라 포즈 생성 (iPhone 없이 파이프라인을 검증하기 위한 도구).

실제 iPhone/ARKit 없이도 수신기·좌표변환·시각화를 끝까지 돌려보려면, 그럴듯한
카메라 포즈를 스스로 만들어내야 한다. 여기서는 카메라가 **원 궤적을 돌며 항상
원의 중심(책상 위 물체가 있을 법한 곳)을 바라보는** 움직임을 ARKit world 규약으로
생성한다. 순수 함수라 단위 테스트가 쉽다.

ARKit world 규약: 오른손, Y=위. 카메라는 OpenGL 관례라 보는 방향이 -Z.
"""

from __future__ import annotations

import numpy as np


def look_at_pose_arkit(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray | None = None,
) -> np.ndarray:
    """카메라가 ``eye``에서 ``target``을 바라보는 ARKit 규약 4x4 포즈를 만든다.

    ARKit 카메라는 -Z 방향을 바라본다(OpenGL 관례). 따라서 카메라 로컬 축을 world
    기준으로 이렇게 잡는다::

        forward = target - eye      (카메라가 향하는 실제 방향)
        z_axis  = -forward          (ARKit 카메라의 +Z는 뒤쪽)
        x_axis  = up × z_axis       (오른쪽)
        y_axis  = z_axis × x_axis   (위)

    Parameters
    ----------
    eye:
        카메라 위치 (3,), ARKit world.
    target:
        바라볼 지점 (3,), ARKit world.
    up:
        world의 '위' 방향. 기본값은 ARKit world의 +Y.

    Returns
    -------
    np.ndarray
        (4, 4) ARKit world → camera 포즈 (T_world_camera).
    """
    if up is None:
        up = np.array([0.0, 1.0, 0.0])  # ARKit world의 위

    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    forward = target - eye
    forward /= np.linalg.norm(forward)

    z_axis = -forward                     # ARKit 카메라 +Z = 뒤
    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)     # 이미 단위벡터

    T = np.eye(4, dtype=np.float64)
    # 회전: 열이 각각 카메라 로컬 축의 world 표현.
    T[:3, 0] = x_axis
    T[:3, 1] = y_axis
    T[:3, 2] = z_axis
    T[:3, 3] = eye
    return T


def orbit_pose_arkit(
    t: float,
    radius: float = 1.0,
    height: float = 1.2,
    target: np.ndarray | None = None,
    period_s: float = 12.0,
) -> np.ndarray:
    """시각 ``t``(초)에서, 원 궤적을 도는 카메라의 ARKit 포즈를 만든다.

    카메라는 수평면(ARKit world의 XZ 평면) 위에서 반지름 ``radius`` 원을 돌며,
    항상 ``target``(기본: 원 중심, 바닥)을 바라본다. ``period_s`` 마다 한 바퀴.

    이 궤적은 데모의 'viewpoint-invariant' 검증과도 맞물린다 — 같은 물체를
    여러 시점에서 봤을 때 좌표가 일정한지 확인하기 좋은 움직임이다.
    """
    if target is None:
        target = np.array([0.0, 0.0, 0.0])

    angle = 2.0 * np.pi * (t / period_s)
    eye = np.array(
        [radius * np.cos(angle), height, radius * np.sin(angle)],
        dtype=np.float64,
    )
    return look_at_pose_arkit(eye, np.asarray(target, dtype=np.float64))
