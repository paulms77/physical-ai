"""Depth → 3D 점 역투영 (backprojection).

핀홀 카메라 모델에서, 픽셀 (u, v)와 그 픽셀의 depth z가 주어지면 카메라 좌표계의
3D 점을 이렇게 복원한다::

    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    z = z

이게 관측기반 표면추정(docs/ARCHITECTURE_v2.md §2)의 3단계 "Backproject"다.
LiDAR depth든 monocular depth든 **같은 함수로** 3D 점을 만들어, 이후 평면 fitting과
좌표 변환이 depth 소스에 무관하게 동작하도록 한다.

좌표계 규약
-----------
여기서 나오는 3D 점은 **카메라 좌표계**다. ARKit 카메라 규약(X 오른쪽, Y 위, Z 뒤,
보는 방향 -Z)을 따르므로, 화면에 보이는(카메라 앞) 점의 z는 **음수**다.
depth 맵이 담고 있는 값은 '거리'(양수)이므로, 카메라 앞 방향(-Z)에 놓으려면
z = -depth 로 부호를 뒤집는다. world로 옮기는 건 transforms 모듈의 몫이다.
"""

from __future__ import annotations

import numpy as np

from ..io.types import CameraIntrinsics


def backproject_pixels(
    pixels_uv: np.ndarray,
    depths: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """선택된 픽셀들을 카메라 좌표계 3D 점으로 역투영한다.

    Parameters
    ----------
    pixels_uv:
        (N, 2) 정수/실수 배열. 각 행이 (u, v) = (열, 행) 픽셀 좌표.
    depths:
        (N,) 배열. 각 픽셀의 depth(거리, 미터, 양수).
    intrinsics:
        depth 맵 해상도에 맞는 카메라 내부 파라미터.

    Returns
    -------
    np.ndarray
        (N, 3) 카메라 좌표계 3D 점. 카메라 앞 점의 z는 음수(ARKit 규약).
    """
    uv = np.asarray(pixels_uv, dtype=np.float64).reshape(-1, 2)
    z = np.asarray(depths, dtype=np.float64).reshape(-1)
    if uv.shape[0] != z.shape[0]:
        raise ValueError(f"픽셀 수({uv.shape[0]})와 depth 수({z.shape[0]})가 다릅니다.")

    u, v = uv[:, 0], uv[:, 1]
    x = (u - intrinsics.cx) / intrinsics.fx * z
    y = (v - intrinsics.cy) / intrinsics.fy * z
    # depth는 거리(양수), 카메라 앞은 -Z(ARKit) 이므로 부호를 뒤집는다.
    return np.stack([x, y, -z], axis=1)


def backproject_depth_map(
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    mask: np.ndarray | None = None,
    stride: int = 1,
) -> np.ndarray:
    """depth 맵 전체(또는 mask 영역)를 3D 점 구름으로 역투영한다.

    NaN(무효 depth)과 0 이하 값은 자동으로 제외한다. 표면 fitting에 넣을 point
    cloud를 만들 때 쓴다.

    Parameters
    ----------
    depth:
        (H, W) depth 맵(미터). NaN/0은 무효로 간주해 버린다.
    intrinsics:
        depth 해상도에 맞는 K.
    mask:
        (H, W) bool. 주어지면 True 픽셀만 역투영한다(예: 책상 표면 영역).
    stride:
        다운샘플 간격. 1이면 전 픽셀, 2면 격자로 1/4만 — CPU 부담을 줄인다.

    Returns
    -------
    np.ndarray
        (M, 3) 유효한 카메라 좌표계 3D 점.
    """
    depth = np.asarray(depth, dtype=np.float64)
    h, w = depth.shape

    vs, us = np.mgrid[0:h:stride, 0:w:stride]
    us = us.ravel()
    vs = vs.ravel()
    z = depth[vs, us]

    valid = np.isfinite(z) & (z > 0)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)[vs, us]

    us, vs, z = us[valid], vs[valid], z[valid]
    if us.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    pixels = np.stack([us, vs], axis=1)
    return backproject_pixels(pixels, z, intrinsics)
