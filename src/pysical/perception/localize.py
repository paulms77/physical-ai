"""관측기반 3D 위치추정 파이프라인.

docs/ARCHITECTURE_v2.md §2의 절차를 코드로 옮긴 곳이다. 한 물체의 2D 검출(mask/bbox)과
프레임(depth + pose + intrinsics)을 받아, 3D world 좌표를 만든다.

세 depth 경로 (같은 픽셀·같은 변환 → 오차가 depth 차이에서만 나옴)
----------------------------------------------------------------
- METHOD_PLANE (A′) : 물체 주변 표면의 3D 점에 RANSAC 평면 fitting → 접지점을 평면에 투영
- METHOD_DEPTH (B)  : 접지점 픽셀의 depth를 직접 3D로 (평면 위 없는 물체용)
- METHOD_LIDAR      : LiDAR depth를 같은 방식으로 (정답지)

모든 경로가 마지막에 카메라 좌표 → ARKit world → ROS map 으로 간다
(geometry.transforms 재사용). 즉 좌표계 처리는 한 곳에만 있다.
"""

from __future__ import annotations

import numpy as np

from ..geometry.backproject import backproject_depth_map, backproject_pixels
from ..geometry.plane_fit import fit_plane_ransac
from ..geometry.transforms import arkit_world_point_to_ros
from ..io.types import CameraIntrinsics, Frame
from ..utils.logging import get_logger
from .types import (
    METHOD_DEPTH,
    METHOD_LIDAR,
    METHOD_PLANE,
    Detection,
    ObjectLocation,
)

log = get_logger(__name__)


def _camera_point_to_world(point_cam: np.ndarray, T_world_camera: np.ndarray) -> np.ndarray:
    """카메라 좌표계 3D 점 → ARKit world → ROS map.

    T_world_camera(ARKit 규약)로 world(ARKit)로 올린 뒤, ROS map 규약으로 변환한다.
    """
    p_h = np.append(point_cam, 1.0)               # 동차 좌표
    p_world_arkit = (T_world_camera @ p_h)[:3]    # ARKit world
    return arkit_world_point_to_ros(p_world_arkit)


def _sample_depth_at(depth: np.ndarray, uv: np.ndarray, window: int = 2) -> float | None:
    """픽셀 (u, v) 주변 작은 창에서 유효 depth의 중앙값을 뽑는다.

    한 픽셀만 읽으면 노이즈/무효값에 취약하므로, 작은 창의 중앙값으로 안정화한다.
    """
    u, v = int(round(uv[0])), int(round(uv[1]))
    h, w = depth.shape
    y0, y1 = max(0, v - window), min(h, v + window + 1)
    x0, x1 = max(0, u - window), min(w, u + window + 1)
    patch = depth[y0:y1, x0:x1]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def localize_by_depth(
    detection: Detection,
    depth: np.ndarray,
    depth_intrinsics: CameraIntrinsics,
    frame: Frame,
    method: str = METHOD_DEPTH,
) -> ObjectLocation | None:
    """방법 B / LiDAR: 접지점 픽셀의 depth를 직접 3D로 만든다.

    평면을 쓰지 않으므로 공중·선반 위 물체도 처리할 수 있다(방법 B가 빛나는 지점).
    """
    uv = detection.ground_pixel()
    # detection 픽셀은 RGB 해상도 기준일 수 있으니, depth 해상도로 맞춰준다.
    uv_d = _rescale_pixel(uv, frame, depth.shape)
    z = _sample_depth_at(depth, uv_d)
    if z is None:
        return None
    point_cam = backproject_pixels(uv_d[None, :], np.array([z]), depth_intrinsics)[0]
    pos_world = _camera_point_to_world(point_cam, frame.T_world_camera)
    return ObjectLocation(detection.label, detection.score, pos_world, method)


def localize_by_plane(
    detection: Detection,
    depth: np.ndarray,
    depth_intrinsics: CameraIntrinsics,
    frame: Frame,
    distance_threshold: float = 0.01,
) -> ObjectLocation | None:
    """방법 A′: 물체 주변 표면을 RANSAC 평면으로 추정하고 접지점을 투영한다.

    절차
    ----
    1. 물체 접지점 아래쪽 영역(표면일 가능성이 큰 곳)의 depth 점 구름을 모은다.
    2. RANSAC으로 지지 평면을 fitting.
    3. 접지점 픽셀을 3D로 만든 뒤, 그 평면에 수직 투영.
    4. 카메라 → world 변환.
    """
    uv = detection.ground_pixel()
    uv_d = _rescale_pixel(uv, frame, depth.shape)

    surface_mask = _surface_region_mask(depth.shape, detection, frame)
    cloud = backproject_depth_map(depth, depth_intrinsics, mask=surface_mask, stride=2)
    if cloud.shape[0] < 30:
        return None

    plane = fit_plane_ransac(cloud, distance_threshold=distance_threshold, min_inliers=20)
    if plane is None:
        return None

    # 접지점을 카메라 광선 위 임의 depth로 만들고 → 평면과 교차(=투영)한다.
    z0 = _sample_depth_at(depth, uv_d) or float(np.median(np.linalg.norm(cloud, axis=1)))
    ground_cam = backproject_pixels(uv_d[None, :], np.array([z0]), depth_intrinsics)[0]
    ground_on_plane = plane.project(ground_cam)

    pos_world = _camera_point_to_world(ground_on_plane, frame.T_world_camera)
    return ObjectLocation(
        detection.label,
        detection.score,
        pos_world,
        METHOD_PLANE,
        plane_inlier_ratio=plane.inlier_ratio,
    )


def _rescale_pixel(uv_rgb: np.ndarray, frame: Frame, depth_shape: tuple[int, int]) -> np.ndarray:
    """RGB 해상도 기준 픽셀을 depth 해상도 기준으로 변환한다.

    detection은 RGB 이미지에서 나오지만 depth는 해상도가 다를 수 있으므로(256×192),
    좌표를 비율로 맞춰준다. RGB가 없으면 그대로 둔다.
    """
    if frame.rgb is None:
        return np.asarray(uv_rgb, dtype=np.float64)
    h_rgb, w_rgb = frame.rgb.shape[:2]
    h_d, w_d = depth_shape
    sx, sy = w_d / w_rgb, h_d / h_rgb
    return np.array([uv_rgb[0] * sx, uv_rgb[1] * sy])


def _surface_region_mask(depth_shape: tuple[int, int], detection: Detection, frame: Frame) -> np.ndarray:
    """물체가 놓인 표면일 가능성이 큰 영역의 마스크를 만든다.

    간단하고 강건한 휴리스틱: 물체 bbox의 **바로 아래 + 좌우로 조금 넓힌** 띠를
    표면 후보로 본다. 물체 자체(mask)는 제외해 평면 fitting을 오염시키지 않는다.
    """
    h_d, w_d = depth_shape
    x1, y1, x2, y2 = detection.bbox_xyxy
    # bbox를 depth 해상도로 스케일
    p1 = _rescale_pixel(np.array([x1, y1]), frame, depth_shape)
    p2 = _rescale_pixel(np.array([x2, y2]), frame, depth_shape)
    bw = p2[0] - p1[0]
    bh = p2[1] - p1[1]

    mask = np.zeros((h_d, w_d), dtype=bool)
    # 표면 띠: 물체 하단 y부터 아래로 물체 높이의 1.5배, 좌우로 폭의 0.5배 확장.
    sy0 = int(max(0, p2[1] - 0.1 * bh))
    sy1 = int(min(h_d, p2[1] + 1.5 * bh))
    sx0 = int(max(0, p1[0] - 0.5 * bw))
    sx1 = int(min(w_d, p2[0] + 0.5 * bw))
    mask[sy0:sy1, sx0:sx1] = True
    return mask


def localize_object(
    detection: Detection,
    frame: Frame,
) -> dict[str, ObjectLocation]:
    """한 물체에 대해 가능한 모든 depth 경로로 위치를 추정한다.

    세 경로를 **같은 검출·같은 프레임**으로 돌려, 비교 실험(Phase 3)에 바로 쓸 수
    있는 형태로 결과를 dict로 반환한다. 호출부는 필요한 경로를 dict에서 골라 쓴다
    (실시간 엔진은 첫 번째 결과를 쓴다).

    Returns
    -------
    dict[str, ObjectLocation]
        {method: 결과}. 가능한 경로만 담긴다(depth 없으면 빈 dict).
    """
    results: dict[str, ObjectLocation] = {}

    # LiDAR 경로 (있으면): 정답지 + 관측평면 둘 다 시도
    if frame.depth_lidar is not None and frame.depth_intrinsics is not None:
        loc = localize_by_depth(
            detection, frame.depth_lidar, frame.depth_intrinsics, frame, method=METHOD_LIDAR
        )
        if loc:
            results[METHOD_LIDAR] = loc
        plane_loc = localize_by_plane(
            detection, frame.depth_lidar, frame.depth_intrinsics, frame
        )
        if plane_loc:
            results[METHOD_PLANE] = plane_loc

    # monocular depth 경로 (있으면): 방법 B
    if frame.depth_mono is not None:
        # mono depth는 RGB 해상도라 RGB intrinsics를 쓴다.
        intr = frame.intrinsics or frame.depth_intrinsics
        if intr is not None:
            loc = localize_by_depth(detection, frame.depth_mono, intr, frame, method=METHOD_DEPTH)
            if loc:
                results[METHOD_DEPTH] = loc

    if not results:
        log.debug("위치추정 실패(depth 없음/부족): label=%s", detection.label)
    return results
