"""위치추정 파이프라인 통합 테스트 (합성 3D 장면).

알려진 3D 장면(바닥 평면 + 그 위 물체)을 카메라로 '렌더'해 depth 맵을 만들고,
파이프라인이 그 물체의 world 좌표를 되찾는지 확인한다. 이게 관측기반 표면추정
전체(역투영→평면fitting→투영→좌표변환)의 안전망이다.

전략
----
카메라를 ARKit world 원점에 '아래를 비스듬히 내려다보게' 두고, 바닥 평면(ARKit
world의 y=0)과 그 위 물체를 depth로 만든다. 정답 world 좌표는 우리가 배치한 값이므로,
추정치와 비교할 수 있다.

실행:  .venv/bin/python tests/test_localize.py
"""

from __future__ import annotations

import numpy as np

from pysical.geometry.mock_camera import look_at_pose_arkit
from pysical.geometry.transforms import arkit_world_point_to_ros
from pysical.io.types import CameraIntrinsics, Frame
from pysical.perception.localize import localize_object
from pysical.perception.types import METHOD_LIDAR, METHOD_PLANE, Detection

# depth/RGB 공통 해상도로 단순화(스케일 이슈 분리). 테스트는 파이프라인 로직에 집중.
W, H = 160, 120
K = CameraIntrinsics(fx=150.0, fy=150.0, cx=W / 2, cy=H / 2)


def _render_floor_with_object(T_world_camera, obj_world, obj_radius=0.05):
    """바닥 평면(y=0)과 그 위 물체를 렌더해 depth 맵을 만든다.

    각 픽셀의 카메라 광선을 world로 쏘아, 바닥(y=0)과의 교차 거리를 depth로 쓴다.
    물체 근처 픽셀은 물체 표면(살짝 더 가까운 depth)으로 덮어씌운다.
    """
    # 카메라 위치·회전 (ARKit world)
    R = T_world_camera[:3, :3]
    cam_pos = T_world_camera[:3, 3]

    depth = np.full((H, W), np.nan, dtype=np.float32)
    for v in range(H):
        for u in range(W):
            # 카메라 좌표계 광선 방향 (ARKit: 앞은 -Z)
            dir_cam = np.array([(u - K.cx) / K.fx, (v - K.cy) / K.fy, -1.0])
            dir_world = R @ dir_cam
            # 바닥 평면 y=0 과의 교차: cam_pos.y + t*dir.y = 0
            if abs(dir_world[1]) < 1e-6:
                continue
            t = -cam_pos[1] / dir_world[1]
            if t <= 0:
                continue
            hit = cam_pos + t * dir_world           # world 상의 바닥 교차점
            # 광선 거리(=depth). 카메라 좌표 z로 환산: 실제 거리 * |dir_cam_z 정규화|
            # 여기서는 유클리드 거리를 depth로 근사(정규화된 방향이 아니므로 보정).
            depth_val = t * np.linalg.norm(dir_world)
            # 물체: world 상에서 obj_world 근처면 조금 더 가깝게(물체 표면).
            if np.linalg.norm(hit[[0, 2]] - obj_world[[0, 2]]) < obj_radius:
                depth_val *= 0.97
            depth[v, u] = depth_val
    return depth


def _make_detection_at(obj_world, T_world_camera):
    """물체 world 위치를 이미지에 투영해 bbox/접지점 픽셀을 만든다."""
    R = T_world_camera[:3, :3]
    cam_pos = T_world_camera[:3, 3]
    rel = obj_world - cam_pos
    p_cam = R.T @ rel                                # world→camera
    # 투영 (ARKit: 앞 -Z)
    u = K.fx * (p_cam[0] / -p_cam[2]) + K.cx
    v = K.fy * (p_cam[1] / -p_cam[2]) + K.cy
    bbox = np.array([u - 8, v - 16, u + 8, v + 2])   # 물체 대략 박스(접지점이 하단)
    return Detection(label="obj", score=0.9, bbox_xyxy=bbox)


def _setup():
    """카메라가 원점 근처에서 바닥 위 물체를 내려다보는 장면."""
    obj_world = np.array([0.2, 0.0, -0.8])           # 바닥(y=0) 위, 카메라 앞
    cam_eye = np.array([0.0, 0.5, 0.0])              # 50cm 높이
    T = look_at_pose_arkit(eye=cam_eye, target=obj_world)
    depth = _render_floor_with_object(T, obj_world)
    det = _make_detection_at(obj_world, T)
    frame = Frame(
        timestamp=0.0,
        T_world_camera=T,
        rgb=np.zeros((H, W, 3), dtype=np.uint8),     # 해상도 정보용
        intrinsics=K,
        depth_lidar=depth,
        depth_intrinsics=K,
    )
    return frame, det, obj_world


def test_lidar_localization_recovers_world_position():
    """LiDAR 경로가 물체 world 좌표를 cm 수준으로 되찾는다."""
    frame, det, obj_world = _setup()
    results = localize_object(det, frame)
    assert METHOD_LIDAR in results, results.keys()

    gt_ros = arkit_world_point_to_ros(obj_world)     # 정답도 ROS map으로
    est = results[METHOD_LIDAR].position_world
    err = np.linalg.norm(est - gt_ros)
    assert err < 0.15, f"LiDAR 오차 {err:.3f}m (est={est}, gt={gt_ros})"


def test_plane_method_finds_supporting_plane():
    """관측평면(A′) 경로가 평면을 찾고 그럴듯한 좌표를 낸다."""
    frame, det, obj_world = _setup()
    results = localize_object(det, frame)
    assert METHOD_PLANE in results, results.keys()
    loc = results[METHOD_PLANE]
    assert loc.plane_inlier_ratio is not None and loc.plane_inlier_ratio > 0.5
    gt_ros = arkit_world_point_to_ros(obj_world)
    err = np.linalg.norm(loc.position_world - gt_ros)
    assert err < 0.2, f"평면 경로 오차 {err:.3f}m"


def test_no_depth_returns_empty():
    """depth가 전혀 없으면 위치추정 결과가 비어 있다."""
    T = look_at_pose_arkit(eye=[0, 0.5, 0], target=[0, 0, -1])
    frame = Frame(timestamp=0.0, T_world_camera=T, intrinsics=K)
    det = Detection(label="x", score=0.5, bbox_xyxy=np.array([10, 10, 30, 40]))
    assert localize_object(det, frame) == {}


def test_ground_pixel_uses_bottom():
    """접지점은 bbox 하단 중앙(중심이 아님)이어야 한다."""
    det = Detection(label="x", score=1.0, bbox_xyxy=np.array([10, 20, 30, 60]))
    uv = det.ground_pixel()
    assert np.isclose(uv[0], 20)   # 가로 중앙
    assert np.isclose(uv[1], 60)   # 하단변


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
