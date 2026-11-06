"""와이어 프로토콜 왕복(round-trip) 테스트.

encode_frame → decode_frame 을 거쳐도 값이 보존되는지 확인한다. 이게 깨지면
iPhone이 보낸 프레임을 노트북이 잘못 해석하게 되므로, 프로토콜의 안전망이다.

실행:  .venv/bin/python tests/test_protocol.py
"""

from __future__ import annotations

import numpy as np

from pysical.geometry.mock_camera import look_at_pose_arkit, orbit_pose_arkit
from pysical.io.protocol import decode_frame, encode_frame
from pysical.io.types import CameraIntrinsics, Frame


def test_roundtrip_pose_only():
    """RGB 없이 포즈만 담은 프레임의 왕복."""
    T = orbit_pose_arkit(t=3.0)
    frame = Frame(timestamp=3.0, T_world_camera=T, tracking_state="normal")

    decoded = decode_frame(encode_frame(frame))

    assert np.isclose(decoded.timestamp, 3.0)
    assert np.allclose(decoded.T_world_camera, T)
    assert decoded.rgb is None
    assert decoded.tracking_state == "normal"


def test_roundtrip_with_rgb_and_intrinsics():
    """RGB + intrinsics를 담은 프레임의 왕복."""
    T = np.eye(4)
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[:, :, 0] = 200  # 빨강 채널
    intr = CameraIntrinsics(fx=600, fy=600, cx=320, cy=240)
    frame = Frame(timestamp=1.5, T_world_camera=T, rgb=rgb, intrinsics=intr)

    decoded = decode_frame(encode_frame(frame))

    assert decoded.rgb is not None
    assert decoded.rgb.shape == (8, 8, 3)
    # JPEG는 손실 압축이라 정확히 같진 않지만, 빨강이 우세해야 한다.
    assert decoded.rgb[:, :, 0].mean() > decoded.rgb[:, :, 2].mean()
    assert decoded.intrinsics is not None
    assert np.isclose(decoded.intrinsics.fx, 600)


def test_roundtrip_with_lidar_depth():
    """LiDAR depth + confidence + depth_intrinsics의 왕복 (mm 양자화 포함)."""
    T = np.eye(4)
    # 256x192 depth (ARKit 규격), 0.5~3m 범위의 값
    depth = np.linspace(0.5, 3.0, 192 * 256, dtype=np.float32).reshape(192, 256)
    conf = np.full((192, 256), 2, dtype=np.uint8)
    dintr = CameraIntrinsics(fx=240, fy=240, cx=128, cy=96)
    frame = Frame(
        timestamp=2.0,
        T_world_camera=T,
        depth_lidar=depth,
        depth_confidence=conf,
        depth_intrinsics=dintr,
    )

    decoded = decode_frame(encode_frame(frame))

    assert decoded.has_lidar_depth
    assert decoded.depth_lidar.shape == (192, 256)
    # mm 양자화라 오차는 1mm 이내여야 한다.
    assert np.nanmax(np.abs(decoded.depth_lidar - depth)) < 1e-3
    assert decoded.depth_confidence is not None
    assert decoded.depth_confidence[0, 0] == 2
    assert np.isclose(decoded.depth_intrinsics.fx, 240)


def test_depth_zero_becomes_nan():
    """양자화 0(무효)은 디코드 시 NaN이 되어 뒤 단계에서 걸러진다."""
    T = np.eye(4)
    depth = np.zeros((4, 4), dtype=np.float32)
    depth[0, 0] = 1.5  # 하나만 유효
    frame = Frame(timestamp=0.0, T_world_camera=T, depth_lidar=depth)

    decoded = decode_frame(encode_frame(frame))
    assert np.isclose(decoded.depth_lidar[0, 0], 1.5, atol=1e-3)
    assert np.isnan(decoded.depth_lidar[1, 1])


def test_intrinsics_scaling():
    """RGB(640x480) intrinsics를 LiDAR(256x192) 해상도로 스케일링."""
    rgb_intr = CameraIntrinsics(fx=600, fy=600, cx=320, cy=240)
    d_intr = rgb_intr.scaled_to(src_size=(640, 480), dst_size=(256, 192))
    # 가로 256/640=0.4, 세로 192/480=0.4 → 모든 성분이 0.4배
    assert np.isclose(d_intr.fx, 240)
    assert np.isclose(d_intr.cx, 128)
    assert np.isclose(d_intr.cy, 96)


def test_backward_compat_pose_only_still_works():
    """depth 필드를 안 채운 Phase 0 스타일 Frame도 그대로 왕복된다(하위호환)."""
    T = orbit_pose_arkit(t=1.0)
    frame = Frame(timestamp=1.0, T_world_camera=T)
    decoded = decode_frame(encode_frame(frame))
    assert decoded.depth_lidar is None
    assert not decoded.has_any_depth
    assert np.allclose(decoded.T_world_camera, T)


def test_orbit_camera_looks_at_center():
    """궤적 카메라는 항상 원 중심(기본 target=원점)을 바라봐야 한다.

    카메라가 보는 방향(-Z_world)이 eye→target 방향과 일치하는지 확인한다.
    """
    T = orbit_pose_arkit(t=2.0, radius=1.0, height=1.0)
    eye = T[:3, 3]
    forward = -T[:3, 2]                     # ARKit 카메라는 -Z를 본다
    expected = np.array([0.0, 0.0, 0.0]) - eye
    expected /= np.linalg.norm(expected)
    assert np.allclose(forward, expected, atol=1e-6), (forward, expected)


def test_look_at_pose_is_orthonormal():
    """생성된 포즈의 회전부는 정규직교(회전 행렬)여야 한다."""
    T = look_at_pose_arkit(eye=[1, 1, 1], target=[0, 0, 0])
    R = T[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-6)


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
