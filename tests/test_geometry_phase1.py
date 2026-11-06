"""Phase 1 기하 유틸 테스트 — 역투영 + RANSAC 평면 fitting.

관측기반 표면추정의 핵심 두 함수가 정확한지, 알려진 입력으로 못박는다.

실행:  .venv/bin/python tests/test_geometry_phase1.py
"""

from __future__ import annotations

import numpy as np

from pysical.geometry.backproject import backproject_depth_map, backproject_pixels
from pysical.geometry.plane_fit import Plane, fit_plane_ransac
from pysical.io.types import CameraIntrinsics

K = CameraIntrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0)


# ---------------- 역투영 ----------------

def test_backproject_center_pixel():
    """주점(cx, cy)의 픽셀은 광축 위 → x=y=0, z=-depth (ARKit 규약)."""
    pts = backproject_pixels(np.array([[320.0, 240.0]]), np.array([2.0]), K)
    assert np.allclose(pts[0], [0.0, 0.0, -2.0]), pts[0]


def test_backproject_offset_pixel():
    """주점에서 fx만큼 오른쪽 픽셀은 depth 거리에서 x=depth 만큼 이동."""
    # u = cx + fx = 320 + 600 = 920, depth=1 → x = (920-320)/600*1 = 1.0
    pts = backproject_pixels(np.array([[920.0, 240.0]]), np.array([1.0]), K)
    assert np.allclose(pts[0], [1.0, 0.0, -1.0]), pts[0]


def test_backproject_depth_map_filters_invalid():
    """NaN/0 depth는 point cloud에서 빠진다."""
    depth = np.full((4, 4), np.nan, dtype=np.float32)
    depth[1, 1] = 2.0
    depth[2, 2] = 0.0        # 무효
    depth[3, 3] = 1.5
    pts = backproject_depth_map(depth, K)
    assert pts.shape[0] == 2   # (1,1)과 (3,3)만 유효


# ---------------- 평면 fitting ----------------

def _make_planar_cloud(normal, d, n=300, noise=0.002, seed=1):
    """주어진 평면 위에 노이즈를 얹은 점 구름 + 이상치 몇 개를 만든다."""
    rng = np.random.default_rng(seed)
    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)
    # 평면 위 임의의 두 접선 방향
    a = np.array([1.0, 0.0, 0.0])
    if abs(normal @ a) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    t1 = np.cross(normal, a); t1 /= np.linalg.norm(t1)
    t2 = np.cross(normal, t1)
    # 평면 위 한 점 (n·x + d = 0 을 만족하는 점)
    p0 = -d * normal
    uv = rng.uniform(-0.5, 0.5, size=(n, 2))
    pts = p0 + uv[:, [0]] * t1 + uv[:, [1]] * t2
    pts += rng.normal(0, noise, size=pts.shape)
    # 이상치 20개 (평면에서 멀리)
    outliers = p0 + rng.uniform(-0.5, 0.5, (20, 3)) + normal * 0.3
    return np.vstack([pts, outliers])


def test_fit_horizontal_plane():
    """수평 평면(법선=+Z, 높이 z=-1)을 정확히 복원한다."""
    cloud = _make_planar_cloud(normal=[0, 0, 1], d=1.0)  # z = -1 평면
    plane = fit_plane_ransac(cloud, distance_threshold=0.01, seed=0)
    assert plane is not None
    # 법선은 ±Z 방향이어야 한다(부호는 무관).
    assert abs(abs(plane.normal[2]) - 1.0) < 1e-2, plane.normal
    # 이상치를 뺀 inlier 비율이 높아야 한다(약 300/320).
    assert plane.inlier_ratio > 0.85, plane.inlier_ratio


def test_plane_projection():
    """접지점을 평면에 투영하면 평면 위에 놓인다(잔차 ≈ 0)."""
    plane = Plane(normal=np.array([0.0, 0.0, 1.0]), d=1.0)  # z=-1 평면
    pt = np.array([0.5, 0.3, -0.7])       # 평면에서 0.3 떠 있는 점
    proj = plane.project(pt)
    assert np.isclose(proj[2], -1.0), proj  # z가 평면(-1)에 붙음
    assert abs(plane.signed_distance(proj)[0]) < 1e-9


def test_fit_returns_none_when_too_few_points():
    """점이 너무 적으면 평면을 못 찾고 None."""
    assert fit_plane_ransac(np.zeros((2, 3))) is None


def test_ransac_is_deterministic_with_seed():
    """같은 시드면 같은 평면(재현성)."""
    cloud = _make_planar_cloud(normal=[0, 1, 0.2], d=0.5)
    p1 = fit_plane_ransac(cloud, seed=42)
    p2 = fit_plane_ransac(cloud, seed=42)
    assert p1 is not None and p2 is not None
    assert np.allclose(p1.normal, p2.normal) and np.isclose(p1.d, p2.d)


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
